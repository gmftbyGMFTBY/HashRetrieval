from .header import *

class BertEmbedding(nn.Module):
    
    '''squeeze strategy: 1. first; 2. first-m; 3. average'''
    
    def __init__(self, m=0, lang='zh'):
        super(BertEmbedding, self).__init__()
        model_name = 'bert-base-chinese' if lang == 'zh' else 'bert-base-uncased'
        self.model = BertModel.from_pretrained(model_name)
        self.m = m

    def forward(self, ids, attn_mask, strategy='first'):
        '''convert ids to embedding tensor; Return: [B, 768]'''
        embd = self.model(ids, attention_mask=attn_mask)[0]    # [B, S, 768]
        if strategy == 'first':
            rest = embd[:, 0, :]
        elif strategy == 'first-m':
            rest = embd[:, :self.m, :]    # [B, M, 768]
        elif strategy == 'average':
            rest = embd.mean(dim=1)    # [B, 768]
        else:
            raise Exception(f'[!] Unknow squeeze strategy {self.squeeze_strategy}')
        return rest
    
class BERTBiEncoder(nn.Module):
    
    '''During training, the other elements in the batch are seen as the negative samples, which will lead to the fast training speed. More details can be found in paper: https://arxiv.org/pdf/1905.01969v2.pdf
    reference: https://github.com/chijames/Poly-Encoder/blob/master/encoder.py
    '''
    
    def __init__(self, lang='zh'):
        super(BERTBiEncoder, self).__init__()
        self.ctx_encoder = BertEmbedding(lang=lang)
        self.can_encoder = BertEmbedding(lang=lang)
        
    def _encode(self, cid, rid, cid_mask, rid_mask):
        cid_rep = self.ctx_encoder(cid, cid_mask)
        rid_rep = self.can_encoder(rid, rid_mask)
        return cid_rep, rid_rep
    
    @torch.no_grad()
    def predict(self, cid, rid, rid_mask):
        batch_size = rid.shape[0]
        cid_rep, rid_rep = self._encode(cid.unsqueeze(0), rid, None, rid_mask)
        cid_rep = cid_rep.squeeze(0)    # [768]
        # cid_rep/rid_rep: [768], [B, 768]
        dot_product = torch.matmul(cid_rep, rid_rep.t())  # [B]
        return dot_product
        
    def forward(self, cid, rid, cid_mask, rid_mask):
        batch_size = cid.shape[0]
        assert batch_size > 1, f'[!] batch size must bigger than 1, cause other elements in the batch will be seen as the negative samples'
        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
        # cid_rep/rid_rep: [B, 768]
        dot_product = torch.matmul(cid_rep, rid_rep.t())  # [B, B]
        # use half for supporting the apex
        mask = to_cuda(torch.eye(batch_size)).half()    # [B, B]
        # calculate accuracy
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        acc = acc_num / batch_size
        # calculate the loss
        loss = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss.sum(dim=1)).mean()
        return loss, acc
    
class BERTBiEncoderAgent:
    
    def __init__(self, multi_gpu, total_step, run_mode='train', local_rank=0):
        try:
            self.gpu_ids = list(range(len(multi_gpu.split(',')))) 
        except:
            raise Exception(f'[!] multi gpu ids are needed, but got: {multi_gpu}')
        self.args = {
            'lr': 5e-5,
            'grad_clip': 1.0,
            'multi_gpu': self.gpu_ids,
            'talk_samples': 256,
            'vocab_file': 'bert-base-chinese',
            'samples': 10,
            'model': 'bert-base-chinese',,
            'amp_level': 'O2',
            'local_rank': local_rank,
            'warmup_steps': int(0.1 * total_step),
            'total_step': total_step,
        }
        self.vocab = BertTokenizer.from_pretrained(self.args['vocab_file'])
        self.model = BERTBiEncoder(lang=self.args['lang'],)
        if torch.cuda.is_available():
            self.model.cuda()
        if run_mode == 'train':
            self.optimizer = transformers.AdamW(
                self.model.parameters(), 
                lr=self.args['lr'],
            )
            self.model, self.optimizer = amp.initialize(
                self.model, 
                self.optimizer,
                opt_level=self.args['amp_level'],
            )
            self.scheduler = transformers.get_linear_schedule_with_warmup(
                self.optimizer, 
                num_warmup_steps=self.args['warmup_steps'], 
                num_training_steps=total_step,
            )
            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=[local_rank], output_device=local_rank,
                find_unused_parameters=True,
            )
        print(self.args)
        
    def train_model(self, train_iter, mode='train', recoder=None, idx_=0):
        self.model.train()
        total_loss, total_acc, batch_num = 0, 0, 0
        pbar = tqdm(train_iter)
        correct, s = 0, 0
        for idx, batch in enumerate(pbar):
            self.optimizer.zero_grad()
            cid, rid, cid_mask, rid_mask = batch
            loss, acc = self.model(cid, rid, cid_mask, rid_mask)
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
            clip_grad_norm_(amp.master_params(self.optimizer), self.args['grad_clip'])
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            total_acc += acc
            batch_num += 1
            
            recoder.add_scalar(f'train-epoch-{idx_}/Loss', total_loss/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunLoss', loss.item(), idx)
            recoder.add_scalar(f'train-epoch-{idx_}/Acc', total_acc/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunAcc', acc, idx)

            pbar.set_description(f'[!] loss: {round(loss.item(), 4)}|{round(total_loss/batch_num, 4)}; acc: {round(acc, 4)}|{round(total_acc/batch_num, 4)}')
        recoder.add_scalar(f'train-whole/Loss', total_loss/batch_num, idx_)
        recoder.add_scalar(f'train-whole/Acc', total_acc/batch_num, idx_)
        return round(total_loss / batch_num, 4)
        
    @torch.no_grad()
    def test_model(self, test_iter, mode='test', recoder=None, idx_=0):
        '''there is only one context in the batch, and response are the candidates that are need to reranked; batch size is the self.args['samples']; the groundtruth is the first one. For douban300w and E-Commerce datasets'''
        self.model.eval()
        r1, r2, r5, r10, counter, mrr = 0, 0, 0, 0, 0, []
        pbar = tqdm(test_iter)
        for idx, batch in tqdm(list(enumerate(pbar))):                
            cid, rids, rids_mask = batch
            batch_size = len(rids)
            if batch_size != self.args['samples']:
                continue
            dot_product = self.model.predict(cid, rids, rids_mask).cpu()    # [B]
            r1 += (torch.topk(dot_product, 1, dim=-1)[1] == 0).sum().item()
            r2 += (torch.topk(dot_product, 2, dim=-1)[1] == 0).sum().item()
            r5 += (torch.topk(dot_product, 5, dim=-1)[1] == 0).sum().item()
            r10 += (torch.topk(dot_product, 10, dim=-1)[1] == 0).sum().item()
            preds = torch.argsort(dot_product, dim=-1).tolist()    # [B, B]
            # mrr
            dot_product = dot_product.numpy()
            y_true = np.zeros(len(dot_product))
            y_true[0] = 1
            mrr.append(label_ranking_average_precision_score([y_true], [dot_product]))
            counter += 1
            
        r1, r2, r5, r10, mrr = round(r1/counter, 4), round(r2/counter, 4), round(r5/counter, 4), round(r10/counter, 4), round(np.mean(mrr), 4)
        print(f'r1@10: {r1}; r2@10: {r2}; r5@10: {r5}; r10@10: {r10}; mrr: {mrr}')
    
    @torch.no_grad()
    def talk(self, msgs, topic=None):
        self.model.eval()
        utterances, inpt_ids, res_ids, attn_mask = self.process_utterances_biencoder(topic, msgs, max_len=self.args['max_len'])
        output = self.model.predict(inpt_ids, res_ids, attn_mask)    # [B]
        item = torch.argmax(output).item()
        msg = utterances[item]
        return msg