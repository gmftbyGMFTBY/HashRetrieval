from .header import *
from .base import RetrievalBaseAgent

class BERTRetrieval(nn.Module):

    def __init__(self, model='bert-base-chinese'):
        super(BERTRetrieval, self).__init__()
        self.model = BertForSequenceClassification.from_pretrained(model, num_labels=2)

    def forward(self, inpt, token_type_ids, attn_mask):
        output = self.model(
            input_ids=inpt,
            attention_mask=attn_mask,
            token_type_ids=token_type_ids,
        )
        logits = output[0]    # [batch, 2]
        return logits
    
class BERTRetrievalAgent(RetrievalBaseAgent):

    def __init__(self, multi_gpu, total_step, run_mode='train', local_rank=0):
        super(BERTRetrievalAgent, self).__init__()
        try:
            self.gpu_ids = list(range(len(multi_gpu.split(',')))) 
        except:
            raise Exception(f'[!] multi gpu ids are needed, but got: {multi_gpu}')
        self.args = {
            'lr': 5e-5,
            'grad_clip': 1.0,
            'samples': 10,
            'multi_gpu': self.gpu_ids,
            'talk_samples': 256,
            'max_len': 256,
            'vocab_file': 'bert-base-chinese',
            'model': 'bert-base-chinese',
            'amp_level': 'O2',
            'local_rank': local_rank,
            'warmup_steps': int(0.1 * total_step),
            'total_step': total_step,
        }
        self.vocab = BertTokenizer.from_pretrained(self.args['vocab_file'])
        self.model = BERTRetrieval(self.args['model'])
        if torch.cuda.is_available():
            self.model.cuda()
        if run_mode == 'train':
            self.optimizer = transformers.AdamW(
                self.model.parameters(), 
                lr=self.args['lr'],
            )
            self.criterion = nn.CrossEntropyLoss()
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
        else:
            # faster
            self.model, self.optimizer = amp.initialize(
                self.model, 
                self.optimizer,
                opt_level=self.args['amp_level'],
            )
            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=[local_rank], output_device=local_rank,
                find_unused_parameters=True,
            )
        pprint.pprint(self.args)

    def train_model(self, train_iter, mode='train', recoder=None, idx_=0):
        self.model.train()
        total_loss, batch_num = 0, 0
        pbar = tqdm(train_iter)
        correct, s = 0, 0
        for idx, batch in enumerate(pbar):
            cids, tids, attn_mask, label = batch
            self.optimizer.zero_grad()
            output = self.model(cids, tids, attn_mask)    # [B, 2]
            loss = self.criterion(output, label.view(-1))
            
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
            clip_grad_norm_(amp.master_params(self.optimizer), self.args['grad_clip'])
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            batch_num += 1
            
            now_correct = torch.max(F.softmax(output, dim=-1), dim=-1)[1]    # [batch]
            now_correct = torch.sum(now_correct == label).item()
            correct += now_correct
            s += len(label)
            
            recoder.add_scalar(f'train-epoch-{idx_}/Loss', total_loss/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunLoss', loss.item(), idx)
            recoder.add_scalar(f'train-epoch-{idx_}/Acc', correct/s, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunAcc', now_correct/len(label), idx)

            pbar.set_description(f'[!] train loss: {round(loss.item(), 4)}|{round(total_loss/batch_num, 4)}; acc: {round(now_correct/len(label), 4)}|{round(correct/s, 4)}')
        recoder.add_scalar(f'train-whole/Loss', total_loss/batch_num, idx_)
        recoder.add_scalar(f'train-whole/Acc', correct/s, idx_)
        return round(total_loss / batch_num, 4)

    @torch.no_grad()
    def test_model(self, test_iter):
        self.model.eval()
        r1, r2, r5, r10, counter, mrr = 0, 0, 0, 0, 0, []
        pbar = tqdm(test_iter)
        rest = []
        for idx, batch in tqdm(enumerate(pbar)):
            cid, tid, attn_mask = batch
            output = self.model(cid, tid, attn_mask)    # [batch, 2]
            output = F.softmax(output, dim=-1)[:, 1]    # [batch]
            preds = [i for i in torch.split(output, self.args['samples'])]
            for pred in preds:
                # pred: [samples]
                r1 += (torch.topk(pred, 1, dim=-1)[1] == 0).sum().item()
                r2 += (torch.topk(pred, 2, dim=-1)[1] == 0).sum().item()
                r5 += (torch.topk(pred, 5, dim=-1)[1] == 0).sum().item()
                r10 += (torch.topk(pred, 10, dim=-1)[1] == 0).sum().item()
                preds = torch.argsort(pred, dim=-1).tolist()    # [B, B]
                # mrr
                pred = pred.numpy()
                y_true = np.zeros(len(pred))
                y_true[0] = 1
                mrr.append(label_ranking_average_precision_score([y_true], [pred]))
                counter += 1
        r1, r2, r5, r10, mrr = round(r1/counter, 4), round(r2/counter, 4), round(r5/counter, 4), round(r10/counter, 4), round(np.mean(mrr), 4)
        print(f'r1@10: {r1}; r2@10: {r2}; r5@10: {r5}; r10@10: {r10}; mrr: {mrr}')