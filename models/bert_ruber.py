from .header import *
from .base import RetrievalBaseAgent
from .dual_bert import BertEmbedding

class BertRUBER(nn.Module):
    
    '''sigmoid for providing the coherence scores'''
    
    def __init__(self, dropout=0.1, ft=False):
        super(BertRUBER, self).__init__()
        self.ctx_encoder = BertEmbedding()
        self.res_encoder = BertEmbedding()
        self.head = nn.Sequential(
            nn.Linear(768*2, 512),
            nn.Dropout(p=dropout),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )
        self.ft = ft
        self.criterion = nn.BCELoss()
        
    def forward(self, cid, rid, cid_mask, rid_mask):
        ipdb.set_trace()
        batch_size = cid.shape[0]
        if self.ft:
            cid_rep, rid_rep = self.ctx_encoder(cid, cid_mask), self.res_encoder(rid, rid_mask)
        else:
            with torch.no_grad():
                cid_rep, rid_rep = self.ctx_encoder(cid, cid_mask), self.res_encoder(rid, rid_mask)
        # cid_rep, rid_rep: [B, 768]
        neg_random_idx = []
        for i in range(batch_size):
            p = list(range(batch_size))
            p.remove(i)
            neg_random_idx.append(random.choice(p))
        neg_rep = torch.stack([rid_rep[i] for i in neg_random_idx])    # [B, 768]
        cid_rep = torch.cat([cid_rep, cid_rep], dim=0)    # [2*B, 768]
        rid_rep = torch.cat([rid_rep, neg_rep], dim=0)    # [2*B, 768]
        label = torch.tensor([1] * batch_size + [0] * batch_size)
        # shuffle
        random_idx = list(range(batch_size))
        random.shuffle(random_idx)
        cid_rep, rid_rep, label = cid_rep[random_idx], rid_rep[random_idx], label[random_idx]
        rep = torch.cat([cid_rep, rid_rep], dim=1)    # [2*B, 768*2]
        rest = self.head(rep).squeeze(1)     # [2*B, 1] -> [2*B]
        loss = self.criterion(rest, label)
        # acc
        acc_num = ((rest > 0.5) == label).sum().item()
        acc = round(acc_num/batch_size, 4)
        return loss, acc
    
    @torch.no_grad()
    def predict(self, cid, rid, cid_mask, rid_mask):
        batch_size = cid.shape[0]
        cid_rep, rid_rep = self.ctx_encoder(cid, cid_mask), self.res_encoder(rid, rid_mask)
        rep = torch.cat([cid_rep, rid_rep], dim=1)    # [B, 768*2]
        rest = self.head(rep).squeeze(1)     # [B, 1] -> [B]
        return rest
    
class RUBERMetric(RetrievalBaseAgent):
    
    def __init__(self, multi_gpu, run_mode='train', local_rank=0, ft=True):
        super(RUBERMetric, self).__init__()
        try:
            self.gpu_ids = list(range(len(multi_gpu.split(',')))) 
        except:
            raise Exception(f'[!] multi gpu ids are needed, but got: {multi_gpu}')
        self.args = {
            'lr': 2e-5,
            'grad_clip': 1.0,
            'multi_gpu': self.gpu_ids,
            'vocab_file': 'bert-base-chinese',
            'model': 'bert-base-chinese',
            'amp_level': 'O2',
            'local_rank': local_rank,
            'dropout': 0.1,
            'ft': ft,
        }
        self.vocab = BertTokenizer.from_pretrained(self.args['vocab_file'])
        self.model = BertRUBER(dropout=self.args['dropout'], ft=ft)
        self.criterion = nn.BCELoss()
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
            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=[local_rank], output_device=local_rank,
                find_unused_parameters=True,
            )
        elif run_mode == 'test':
            # test mode only need one gpu
            self.model = amp.initialize(self.model, opt_level=self.args['amp_level'])
        pprint.pprint(self.args)
        
    def train_model(self, train_iter, mode='train', recoder=None, idx_=0):
        self.model.train()
        total_loss, total_acc, batch_num = 0, 0, 0
        pbar = tqdm(train_iter)
        for idx, batch in enumerate(pbar):
            self.optimizer.zero_grad()
            loss, acc = self.model(*batch)
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
            clip_grad_norm_(amp.master_params(self.optimizer), self.args['grad_clip'])
            self.optimizer.step()

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
    def inference(self, iter_):
        self.model.eval()
        rest = []
        for batch in tqdm(iter_):
            scores = self.model.predict(*batch).cpu().numpy()    # [B]
            rest.extend(scores)
        score = round(np.mean(rest), 4)
        if self.args['ft']:
            print(f'[!] Bert-RUBER-ft score: {score}')
        else:
            print(f'[!] Bert-RUBER score: {score}')