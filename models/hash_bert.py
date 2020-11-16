from .header import *
from .base import RetrievalBaseAgent
from .dual_bert import BERTBiEncoder

'''Deep Hashing for high-efficient ANN search'''

class HashBERTBiEncoderModel(nn.Module):
    
    def __init__(self, hidden_size, hash_code_size, dropout=0.):
        super(HashBERTBiEncoderModel, self).__init__()
        self.hash_code_size = hash_code_size
        self.encoder = BERTBiEncoder()
        
        self.ctx_hash_encoder = nn.Sequential(
            nn.Linear(768, hidden_size),
            nn.Tanh(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, hash_code_size),
        )
        
        self.ctx_hash_decoder = nn.Sequential(
            nn.Linear(hash_code_size, hidden_size),
            
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, 768),
        )
        
        self.can_hash_encoder = nn.Sequential(
            nn.Linear(768, hidden_size),
            nn.Tanh(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, hash_code_size),
        )
        
        self.can_hash_decoder = nn.Sequential(
            nn.Linear(hash_code_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, 768),
        )
        
    def load_encoder(self, path):
        state_dict = torch.load(path)
        self.encoder.load_state_dict(state_dict)
        print(f'[!] load the dual-bert model parameters successfully')
        
    @torch.no_grad()
    def inference(self, ids, attn_mask):
        rid_rep = self.encoder.can_encoder(ids, attn_mask)
        hash_code = torch.sign(self.can_hash_encoder(rid_rep))
        return hash_code
    
    @torch.no_grad()
    def predict(self, cid, rid, rid_mask):
        batch_size = rid.shape[0]
        cid_rep, rid_rep = self._encode(cid.unsqueeze(0), rid, None, rid_mask)
        cid_rep = cid.squeeze(0)
        # [768]; [B, 768] -> [H]; [B, H]
        ctx_hash_code = torch.sign(self.ctx_hash_encoder(cid_rep))    # [Hash]
        can_hash_code = torch.sign(self.can_hash_encoder(rid_rep))    # [B, Hash]
        matrix = torch.matmul(cid_hash_code, can_hash_code.t())    # [B]
        distance = 0.5 * (self.hash_code_size - matrix)    # hamming distance: ||b_i, b_j||_{H} = 0.5 * (K - b_i^Tb_j); distance: [B]
        return distance
        
    def forward(self, cid, rid, cid_mask, rid_mask):
        '''do we need the dot production loss? In my opinion, the hash loss is the replaction of the dot production loss. But need the experiment results to show it.'''
        batch_size = cid.shape[0]
        assert batch_size > 1, f'[!] batch size must bigger than 1, cause other elements in the batch will be seen as the negative samples'
        
        with torch.no_grad():
            # cid_rep/rid_rep: [B, 768]
            cid_rep, rid_rep = self.encoder._encode(cid, rid, cid_mask, rid_mask)
        
        # Hash function
        ctx_hash_code = self.ctx_hash_encoder(cid_rep)    # [B, Hash]
        can_hash_code = self.can_hash_encoder(rid_rep)    # [B, Hash]
        cid_rep_recons = self.ctx_hash_decoder(ctx_hash_code)    # [B, 768]
        rid_rep_recons = self.can_hash_decoder(can_hash_code)    # [B, 768]
        
        # ===== calculate preserved loss ===== #
        preserved_loss = torch.norm(cid_rep_recons - cid_rep, p=2, dim=1).mean() + torch.norm(rid_rep_recons - rid_rep, p=2, dim=1).mean()
        # preserved_loss = self.criterion(cid_rep_recons, cid_rep) + self.criterion(rid_rep_recons, rid_rep)
        
        # ===== calculate quantization loss ===== #
        ctx_hash_code_h, can_hash_code_h = torch.sign(ctx_hash_code), torch.sign(can_hash_code)
        quantization_loss = torch.norm(ctx_hash_code - ctx_hash_code_h, p=2, dim=1).mean() + torch.norm(can_hash_code - can_hash_code_h, p=2, dim=1).mean()
        
        # ===== calculate hamming distance ===== #
        matrix = torch.matmul(ctx_hash_code_h, can_hash_code_h.t())    # [B, B]
        hamming_distance = 0.5 * (self.hash_code_size - matrix)    # hamming distance: ||b_i, b_j||_{H} = 0.5 * (K - b_i^Tb_j); [B, B]
        
        acc_num = (hamming_distance.min(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        acc = acc_num / batch_size
        return acc, preserved_loss, quantization_loss

class HashModelAgent(RetrievalBaseAgent):
    
    def __init__(self, multi_gpu, total_step, run_mode='train', local_rank=0, path=None):
        super(HashModelAgent, self).__init__()
        try:
            self.gpu_ids = list(range(len(multi_gpu.split(',')))) 
        except:
            raise Exception(f'[!] multi gpu ids are needed, but got: {multi_gpu}')
        self.args = {
            'lr': 5e-3,
            'grad_clip': 1.0,
            'multi_gpu': self.gpu_ids,
            'local_rank': local_rank,
            'dropout': 0.1,
            'hidden_size': 512,
            'hash_code_size': 128,
            'total_steps': total_step,
            'samples': 10,
            'amp_level': 'O2',
            'path': path,
            'q_alpha': 0, #s1e-3,
        }
        
        self.model = HashBERTBiEncoderModel(
            self.args['hidden_size'], 
            self.args['hash_code_size'],
            dropout=self.args['dropout'],
        )
        self.model.load_encoder(self.args['path'])
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
        elif run_mode == 'inferance':
            self.model = amp.initialize(self.model, opt_level=self.args['amp_level'])
            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=[local_rank], output_device=local_rank,
                find_unused_parameters=True,
            )
        pprint.pprint(self.args)
        
    def train_model(self, train_iter, mode='train', recoder=None, idx_=0):
        self.model.train()
        total_acc, total_loss, total_p_loss, total_q_loss, batch_num = 0, 0, 0, 0, 0
        pbar = tqdm(train_iter)
        for idx, batch in enumerate(pbar):
            self.optimizer.zero_grad()
            cid, rid, cid_mask, rid_mask = batch
            acc, preserved_loss, quantization_loss = self.model(
                cid, rid, cid_mask, rid_mask,
            )
            quantization_loss = self.args['q_alpha'] * quantization_loss
            loss = preserved_loss + quantization_loss
            loss.backward()
#             clip_grad_norm_(self.model.parameters(), self.args['grad_clip'])
            self.optimizer.step()

            total_loss += loss.item()
            total_acc += acc
            total_p_loss += preserved_loss.item()
            total_q_loss += quantization_loss.item()
            # total_h_loss += hash_loss.item()
            batch_num += 1
            
            recoder.add_scalar(f'train-epoch-{idx_}/PreservedLoss', total_p_loss/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunPreservedLoss', preserved_loss.item(), idx)
            recoder.add_scalar(f'train-epoch-{idx_}/QuantizationLoss', total_q_loss/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunQuantizationLoss', quantization_loss.item(), idx)
            recoder.add_scalar(f'train-epoch-{idx_}/Loss', total_loss/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunLoss', loss.item(), idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunAcc', acc, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/Acc', total_acc/batch_num, idx)
            
            pbar.set_description(f'[!] loss(p|q|t): {round(total_p_loss/batch_num, 4)}|{round(total_q_loss/batch_num, 4)}|{round(total_loss/batch_num, 4)}; acc: {round(acc, 4)}|{round(total_acc/batch_num, 4)}')
        
        recoder.add_scalar(f'train-whole/Loss', total_loss/batch_num, idx_)
        recoder.add_scalar(f'train-whole/PreservedLoss', total_p_loss/batch_num, idx_)
        recoder.add_scalar(f'train-whole/QuantizationLoss', total_q_loss/batch_num, idx_)
        recoder.add_scalar(f'train-whole/Acc', total_acc/batch_num, idx_)
        return round(total_loss / batch_num, 4)
    
    @torch.no_grad()
    def test_model(self, test_iter, mode='test', recoder=None, idx_=0):
        '''replace the dot production with the hamming distance calculated by the hash encoder'''
        self.model.eval()
        r1, r2, r5, r10, counter, mrr = 0, 0, 0, 0, 0, []
        pbar = tqdm(test_iter)
        for idx, batch in tqdm(list(enumerate(pbar))):                
            cid, rids, rids_mask = batch
            batch_size = len(rids)
            if batch_size != self.args['samples']:
                continue
            hamming_distance = self.model.predict(cid, rids, rids_mask).cpu()    # [B]
            r1 += (torch.topk(hamming_distance, 1, dim=-1)[1] == 0).sum().item()
            r2 += (torch.topk(hamming_distance, 2, dim=-1)[1] == 0).sum().item()
            r5 += (torch.topk(hamming_distance, 5, dim=-1)[1] == 0).sum().item()
            r10 += (torch.topk(hamming_distance, 10, dim=-1)[1] == 0).sum().item()
            preds = torch.argsort(hamming_distance, dim=-1).tolist()    # [B, B]
            # mrr
            hamming_distance = hamming_distance.numpy()
            y_true = np.zeros(len(hamming_distance))
            y_true[0] = 1
            mrr.append(label_ranking_average_precision_score([y_true], [hamming_distance]))
            counter += 1
            
        r1, r2, r5, r10, mrr = round(r1/counter, 4), round(r2/counter, 4), round(r5/counter, 4), round(r10/counter, 4), round(np.mean(mrr), 4)
        print(f'r1@10: {r1}; r2@10: {r2}; r5@10: {r5}; r10@10: {r10}; mrr: {mrr}')
        
    @torch.no_grad()
    def inference(self, iter_, path):
        '''only use the response encoder'''
        self.model.eval()
        pbar = tqdm(list(enumerate(iter_)))
        collections_matrix, collections_text = [], []
        for idx, batch in pbar:
            ids, attn_mask, texts = batch
            rep = self.model.inference(ids, attn_mask).cpu().numpy()    # [B, E]
            collections_matrix.append(rep)
            collections_text.extend(texts)
        collections_matrix = np.concatenate(collections_matrix)    # [S, E]
        torch.save((collections_matrix, collections_text), path)
        print(f'[!] inference all the utterances over and save into {path}')