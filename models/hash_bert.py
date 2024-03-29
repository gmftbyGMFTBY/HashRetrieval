from .header import *
from .base import RetrievalBaseAgent
from .dual_bert import BertEmbedding

'''Deep Hashing for high-efficient ANN search'''

class DualBert(nn.Module):
    
    '''dual bert only for inference'''
    
    def __init__(self):
        super(DualBert, self).__init__()
        self.ctx_encoder = BertEmbedding()
        self.can_encoder = BertEmbedding()
       
    @torch.no_grad()
    def forward(self, cid, rid, cid_mask, rid_mask):
        self.ctx_encoder.eval()
        self.can_encoder.eval()
        cid_rep = self.ctx_encoder(cid, cid_mask)
        rid_rep = self.can_encoder(rid, rid_mask)
        return cid_rep, rid_rep
    
    @torch.no_grad()
    def get_q(self, ids, mask):
        self.ctx_encoder.eval()
        cid_rep = self.ctx_encoder(ids, mask)
        return cid_rep
    
    @torch.no_grad()
    def get_r(self, ids, mask):
        self.can_encoder.eval()
        rid_rep = self.can_encoder(ids, mask)
        return rid_rep

class HashBERTBiEncoderModel(nn.Module):
    
    def __init__(self, hidden_size, hash_code_size, dropout=0.1):
        super(HashBERTBiEncoderModel, self).__init__()
        self.hash_code_size = hash_code_size
        
        self.ctx_hash_encoder = nn.Sequential(
            nn.Linear(768, hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, hash_code_size),
        )
        
        self.ctx_hash_decoder = nn.Sequential(
            nn.Linear(hash_code_size, hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, 768),
        )
        
        self.can_hash_encoder = nn.Sequential(
            nn.Linear(768, hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, hash_code_size),
        )
        
        self.can_hash_decoder = nn.Sequential(
            nn.Linear(hash_code_size, hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, 768),
        )
        
    @torch.no_grad()
    def inference(self, rid_rep):
        self.can_hash_encoder.eval()
        hash_code = torch.sign(self.can_hash_encoder(rid_rep))
        hash_code = torch.where(
            hash_code == -1, 
            torch.zeros_like(hash_code), 
            hash_code,
        )
        return hash_code
    
    @torch.no_grad()
    def get_q(self, cid_rep):
        self.ctx_hash_encoder.eval()
        hash_code = torch.sign(self.ctx_hash_encoder(cid_rep))
        hash_code = torch.where(
            hash_code == -1, 
            torch.zeros_like(hash_code), 
            hash_code,
        )
        return hash_code
    
    @torch.no_grad()
    def predict(self, cid, rid, rid_mask):
        self.ctx_hash_encoder.eval()
        self.can_hash_encoder.eval()
        batch_size = rid.shape[0]
        cid_rep, rid_rep = self._encode(cid.unsqueeze(0), rid, None, rid_mask)
        cid_rep = cid.squeeze(0)
        # [768]; [B, 768] -> [H]; [B, H]
        ctx_hash_code = torch.sign(self.ctx_hash_encoder(cid_rep))    # [Hash]
        can_hash_code = torch.sign(self.can_hash_encoder(rid_rep))    # [B, Hash]
        matrix = torch.matmul(cid_hash_code, can_hash_code.t())    # [B]
        distance = 0.5 * (self.hash_code_size - matrix)    # hamming distance: ||b_i, b_j||_{H} = 0.5 * (K - b_i^Tb_j); distance: [B]
        
    def forward(self, cid_rep, rid_rep):
        batch_size = cid_rep.shape[0]
        assert batch_size > 1, f'[!] batch size must bigger than 1, cause other elements in the batch will be seen as the negative samples'
        
        # Hash function
        ctx_hash_code = self.ctx_hash_encoder(cid_rep)    # [B, Hash]
        can_hash_code = self.can_hash_encoder(rid_rep)    # [B, Hash]
        cid_rep_recons = self.ctx_hash_decoder(ctx_hash_code)    # [B, 768]
        rid_rep_recons = self.can_hash_decoder(can_hash_code)    # [B, 768]
        
        # ===== calculate preserved loss ===== #
        preserved_loss = torch.norm(cid_rep_recons - cid_rep, p=2, dim=1).mean() + torch.norm(rid_rep_recons - rid_rep, p=2, dim=1).mean()
        
        # ===== calculate quantization loss ===== #
        ctx_hash_code_h, can_hash_code_h = torch.sign(ctx_hash_code).detach(), torch.sign(can_hash_code).detach()
        quantization_loss = torch.norm(ctx_hash_code - ctx_hash_code_h, p=2, dim=1).mean() + torch.norm(can_hash_code - can_hash_code_h, p=2, dim=1).mean()
        
        # ===== calculate hash loss ===== #
        matrix = torch.matmul(ctx_hash_code, can_hash_code.T)    # [B, B] similarity matrix
        label_matrix = self.hash_code_size * torch.eye(batch_size).cuda()
        hash_loss = torch.norm(matrix - label_matrix, p=2).mean()
        
        # ===== calculate hamming distance for accuracy ===== #
        matrix = torch.matmul(ctx_hash_code_h, can_hash_code_h.t())    # [B, B]
        hamming_distance = 0.5 * (self.hash_code_size - matrix)    # hamming distance: ||b_i, b_j||_{H} = 0.5 * (K - b_i^Tb_j); [B, B]
        acc_num = (hamming_distance.min(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        acc = acc_num / batch_size
        
        # ===== calculate reference acc ===== #
        dot_product = torch.matmul(cid_rep, rid_rep.t())  # [B, B]
        ref_acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        ref_acc = ref_acc_num / batch_size
        
        return acc, ref_acc, hash_loss, preserved_loss, quantization_loss

class HashModelAgent(RetrievalBaseAgent):
    
    def __init__(self, hash_code_size, neg_samples, multi_gpu, total_step, run_mode='train', local_rank=0, path=None):
        super(HashModelAgent, self).__init__()
        try:
            self.gpu_ids = list(range(len(multi_gpu.split(',')))) 
        except:
            raise Exception(f'[!] multi gpu ids are needed, but got: {multi_gpu}')
        self.args = {
            'lr': 5e-4,
            'grad_clip': 1.0,
            'multi_gpu': self.gpu_ids,
            'local_rank': local_rank,
            'dropout': 0.1,
            'hidden_size': 512,
            'hash_code_size': hash_code_size,
            'neg_samples': neg_samples,
            'total_steps': total_step,
            'samples': 10,
            'amp_level': 'O2',
            'path': path,
            'q_alpha': 1e-4,
            'q_alpha_max': 1e-1,
        }
        
        self.model = HashBERTBiEncoderModel(
            self.args['hidden_size'],
            self.args['hash_code_size'],
            dropout=self.args['dropout'],
        )
        self.bert_encoder = DualBert()
        self.load_encoder(self.args['path'])
        if torch.cuda.is_available():
            self.model.cuda()
            self.bert_encoder.cuda()
        if run_mode == 'train':
            self.args['q_alpha_step'] = (self.args['q_alpha_max'] - self.args['q_alpha']) / int(total_step / len(self.gpu_ids))
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.args['lr'],
            )
            # ===== only bert model need the apex, hash module cannot use apex because apex make the gradient losing ===== #
            self.bert_encoder = amp.initialize(
                self.bert_encoder,
                opt_level=self.args['amp_level'],
            )
            self.bert_encoder = nn.parallel.DistributedDataParallel(
                self.bert_encoder, 
                device_ids=[local_rank], 
                output_device=local_rank,
                find_unused_parameters=True,
            )
            self.model = nn.parallel.DistributedDataParallel(
                self.model, 
                device_ids=[local_rank], 
                output_device=local_rank,
                find_unused_parameters=True,
            )
        elif run_mode == 'inferance':
            self.bert_encoder = amp.initialize(self.bert_encoder, opt_level=self.args['amp_level'])
            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=[local_rank], 
                output_device=local_rank,
                find_unused_parameters=True,
            )
            self.bert_encoder = nn.parallel.DistributedDataParallel(
                self.bert_encoder, 
                device_ids=[local_rank], 
                output_device=local_rank,
                find_unused_parameters=True,
            )
        pprint.pprint(self.args)
        
    def load_model(self, path):
        # load the bert encoder
        self.load_encoder(self.args['path'])
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        try:
            self.model.module.load_state_dict(state_dict)
        except:
            self.model.load_state_dict(state_dict)
        print(f'[!] load model from {path}')
        
    def load_encoder(self, path):
        '''MUST LOAD TO CPU FIRST'''
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        self.bert_encoder.load_state_dict(state_dict)
        print(f'[!] load the dual-bert model parameters successfully')
        
    def train_model(self, train_iter, mode='train', recoder=None, idx_=0):
        self.model.train()
        total_acc, total_ref_acc, total_loss, total_h_loss, total_p_loss, total_q_loss, batch_num = 0, 0, 0, 0, 0, 0, 0
        pbar = tqdm(train_iter)
        for idx, batch in enumerate(pbar):
            self.optimizer.zero_grad()
            cid, rid, cid_mask, rid_mask = batch
            cid_rep, rid_rep = self.bert_encoder(cid, rid, cid_mask, rid_mask)
            acc, ref_acc, hash_loss, preserved_loss, quantization_loss = self.model(
                cid_rep, rid_rep,
            )
            quantization_loss = self.args['q_alpha'] * quantization_loss
            loss = hash_loss + preserved_loss + quantization_loss
            loss.backward()
            clip_grad_norm_(self.model.parameters(), self.args['grad_clip'])
            self.optimizer.step()
            self.args['q_alpha'] += self.args['q_alpha_step']

            total_loss += loss.item()
            total_acc += acc
            total_ref_acc += ref_acc
            total_p_loss += preserved_loss.item()
            total_q_loss += quantization_loss.item()
            total_h_loss += hash_loss.item()
            batch_num += 1
            
            recoder.add_scalar(f'train-epoch-{idx_}/PreservedLoss', total_p_loss/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunPreservedLoss', preserved_loss.item(), idx)
            recoder.add_scalar(f'train-epoch-{idx_}/QuantizationLoss', total_q_loss/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunQuantizationLoss', quantization_loss.item(), idx)
            recoder.add_scalar(f'train-epoch-{idx_}/HashLoss', total_h_loss/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunHashLoss', hash_loss.item(), idx)
            recoder.add_scalar(f'train-epoch-{idx_}/Loss', total_loss/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunLoss', loss.item(), idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunAcc', acc, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/Acc', total_acc/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunRefAcc', ref_acc, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RefAcc', total_ref_acc/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/QuantizationAlpha', self.args['q_alpha'], idx)
            
            pbar.set_description(f'[!] q_weight: {round(self.args["q_alpha"], 4)}; loss(p|q|h|t): {round(total_p_loss/batch_num, 4)}|{round(total_q_loss/batch_num, 4)}|{round(total_h_loss/batch_num, 4)}|{round(total_loss/batch_num, 4)}; acc(acc|ref_acc): {round(total_acc/batch_num, 4)}|{round(total_ref_acc/batch_num, 4)}')
        
        recoder.add_scalar(f'train-whole/Loss', total_loss/batch_num, idx_)
        recoder.add_scalar(f'train-whole/PreservedLoss', total_p_loss/batch_num, idx_)
        recoder.add_scalar(f'train-whole/QuantizationLoss', total_q_loss/batch_num, idx_)
        recoder.add_scalar(f'train-whole/HashLoss', total_h_loss/batch_num, idx_)
        recoder.add_scalar(f'train-whole/Acc', total_acc/batch_num, idx_)
        recoder.add_scalar(f'train-whole/RefAcc', total_ref_acc/batch_num, idx_)
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
            rid_rep = self.bert_encoder.get_r(ids, attn_mask)
            rep = self.model.inference(rid_rep).cpu().numpy()    # [B, E]
            collections_matrix.append(rep)
            collections_text.extend(texts)
        collections_matrix = np.concatenate(collections_matrix).astype('int')    # [S, E]
        
        # nbits * 8
        assert self.args['hash_code_size'] % 8 == 0
        collections_matrix = np.split(collections_matrix, int(self.args['hash_code_size']/8), axis=1)
        collections_matrix = np.ascontiguousarray(
            np.stack(
                [np.packbits(i) for i in collections_matrix]
            ).transpose().astype('uint8')
        )
        
        torch.save((collections_matrix, collections_text), path)
        print(f'[!] inference all the utterances over and save into {path}')