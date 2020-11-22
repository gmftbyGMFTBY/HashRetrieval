from header import *
from models import *
from utils import *
from dataloader import *

def parser_args():
    parser = argparse.ArgumentParser(description='chat parameters')
    parser.add_argument('--dataset', default='zh50w', type=str)
    parser.add_argument('--coarse', type=str, help='es; hash; dense')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--seed', type=float, default=30)
    parser.add_argument('--max_len', type=int, default=300)
    parser.add_argument('--topk', type=int, default=20)
    parser.add_argument('--test_mode', type=str, default='coarse')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--dimension', type=int, default=768)
    return parser.parse_args()

class HashVectorCoarseRetrieval:
    
    '''dual-bert or hash-bert'''
    
    def __init__(self, dataset, path1, path2, topk=20, max_len=256, gpu=-1, hash_code_size=128):
        self.vocab = BertTokenizer.from_pretrained('bert-base-chinese')
        self.bert_model = DualBert()
        self.model = HashBERTBiEncoderModel(512, hash_code_size)
        self._load(
            f'ckpt/{args["dataset"]}/hash-bert/best.pt',
            f'ckpt/{args["dataset"]}/dual-bert/best.pt',
        )
        if torch.cuda.is_available():
            self.model.cuda()
            self.bert_model.cuda()
        self.topk, self.max_len, self.pad, self.hash_code_size = topk, max_len, self.vocab.pad_token_id, hash_code_size
        
        self.searcher = Searcher(
            dataset, vector=True, binary=True, dimension=hash_code_size, gpu=gpu,
        )
        self.searcher.load(path1, path2)
        
    def _load(self, path1, path2):
        self.model.load_state_dict(torch.load(path1))
        self.bert_model.load_state_dict(torch.load(path2))
        print(f'[!] load model from {path1} and {path2}')
        
    def _length_limit(self, ids):
        if len(ids) > self.max_len:
            ids = [ids[0]] + ids[-(self.max_len-1):]
        return ids
    
    def generate_mask(self, ids):
        attn_mask_index = ids.nonzero().tolist()
        attn_mask_index_x, attn_mask_index_y = [i[0] for i in attn_mask_index], [i[1] for i in attn_mask_index]
        attn_mask = torch.zeros_like(ids)
        attn_mask[attn_mask_index_x, attn_mask_index_y] = 1
        return attn_mask
        
    def _tokenize(self, msgs):
        cids = [torch.LongTensor(self._length_limit(i)) for i in self.vocab.batch_encode_plus(msgs)['input_ids']]
        cids = pad_sequence(cids, batch_first=True, padding_value=self.pad)
        mask = self.generate_mask(cids)
        if torch.cuda.is_available():
            cids, mask = cids.cuda(), mask.cuda()
        return cids, mask
    
    @torch.no_grad()
    def _encode(self, msgs):
        self.model.eval()
        cids, mask = self._tokenize(msgs)
        cid_rep = self.bert_model.get_q(cids, mask)    # [B, 768]
        queries = self.model.get_q(cid_rep).cpu().numpy().astype('int')    # [B, hash]
        
        # nbits * 8
        assert self.hash_code_size % 8 == 0
        queries = np.split(queries, int(self.hash_code_size/8), axis=1)
        queries = np.ascontiguousarray(
            np.stack(
                [np.packbits(i) for i in queries]
            ).transpose().astype('uint8')
        )
        return queries
            
    def search(self, msgs):
        '''a batch of queries'''
        queries = self._encode(msgs)
        rest, t = self.searcher.search(queries, topk=self.topk)
        return rest, t

class DenseVectorCoarseRetrieval:
    
    '''dual-bert or hash-bert'''
    
    def __init__(self, dataset, path1, path2, topk=20, max_len=256, gpu=-1):
        self.vocab = BertTokenizer.from_pretrained('bert-base-chinese')
        self.model = BERTBiEncoder()
        self._load(f'ckpt/{args["dataset"]}/dual-bert/best.pt')
        if torch.cuda.is_available():
            self.model.cuda()
        self.topk, self.max_len, self.pad = topk, max_len, self.vocab.pad_token_id
        
        self.searcher = Searcher(dataset, vector=True, binary=False, dimension=768, gpu=gpu)
        self.searcher.load(path1, path2)
        
    def _load(self, path):
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict)
        print(f'[!] load model from {path}')
        
    def _length_limit(self, ids):
        if len(ids) > self.max_len:
            ids = [ids[0]] + ids[-(self.max_len-1):]
        return ids
    
    def generate_mask(self, ids):
        attn_mask_index = ids.nonzero().tolist()
        attn_mask_index_x, attn_mask_index_y = [i[0] for i in attn_mask_index], [i[1] for i in attn_mask_index]
        attn_mask = torch.zeros_like(ids)
        attn_mask[attn_mask_index_x, attn_mask_index_y] = 1
        return attn_mask
        
    def _tokenize(self, msgs):
        cids = [torch.LongTensor(self._length_limit(i)) for i in self.vocab.batch_encode_plus(msgs)['input_ids']]
        cids = pad_sequence(cids, batch_first=True, padding_value=self.pad)
        mask = self.generate_mask(cids)
        if torch.cuda.is_available():
            cids, mask = cids.cuda(), mask.cuda()
        return cids, mask
    
    @torch.no_grad()
    def _encode(self, msgs):
        self.model.eval()
        cids, mask = self._tokenize(msgs)
        queries = self.model.get_q(cids, mask).cpu().numpy()
        return queries
            
    def search(self, msgs):
        '''a batch of queries'''
        queries = self._encode(msgs)
        rest, t = self.searcher.search(queries, topk=self.topk)
        return rest, t
        
class ESCoarseRetrieval:
    
    def __init__(self, dataset, topk=20):
        self.topk = topk
        self.searcher = Searcher(dataset)
        
    def search(self, msgs):
        rest, time = self.searcher.search(msgs, topk=self.topk)
        return rest, time
    
class Ranker:
    
    '''cross-bert ranker model'''
    
    def __init__(self, max_len=256):
        self.vocab = BertTokenizer.from_pretrained('bert-base-chinese')
        self.model = BERTRetrieval()
        self._load(f'ckpt/{args["dataset"]}/cross-bert/best.pt')
        if torch.cuda.is_available():
            self.model.cuda()
        self.max_len, self.pad = max_len, self.vocab.pad_token_id
        
    def _load(self, path):
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict)
        print(f'[!] load model from {path}')
        
    def _length_limit(self, ids):
        if len(ids) > self.max_len:
            ids = [ids[0]] + ids[-(self.max_len-1):]
        return ids
    
    def generate_mask(self, ids):
        attn_mask_index = ids.nonzero().tolist()
        attn_mask_index_x, attn_mask_index_y = [i[0] for i in attn_mask_index], [i[1] for i in attn_mask_index]
        attn_mask = torch.zeros_like(ids)
        attn_mask[attn_mask_index_x, attn_mask_index_y] = 1
        return attn_mask
        
    def _tokenize(self, ctxs, reses):
        msgs = [(i, j) for i, j in zip(ctxs, reses)]
        item = self.vocab.batch_encode_plus(msgs)
        ids = [torch.LongTensor(self._length_limit(i)) for i in item['input_ids']]
        tids = [torch.LongTensor(self._length_limit(i)) for i in item['token_type_ids']]
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        tids = pad_sequence(tids, batch_first=True, padding_value=self.pad)
        mask = self.generate_mask(ids)
        if torch.cuda.is_available():
            ids, tids, mask = ids.cuda(), tids.cuda(), mask.cuda()
        return ids, tids, mask
    
    @torch.no_grad()
    def _encode(self, ctxs, reses):
        self.model.eval()
        ids, tids, mask = self._tokenize(ctxs, reses)
        scores = self.model.rank(ids, tids, mask).cpu()
        return scores
            
    def rank(self, ctxs, reses):
        '''a batch of queries and responses; return the order and the average coherence scores'''
        scores = self._encode(ctxs, reses)[:, -1]    # [B]
        return scores

class Agent:
    
    '''chatbot agent:
    * coarse: es/dense/hash'''
    
    def __init__(self, dataset, coarse='es', topk=200, gpu=-1, max_len=256, dimension=768):
        self.topk, self.coarse = topk, coarse
        if coarse == 'es':
            self.searcher = ESCoarseRetrieval(dataset, topk=topk)
        elif coarse == 'dense':
            self.searcher = DenseVectorCoarseRetrieval(
                dataset,
                f'data/{dataset}/dual-bert.faiss_ckpt',
                f'data/{dataset}/dual-bert.corpus_ckpt',
                topk=topk,
                gpu=gpu,
                max_len=max_len,
            )
        elif coarse == 'hash':
            self.searcher = HashVectorCoarseRetrieval(
                dataset,
                f'data/{dataset}/hash-bert.faiss_ckpt',
                f'data/{dataset}/hash-bert.corpus_ckpt',
                topk=topk,
                gpu=gpu,
                max_len=max_len,
                hash_code_size=dimension,
            )
        else:
            raise Exception(f'[!] cannot find the coarse retrieval model: {coarse}')
            
        # post rank
        self.ranker = Ranker(max_len=max_len)
    
    @torch.no_grad()
    def test_coarse(self, test_iter, path):
        '''obtain the top-20/top-100 performance:
        Top-20 & Top-100 retrieval accuracy on test sets, measured as the percentage of top 20/100 retrieved utterances that contain the ground-truth.'''
        correct, time_cost, counter, avg_coherence = 0, [], 0, []
        generated_responses = []
        for ctxs, reses in tqdm(test_iter):
            # stage 1: coarse retrieval
            rest, t = self.searcher.search(ctxs)    # [Queries, Topk]; time cost
            time_cost.append(t)
            for res, rest_ in zip(reses, rest):
                if res in rest_:
                    correct += 1
                counter += 1
            # stage 2: post rank
            for c, r_, r in zip(ctxs, reses, rest):
                scores = self.ranker.rank([c] * self.topk, r)
                avg_coherence.append(scores.mean().item())
                order = torch.argsort(scores, descending=True)
                best = r[order[0]]
                generated_responses.append((c, r_, best))
        # write the data in the last
        with open(path, 'w') as f:
            for c, r, g in generated_responses:
                f.write(f'[CTX]: {c}\n[REF]: {r}\n[GEN]: {g}\n\n')
        print(f'[!] write the generated samples into {path}')
        print(f'[!] MODE: {self.coarse}; Top-{self.topk} Accuracy: {round(correct/counter, 4)}; Top-{self.topk} Avg Coherence: {round(np.mean(avg_coherence), 4)}; average time cost: {round(np.mean(time_cost), 4)}s')
    
if __name__ == "__main__":
    args = vars(parser_args())
    
    random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args['seed'])
    
    test_iter = load_utterance_text_dataset(args)
    agent = Agent(
        args['dataset'], 
        coarse=args['coarse'], 
        topk=args['topk'], 
        gpu=args['gpu'], 
        max_len=args['max_len'], 
        dimension=args['dimension'],
    )
    # generate the rest file into generated/<dataset_name>/<es/dense/hash>/rest.txt
    agent.test_coarse(
        test_iter, 
        f'generated/{args["dataset"]}/{args["coarse"]}/rest.txt'
    )
    
    exit()
    
    # use bert-ruber or bert-ruber-ft metric to give the scores, which is a good replacement of the human evaluation
    args['mode'] = 'test'
    ruber_dataset = load_bert_ruber_dataset(args)
    
    bert_ruber_ft = RUBERMetric(args['gpu'], run_mode='test', ft=True)
    bert_ruber_ft.inference(ruber_dataset)
    
    bert_ruber = RUBERMetric(args['gpu'], run_mode='test', ft=False)
    bert_ruber.inference(ruber_dataset)