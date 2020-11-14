from header import *
from models import *
from utils import *
from dataloader import load_utterance_text_dataset

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
    return parser.parse_args()

class HashVectorCoarseRetrieval:
    
    pass

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
    
    def __init__(self, ):
        self.vocab = BertTokenizer.from_pretrained(self.args['vocab_file'])
        self.model = BERTRetrieval(self.args['model'])
        if torch.cuda.is_available():
            self.model.cuda()
        self.model = amp.initialize(self.model, opt_level='O2')
        
    def _tokenize(self):
        pass
        
    @torch.no_grad()
    def rank(self, msgs):
        self.model.eval()
        pass

class Agent:
    
    '''chatbot agent:
    1. coarse: es/dense/hash'''
    
    def __init__(self, dataset, coarse='es', topk=200, gpu=-1):
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
            )
        elif coarse == 'hash':
            self.searcher = HashVectorCoarseRetrieval(
                dataset,
                f'rest/{dataset}/hash-bert.faiss_ckpt',
                topk=topk,
            )
        else:
            raise Exception(f'[!] cannot find the coarse retrieval model: {coarse}')
            
        # post rank
        # self.ranker = Ranker()
    
    @torch.no_grad()
    def test_coarse(self, test_iter):
        '''obtain the top-20/top-100 performance: 
        Top-20 & Top-100 retrieval accuracy on test sets, measured as the percentage of top 20/100 retrieved utterances that contain the ground-truth.'''
        correct, time_cost, counter = 0, [], 0
        for ctxs, reses in tqdm(test_iter):
            rest, t = self.searcher.search(ctxs)    # [Queries, Topk]; time cost
            time_cost.append(t)
            for res, rest_ in zip(reses, rest):
                if res in rest_:
                    correct += 1
                counter += 1
        print(f'[!] MODE: {self.coarse}; Top-{self.topk}: {round(correct/counter, 4)}; average time cost: {round(np.mean(time_cost), 4)}s')
    
    @torch.no_grad()
    def test_overall(self, test_iter):
        pass
    
    @torch.no_grad()
    def talk(self, msgs):
        pass
    
if __name__ == "__main__":
    args = vars(parser_args())
    
    random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args['seed'])
    
    test_iter = load_utterance_text_dataset(args)
    agent = Agent(args['dataset'], coarse=args['coarse'], topk=args['topk'], gpu=args['gpu'])
    if args['test_mode'] == 'coarse':
        agent.test_coarse(test_iter)
    elif args['test_mode'] == 'overall':
        pass
    else:
        raise Exception(f'[!] cannot find the test mode: {args["test_mode"]}')