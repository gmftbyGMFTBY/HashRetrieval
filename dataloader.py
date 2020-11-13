import torch, os, ipdb
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

# ========== File loader ========== #
def dual_bert_read_train(path):
    '''for douban300w; e-commerce dataset'''
    with open(path) as f:
        dataset = []
        for line in f.readlines():
            line = line.strip().split('\t')
            label, ctx, res = int(line[0]), line[1:-1], line[-1]
            ctx = ' [SEP] '.join([''.join(i.split()) for i in ctx])
            res = ''.join(res.split())
            if label == 1:
                dataset.append((ctx, res))
        return dataset
    
def dual_bert_read_test(path, samples=10):
    '''for douban300w; e-commerce dataset'''
    with open(path) as f:
        dataset = []
        lines = f.readlines()
        for idx in range(0, len(lines), samples):
            session, label1, step = [], 0, 0
            for line in lines[idx:idx+samples]:
                line = line.strip().split('\t')
                label, ctx, res = int(line[0]), line[1:-1], line[-1]
                if step == 0:
                    assert label == 1
                label1 += label
                assert label1 <= 1
                ctx = ' [SEP] '.join([''.join(i.split()) for i in ctx])
                res = ''.join(res.split()) 
                session.append((label, res))
                step += 1
            session = (ctx, session)
            dataset.append(session)
        return dataset
    
def bert_read_train(path):
    with open(path) as f:
        dataset = []
        for line in f.readlines():
            line = line.strip().split('\t')
            label, utterances = int(line[0]), line[1:]
            utterance = [''.join(i.split()) for i in utterances]
            dataset.append((label, (' [SEP] '.join(utterance[:-1]), utterance[-1])))
        return dataset

def bert_read_test(path, samples=10):
    with open(path) as f:
        dataset = []
        lines = f.readlines()
        for idx in range(0, len(lines), samples):
            session, label1 = [], 0
            for line in lines[idx:idx+samples]:
                line = line.strip().split('\t')
                label, utterances = int(line[0]), line[1:]
                label1 += label
                assert label1 <= 1
                utterance = [''.join(i.split()) for i in utterances]
                session.append((label, (' [SEP] '.join(utterance[:-1]), utterance[-1])))
            dataset.append(session)
        return dataset
    
def bert_embd_read(paths):
    '''train and test datasets'''
    def _load(path):
        with open(path) as f:
            utterances = set()
            for line in f.readlines():
                line = line.strip().split('\t')
                label, us = int(line[0]), line[1:]
                if label == 1:
                    us = [''.join(i.split()) for i in us]
                else:
                    us = [''.join(us[-1].split())]
                utterances |= set(us)
        return utterances
    dataset = set()
    for path in paths:
        dataset |= _load(path)
    dataset = list(dataset)
    print(f'[!] collect {len(dataset)} samples from {paths}')
    return dataset

# ========== Dataset ========== #
class UtteranceTextDataset(Dataset):
    
    '''pure text for agent.py to test (only test mode)'''
    
    def __init__(self, path, max_len=300):
        self.max_len = max_len
        data = dual_bert_read_train(path)
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        ctx, res = self.data[i]
        return ctx, res
    
    def collate(self, batch):
        ctxs, reses = [i[0] for i in batch], [i[1] for i in batch]
        return ctxs, reses

class BertEmbeddingDataset(Dataset):
    
    '''use dual-bert model (response encoder) to generate the embedding for utterances (off-line saving)'''
    
    def __init__(self, paths, max_len=300):
        self.max_len = max_len
        self.vocab = BertTokenizer.from_pretrained('bert-base-chinese')
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.pp_path = f'{os.path.splitext(paths[0])[0]}_embd.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        data = bert_embd_read(paths)
        self.data = []
        for utterance in tqdm(data):
            item = self._length_limit(self.vocab.encode(utterance))
            self.data.append({'ids': item, 'text': utterance})
        torch.save(self.data, self.pp_path)
        print(f'[!] save dataset into {self.pp_path}')
            
    def _length_limit(self, ids):
        if len(ids) > self.max_len:
            ids = [ids[0]] + ids[-(self.max_len-1):]
        return ids
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ids = torch.LongTensor(self.data[i]['ids'])
        text = self.data[i]['text']
        return ids, text
        
    def generate_mask(self, ids):
        attn_mask_index = ids.nonzero().tolist()
        attn_mask_index_x, attn_mask_index_y = [i[0] for i in attn_mask_index], [i[1] for i in attn_mask_index]
        attn_mask = torch.zeros_like(ids)
        attn_mask[attn_mask_index_x, attn_mask_index_y] = 1
        return attn_mask
        
    def collate(self, batch):
        ids, texts = [i[0] for i in batch], [i[1] for i in batch]
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        attn_mask = self.generate_mask(ids)
        if torch.cuda.is_available():
            ids, attn_mask = ids.cuda(), attn_mask.cuda()
        return ids, attn_mask, texts

class RetrievalDataset(Dataset):
    
    '''Only for Douban300w and E-Commerce datasets; test batch size must be 1'''
    
    def __init__(self, path, mode='train', max_len=300):
        self.mode, self.max_len = mode, max_len
        self.vocab = BertTokenizer.from_pretrained('bert-base-chinese')
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.pp_path = f'{os.path.splitext(path)[0]}_dual.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        if mode == 'train':
            data = dual_bert_read_train(path)
        else:
            data = dual_bert_read_test(path, samples=10)
        self.data = []
        if mode == 'train':
            contexts = [i[0] for i in data]
            responses = [i[1] for i in data]
            for context, response in tqdm(list(zip(contexts, responses))):
                item = self.vocab.batch_encode_plus([context, response])
                cid, rid = item['input_ids']
                cid, rid = self._length_limit(cid), self._length_limit(rid)
                self.data.append({'cid': cid, 'rid': rid})
        else:
            # the label 1 must in the index 0 position
            for context, session in tqdm(data):
                responses = [i[1] for i in session]
                item = self.vocab.batch_encode_plus([context] + responses)
                cid, rids = item['input_ids'][0], item['input_ids'][1:]
                cid, rids = self._length_limit(cid), [self._length_limit(i) for i in rids]
                self.data.append({'cid': cid, 'rids': rids})
        self.save()
                
    def _length_limit(self, ids):
        if len(ids) > self.max_len:
            ids = [ids[0]] + ids[-(self.max_len-1):]
        return ids
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.mode == 'train':
            cid, rid = torch.LongTensor(bundle['cid']), torch.LongTensor(bundle['rid'])
            return cid, rid
        else:
            cid, rids = torch.LongTensor(bundle['cid']), [torch.LongTensor(i) for i in bundle['rids']]
            return cid, rids
    
    def save(self):
        torch.save(self.data, self.pp_path)
        print(f'[!] save dataset into {self.pp_path}')
        
    def generate_mask(self, ids):
        attn_mask_index = ids.nonzero().tolist()
        attn_mask_index_x, attn_mask_index_y = [i[0] for i in attn_mask_index], [i[1] for i in attn_mask_index]
        attn_mask = torch.zeros_like(ids)
        attn_mask[attn_mask_index_x, attn_mask_index_y] = 1
        return attn_mask
        
    def collate(self, batch):
        if self.mode == 'train':
            cid, rid = [i[0] for i in batch], [i[1] for i in batch]
            cid = pad_sequence(cid, batch_first=True, padding_value=self.pad)
            rid = pad_sequence(rid, batch_first=True, padding_value=self.pad)
            cid_mask = self.generate_mask(cid)
            rid_mask = self.generate_mask(rid)
            if torch.cuda.is_available():
                cid, rid, cid_mask, rid_mask = cid.cuda(), rid.cuda(), cid_mask.cuda(), rid_mask.cuda()
            return cid, rid, cid_mask, rid_mask
        else:
            assert len(batch) == 1, f'[!] test bacth size must be 1'
            cid, rids = batch[0]
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            rids_mask = self.generate_mask(rids)
            if torch.cuda.is_available():
                cid, rids, rids_mask = cid.cuda(), rids.cuda(), rids_mask.cuda()
            return cid, rids, rids_mask
        
class BERTIRDataset(Dataset):

    def __init__(self, path, mode='train', max_len=300):
        self.mode, self.max_len = mode, max_len 
        if mode == 'train':
            dataset = bert_read_train(path)
        else:
            dataset = bert_read_test(path)
        self.vocab = BertTokenizer.from_pretrained('bert-base-chinese')
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.pp_path = f'{os.path.splitext(path)[0]}_cross.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        self.data = []
        
        if mode == 'train':
            for label, utterance in tqdm(dataset):
                item = self.vocab.batch_encode_plus([utterance])
                self.data.append({
                    'ids': self._length_limit(item['input_ids'][0]),
                    'tids': self._length_limit(item['token_type_ids'][0]),
                    'label': label,
                })
        else:
            for session in tqdm(dataset):
                utterances = [i[1] for i in session]
                item = self.vocab.batch_encode_plus(utterances)
                self.data.append({
                    'ids': self._length_limit(item['input_ids']),
                    'tids': self._length_limit(item['token_type_ids']),
                })
        self.save()
        
    def save(self):
        torch.save(self.data, self.pp_path)
        print(f'[!] save dataset into {self.pp_path}')
        
    def _length_limit(self, ids):
        if len(ids) > self.max_len:
            ids = [ids[0]] + ids[-(self.max_len-1):]
        return ids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.mode == 'train':
            cids = torch.LongTensor(bundle['ids'])
            tids = torch.LongTensor(bundle['tids'])
            return cids, tids, bundle['label']
        else:
            cids, tids = [torch.LongTensor(i) for i in bundle['ids']], [torch.LongTensor(i) for i in bundle['tids']]
            return cids, tid
        
    def generate_mask(self, ids):
        attn_mask_index = ids.nonzero().tolist()
        attn_mask_index_x, attn_mask_index_y = [i[0] for i in attn_mask_index], [i[1] for i in attn_mask_index]
        attn_mask = torch.zeros_like(ids)
        attn_mask[attn_mask_index_x, attn_mask_index_y] = 1
        return attn_mask
    
    def collate(self, batch):
        if self.mode == 'train':
            cids, tids, label = [i[0] for i in batch], [i[1] for i in batch], torch.LongTensor([i[2] for i in batch])
            cids = pad_sequence(cids, batch_first=True, padding_value=self.pad)
            tids = pad_sequence(tids, batch_first=True, padding_value=self.pad)
            attn_mask = self.generate_mask(cids)
            if torch.cuda.is_available():
                cids, tids, attn_mask, label = cids.cuda(), tids.cuda(), attn_mask.cuda(), label.cuda()
            return cids, tids, attn_mask, label
        else:
            # donot shuffle
            cids, tids = [], []
            for i in batch:
                ids.extend(i[0])
                tids.extend(i[1])
            cids = pad_sequence(cids, batch_first=True, padding_value=self.pad)
            tids = pad_sequence(tids, batch_first=True, padding_value=self.pad)
            attn_mask = self.generate_mask(cids)
            if torch.cuda.is_available():
                cids, tids, attn_mask = cids.cuda(), tids.cuda(), attn_mask.cuda()
            return cids, tids, attn_mask
           
# ========== dataset init ========== #
def load_dataset(args):
    if args['mode'] == 'inference':
        return load_bert_embd_dataset(args)
    else:
        if args['model'] == 'dual-bert':
            return load_bert_irbi_dataset(args)
        elif args['model'] == 'cross-bert':
            return load_bert_ir_dataset(args)
        else:
            raise Exception()
            
def load_utterance_text_dataset(args):
    '''called by agent.py'''
    data = UtteranceTextDataset(f'data/{args["dataset"]}/test.txt')
    iter_ = DataLoader(
        data, shuffle=False, batch_size=args['batch_size'], collate_fn=data.collate,
    )
    return iter_

def load_bert_irbi_dataset(args):
    path = f'data/{args["dataset"]}/{args["mode"]}.txt'
    data = RetrievalDataset(path, mode=args['mode'], max_len=args['max_len'])
    if args['mode'] == 'train':
        train_sampler = torch.utils.data.distributed.DistributedSampler(data)
        iter_ = DataLoader(
            data, shuffle=False, batch_size=args['batch_size'], collate_fn=data.collate, 
            sampler=train_sampler,
        )
    else:
        iter_ = DataLoader(
            data, shuffle=False, batch_size=args['batch_size'], collate_fn=data.collate,
        )
    args['total_steps'] = len(data) * args['epoch'] / args['batch_size']
    return iter_

def load_bert_ir_dataset(args):
    path = f'data/{args["dataset"]}/{args["mode"]}.txt'
    if args['mode'] == 'train':
        data = BERTIRDataset(path, mode=args['mode'], max_len=args['max_len'])
        train_sampler = torch.utils.data.distributed.DistributedSampler(data)
        iter_ = DataLoader(
            data, shuffle=False, batch_size=args['batch_size'], 
            collate_fn=data.collate, sampler=train_sampler,
        )
    else:
        data = BERTIRDataset(path, mode=args['mode'], max_len=args['max_len'])
        iter_ = DataLoader(
            data, shuffle=False, batch_size=args['batch_size'], collate_fn=data.collate
        )
    args['total_steps'] = len(data) * args['epoch'] / args['batch_size']
    return iter_

def load_bert_embd_dataset(args):
    '''not for training, just for inferencing'''
    paths = [f'data/{args["dataset"]}/train.txt', f'data/{args["dataset"]}/test.txt']
    data = BertEmbeddingDataset(paths, max_len=args['max_len'])
    train_sampler = torch.utils.data.distributed.DistributedSampler(data)
    iter_ = DataLoader(
        data, shuffle=False, batch_size=args['batch_size'], 
        collate_fn=data.collate, sampler=train_sampler,
    )
    args['total_steps'] = 0
    return iter_

if __name__ == "__main__":
    pass