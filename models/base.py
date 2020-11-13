from .header import *

class RetrievalBaseAgent:

    def __init__(self):
        pass

    def save_model(self, path):
        state_dict = self.model.module.state_dict()
        torch.save(state_dict, path)
        print(f'[!] save model into {path}')
    
    def load_model(self, path):
        state_dict = torch.load(path)
        try:
            self.model.module.load_state_dict(state_dict)
        except:
            self.model.load_state_dict(state_dict)
        print(f'[!] load model from {path}')

    def train_model(self, train_iter, mode='train'):
        raise NotImplementedError

    def test_model(self, test_iter, path):
        raise NotImplementedError
        
    def process_utterances(self, topic, msgs, max_len=0):
        def _length_limit(ids):
            if len(ids) > max_len:
                ids = [ids[0]] + ids[-(max_len-1):]
            return ids
        utterances = self.searcher.search(msgs, samples=self.args['talk_samples'], topic=topic)
        utterances = [i['utterance'] for i in utterances]
        utterances = list(set(utterances) - set(self.history))
        inpt_ids = self.vocab.batch_encode_plus([msgs] + utterances)['input_ids']
        context_inpt_ids, response_inpt_ids = inpt_ids[0], inpt_ids[1:]
        context_inpt_ids = torch.LongTensor(_length_limit(context_inpt_ids))
        response_inpt_ids = [torch.LongTensor(_length_limit(i)) for i in response_inpt_ids]
        response_inpt_ids = pad_sequence(response_inpt_ids, batch_first=True, padding_value=self.args['pad'])
        attn_mask_index = response_inpt_ids.nonzero().tolist()
        attn_mask_index_x, attn_mask_index_y = [i[0] for i in attn_mask_index], [i[1] for i in attn_mask_index]
        attn_mask = torch.zeros_like(response_inpt_ids)
        attn_mask[attn_mask_index_x, attn_mask_index_y] = 1
        
        if torch.cuda.is_available():
            context_inpt_ids, response_inpt_ids, attn_mask = context_inpt_ids.cuda(), response_inpt_ids.cuda(), attn_mask.cuda()
        return utterances, context_inpt_ids, response_inpt_ids, attn_mask

    def process_utterances(self, topic, msgs, max_len=0, context=True):
        '''Process the utterances searched by Elasticsearch; input_ids/token_type_ids/attn_mask'''
        if not context:
            msgs = ''
        utterances_ = self.searcher.search(msgs, samples=self.args['talk_samples'], topic=topic)
        utterances_ = [i['utterance'] for i in utterances_]
        # remove the utterances that in the self.history
        utterances_ = list(set(utterances_) - set(self.history))
        
        # construct inpt_ids, token_type_ids, attn_mask
        inpt_ids = self.vocab.batch_encode_plus([msgs] + utterances_)['input_ids']
        context_inpt_ids, responses_inpt_ids = inpt_ids[0], inpt_ids[1:]
        context_token_type_ids = [0] * len(context_inpt_ids)
        responses_token_type_ids = [[1] * len(i) for i in responses_inpt_ids]
        
        # length limitation
        collection = []
        for r1, r2 in zip(responses_inpt_ids, responses_token_type_ids):
            p1, p2 = context_inpt_ids + r1[1:], context_token_type_ids + r2[1:]
            if len(p1) > max_len:
                cut_size = len(p1) - max_len + 1
                p1, p2 = [p1[0]] + p1[cut_size:], [p2[0]] + p2[cut_size:]
            collection.append((p1, p2))
            
        inpt_ids = [torch.LongTensor(i[0]) for i in collection]
        token_type_ids = [torch.LongTensor(i[1]) for i in collection]
        
        inpt_ids = pad_sequence(inpt_ids, batch_first=True, padding_value=self.args['pad'])
        token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=self.args['pad'])
        attn_mask_index = inpt_ids.nonzero().tolist()
        attn_mask_index_x, attn_mask_index_y = [i[0] for i in attn_mask_index], [i[1] for i in attn_mask_index]
        attn_mask = torch.zeros_like(inpt_ids)
        attn_mask[attn_mask_index_x, attn_mask_index_y] = 1
        
        if torch.cuda.is_available():
            inpt_ids, token_type_ids, attn_mask = inpt_ids.cuda(), token_type_ids.cuda(), attn_mask.cuda()
        return utterances_, inpt_ids, token_type_ids, attn_mask

    def talk(self, msgs):
        raise NotImplementedError

    def get_res(self, data):
        msgs = ' [SEP] '.join([i['msg'] for i in data['msgs']])
        return self.talk(msgs)
