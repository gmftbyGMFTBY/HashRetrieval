import time, json, ipdb, faiss
from tqdm import tqdm
import numpy as np
from elasticsearch import Elasticsearch, helpers
from .load_dataset import *

class Searcher:

    '''Real-vector (FAISS) or Term Searcher (Elasticsearch)'''

    def __init__(self, dataset, vector=False, binary=False, dimension=768):
        self.dataset, self.dimension, self.mode, self.binary = dataset, dimension, vector, binary
        if vector:
            # faiss
            func = faiss.IndexBinaryFlat if binary else faiss.IndexFlatL2
            self.searcher = func(dimension)
            self.corpus = []
        else:
            # elasticsearch
            self.searcher = Elasticsearch()
            # self.searcher.indices.delete(index=dataset)
            if not self.searcher.indices.exists(index=dataset):
                mapping = {
                    'properties': {
                        'utterance': {
                            'type': 'text',
                            'analyzer': 'ik_max_word',
                            'search_analyzer': 'ik_smart',
                        }
                    }
                }
                self.searcher.indices.create(index=dataset)
                self.searcher.indices.put_mapping(body=mapping, index=dataset)      

    def _build_faiss(self, dataset):
        '''dataset is a list of tuple (utterance, vector)'''
        matrix = np.array([i[1] for i in dataset])    # [N, dimension]
        self.corpus = [i[0] for i in dataset]     # [N]
        assert matrix.shape[1] == self.dimension
        self.searcher.add(matrix)
        return self.searcher.ntotal

    def _build_es(self, dataset):
        '''dataset is a list of string (utterance)'''
        count = self.searcher.count(index=self.dataset)['count']
        actions = []
        for i, utterance in enumerate(tqdm(dataset)):
            actions.append({
                '_index': self.dataset, 
                '_id': i + count,
                'utterance': utterance,
            })
        helpers.bulk(self.searcher, actions)
        return self.searcher.count(index=self.dataset)["count"]

    def build(self, dataset):
        num = self._build_faiss(dataset) if self.mode else self._build_es(dataset)
        print(f'[!] init the collections with {num} samples')

    def _search_faiss(self, vector, topk=20):
        '''batch search; vector: [bsz, dimension]'''
        queries = len(vector)
        _, I = self.searcher.search(vector, topk)    # I: [bsz, topk]
        rest = [[self.corpus[i] for i in neighbors] for neighbors in I]
        return rest

    def _search_es(self, utterances, topk=20):
        '''utterances: a list of string'''
        search_arr = []
        for query in utterances:
            search_arr.append({'index': self.dataset})
            search_arr.append({'query': {'match': {'utterance': query}}, 'size': topk})
        request = ''
        for each in search_arr:
            request += f'{json.dumps(each)} \n'
        collections = self.searcher.msearch(body=request)['responses']
        rest = [[i['_source']['utterance'] for i in r['hits']['hits']] for r in collections]
        return rest

    def search(self, queries, topk=20):
        '''return :rest: [Queries, topk] string as the results; :time: time cost for the given batch retrieval;'''
        bt = time.time()
        rest = self._search_faiss(queries) if self.mode else self._search_es(queries)
        et = time.time()
        return rest, round(et - bt, 4)

    def save(self, path):
        '''only faiss need this procedure'''
        if self.binary:
            faiss.write_index_binary(self.searcher, path)
        else:
            faiss.write_index(self.searcher, path)

    def load(self, path):
        '''only faiss need this procedure'''
        if self.binary:
            self.searcher = faiss.read_index_binary(path)
        else:
            self.searcher = faiss.read_index(path)

if __name__ == "__main__":
    # elasticsearch test
    # dataset = load_dataset_es('data/ecommerce/train.txt', 'data/ecommerce/test.txt')
    searcher = Searcher('ecommerce')
    # searcher.build(dataset)
    rest, time_cost = searcher.search(['今天打算去香山转一圈', '我最喜欢的手机品牌就是华为了，支持国货'])
    print(rest)
    print(f'[!] time cost: {time_cost}')