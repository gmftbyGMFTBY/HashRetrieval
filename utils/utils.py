import time, json, ipdb, faiss
from tqdm import tqdm
import numpy as np
from elasticsearch import Elasticsearch

class Searcher:

    '''Real-vector (FAISS) or Term Searcher (Elasticsearch)'''

    def __init__(self, dataset, vector=False, binary=False, dimension=768, delete=False):
        self.dataset, self.dimension, self.mode, self.binary = dataset, dimension, vector, binary
        if vector:
            # faiss
            func = faiss.IndexBinaryFlat if binary else faiss.IndexFlatL2
            self.searcher = func(dimension)
        else:
            # elasticsearch
            self.searcher = Elasticsearch()
            if delete and self.searcher.indices.exists(index=dataset):
                mapping = {
                    'properties': {
                        'utterance': {
                            'type': 'text',
                            'analyzer': 'ik_max_word',
                            'search_analyzer': 'ik_smart',
                        },
                        'keyword': {
                            'type': 'keyword',
                        }
                    }
                }
                self.searcher.delete(index=dataset)
                self.searcher.indices.create(index=dataset)
                self.searcher.infices.put_mapping(body=mapping, index=dataset)

    def _build_faiss(self, dataset):
        '''dataset is a list of tuple (utterance, vector)'''
        matrix = np.array([i[0] for i in dataset])    # [N, dimension]
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
                'keyword': utterance,
            })
        helpers.bulk(self.searcher, actions)
        return self.searcher.count(index=self.dataset)["count"]

    def build(self, dataset):
        num = self._build_faiss(dataset) if self.mode else self._build_es(dataset)
        print(f'[!] collect {num} samples')

    def _search_faiss(self, vector, topk=20):
        '''batch search; vector: [bsz, dimension]'''
        queries = len(vector)
        D, I = self.searcher.search(vector, topk)
        ipdb.set_trace()
        return I

    def _search_es(self, utterances, topk=20):
        '''utterances: a list of string'''
        search_arr = []
        for query in querys:
            search_arr.append({'index': self.dataset})
            search_arr.append({'query': {'match': {'utterance': query}}, 'size': topk})
        request = ''
        for each in search_arr:
            request += f'{json.dumps(each)} \n'
        rest = self.es.msearch(body=request)
        return rest

    def search(self, queries, topk=20):
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

