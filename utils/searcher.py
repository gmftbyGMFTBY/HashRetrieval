import time, json, ipdb, faiss, argparse, pickle
from tqdm import tqdm
import numpy as np
from elasticsearch import Elasticsearch, helpers
from .load_dataset import *

def parser_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', default='zh50w', type=str)
    parser.add_argument('--mode', default='es', type=str, help='es or faiss')
    parser.add_argument('--model', default='dual-bert', type=str, help='dual-bert or hash-bert')
    parser.add_argument('--dim', default=768, type=int)
    parser.add_argument('--gpu', default=-1, type=int)
    return parser.parse_args()

class Searcher:

    '''Real-vector (FAISS) or Term-Frequency Searcher (Elasticsearch)'''

    def __init__(self, dataset, vector=False, binary=False, dimension=768, build=False, gpu=-1):
        self.dataset, self.dimension, self.mode, self.binary = dataset, dimension, vector, binary
        if vector:
            # faiss
            func = faiss.IndexBinaryFlat if binary else faiss.IndexFlatL2
            self.searcher = func(dimension)
            if gpu >= 0:
                # GpuIndexBinaryFlat: https://github.com/facebookresearch/faiss/blob/master/faiss/gpu/test/test_gpu_index.py#L176
                res = faiss.StandardGpuResources()  # use a single GPU
                if binary:
                    self.searcher = faiss.GpuIndexBinaryFlat(res, dimension)
                else:
                    self.searcher = faiss.GpuIndexFlatL2(res, dimension)
                print(f'[!] gpu(cuda:{gpu}) is used for faiss to speed up')
            self.corpus = []
        else:
            # elasticsearch
            self.searcher = Elasticsearch()
            if build:
                self.searcher.indices.delete(index=dataset)
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
        '''dataset is a list of tuple (vector, utterance)'''
        matrix = np.array([i[0] for i in dataset])    # [N, dimension]
        self.corpus = [i[1] for i in dataset]     # [N]
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
        print(f'[!] build the collections with {num} samples')

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
        rest = self._search_faiss(queries, topk=topk) if self.mode else self._search_es(queries, topk=topk)
        et = time.time()
        return rest, round(et - bt, 4)

    def save(self, path1, path2):
        '''only faiss need this procedure'''
        if self.binary:
            faiss.write_index_binary(self.searcher, path1)
        else:
            faiss.write_index(self.searcher, path1)
        # save the text
        with open(path2, 'wb') as f:
            pickle.dump(self.corpus, f)

    def load(self, path1, path2):
        '''only faiss need this procedure'''
        if self.binary:
            self.searcher = faiss.read_index_binary(path1)
        else:
            self.searcher = faiss.read_index(path1)
        with open(path2, 'rb') as f:
            self.corpus = pickle.load(f)

if __name__ == "__main__":
    args = vars(parser_args())
    dataset_name, mode, model, dim, gpu = args['dataset'], args['mode'], args['model'], args['dim'], args['gpu']
    if mode == 'es':
        dataset = load_dataset_es(f'data/{dataset_name}/train.txt', f'data/{dataset_name}/test.txt')
        searcher = Searcher(dataset_name, build=True)
        searcher.build(dataset)
        print(f'[!] build elasticsearch index {dataset_name} over')
    else:
        dataset = load_dataset_faiss(f'rest/{dataset_name}/{model}/rest.pt')
        binary = False if model == 'dual-bert' else True
        searcher = Searcher(dataset_name, vector=True, binary=binary, dimension=dim, gpu=gpu)
        searcher.build(dataset)
        searcher.save(f'data/{dataset_name}/{model}.faiss_ckpt', f'data/{dataset_name}/{model}.corpus_ckpt')
        print(f'[!] build faiss index {dataset_name} over')