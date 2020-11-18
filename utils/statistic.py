from tqdm import tqdm
import argparse
import numpy as np
from itertools import chain
import jieba
import ipdb

'''statistic of the corpus'''

def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--dataset', default='ecommerce', type=str)
    return parser.parse_args()

def load_stopwords(path):
    with open(path) as f:
        words = [i.strip() for i in f.readlines()]
    return words

def read_corpus(path):
    with open(path) as f:
        dataset = []
        for line in tqdm(f.readlines()):
            line = line.strip().split('\t')
            label, line = line[0], line[1:]
            line = [''.join(i.split()) for i in line]
            if label == '1':
                dataset.append(line)
    return dataset

def session_size(dataset):
    l = [len(i) for i in dataset]
    return np.max(l), np.min(l),round(np.mean(l), 2)

def utterance_size(dataset):
    dataset = list(chain(*dataset))
    l = [len(utterance) for utterance in tqdm(dataset)]
    return np.max(l), np.min(l), round(np.mean(l), 2)

def coherent_check(dataset):
    counter = 0
    for i in tqdm(dataset):
        res = i[-1]
        ctx = ''.join(i[:-1])
        words = list(jieba.cut(res))
        for word in words:
            if word not in stopwords and word in ctx:
                counter += 1
                break
    return round(counter/len(dataset), 4)

if __name__ == "__main__":
    args = vars(parser_args())
    dataset = args['dataset']
    stopwords = load_stopwords('data/stopwords.txt')
    dataset_ = read_corpus(f'data/{dataset}/train.txt')
    max_l, min_l, avg_l = session_size(dataset_)
    print(f'[!] {dataset} turn size (max|min|avg): {max_l}|{min_l}|{avg_l}')
    max_l, min_l, avg_l = utterance_size(dataset_)
    print(f'[!] {dataset} words in utterance (max|min|avg): {max_l}|{min_l}|{avg_l}')
    coherence_ratio = coherent_check(dataset_)
    print(f'[!] {dataset} coherencey ratio: {coherence_ratio}')