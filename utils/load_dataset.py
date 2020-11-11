from tqdm import tqdm
import csv, ipdb, torch

def load_dataset_es(train_path, test_path):
    '''only return a list of the utterances (train + test dataset); use the set for filtering the duplicated utterances and the elasticsearch will not need the collapse'''
    def _load(path):
        utterances = set()
        with open(path) as f:
            csvf = csv.reader(f, delimiter='\t')
            csfv = tqdm(list(csvf))
            for line in csfv:
                if line[0] == '1':
                    us = [''.join(i.split()) for i in line[1:]]
                else:
                    us = [''.join(line[-1].split())]
                utterances |= set(us)
        return utterances
    dataset = set()
    dataset |= _load(train_path)
    dataset |= _load(test_path)
    dataset = list(dataset)
    print(f'[!] collect {len(dataset)} samples from {train_path} and {test_path}')
    return dataset

def load_dataset_faiss(train_path, test_path):
    '''train_data or test_data: [(utterance, vector), ...]'''
    train_data, test_data = torch.load(train_path), torch.load(test_path)
    return train_data + test_data