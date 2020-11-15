from tqdm import tqdm
import csv, ipdb, torch, joblib

def load_dataset_es(train_path, test_path):
    '''only return a list of the utterances (train + test dataset); use the set for filtering the duplicated utterances and the elasticsearch will not need the collapse'''
    def _load(path):
        utterances = set()
        with open(path) as f:
            for line in tqdm(f.readlines()):
                line = line.strip().split('\t')
                if line[0] == '1':
                    utterances |= set([''.join(line[-1].split())])
        return utterances
    dataset = set()
    dataset |= _load(train_path)
    dataset |= _load(test_path)
    dataset = list(dataset)
    print(f'[!] collect {len(dataset)} samples from {train_path} and {test_path}')
    return dataset

def load_dataset_faiss(path):
    '''train_data or test_data: [(utterance, vector), ...]'''
    # dataset_matrix, dataset_text = torch.load(path)
    with open(path, 'rb') as f:
        dataset_matrix, dataset_text = joblib.load(f)
    dataset = [(m, t) for m, t in zip(dataset_matrix, dataset_text)]
    print(f'[!] collect {len(dataset)} samples from {path}')
    return dataset