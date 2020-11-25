import random, argparse, ipdb, pickle
from tqdm import tqdm

def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--dataset', default='zh50w', type=str)
    parser.add_argument('--seed', type=float, default=30)
    parser.add_argument('--samples', type=int, default=200)
    parser.add_argument('--max_length', type=int, default=150)
    return parser.parse_args()

def filter(string, abort=True):
    if len(string) > args['max_length']:
        return False
    if abort:
        return True
    for c in set('abcdefghijklmnopqrstuvwxyz123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
        if c in string:
            return False
    return True

def read_dataset(folder, samples=500):
    '''filter:
    1. ignore the long context session;
    2. ignore the number and the alpha in the context session (number in ecommerce, and alpha beta character)'''
    def _load(path):
        with open(path) as f:
            dataset = []
            for i in tqdm(f.read().split('\n\n')):
                if i.strip():
                    ctx, ref, res = i.split('\n')
                    ctx_ = ctx[7:].replace(' [SEP] ', '')
                    if filter(ctx_, abort=True if 'douban' in folder else False):
                        dataset.append((ctx, ref, res))
        return dataset
    dense = _load(f'{folder}/dense/rest.txt')
    hash = _load(f'{folder}/hash/rest.txt')
    es = _load(f'{folder}/es/rest.txt')
    assert len(dense) == len(hash) and len(hash) == len(es)
    sample_idx = random.sample(range(len(dense)), samples)
    dense = [dense[i] for i in sample_idx]
    hash = [hash[i] for i in sample_idx]
    es = [es[i] for i in sample_idx]
    return dense, hash, es

def write_dataset(dense, hash, es, path_handle):
    def _write(d1, d2, path):
        with open(path, 'w') as f:
            for d1_, d2_ in zip(d1, d2):
                c, r, g1 = d1_
                c, r, g2 = d2_
                f.write(f'{c}\n{r}\n{g1}\n{g2}\n\n')
        print(f'[!] write {len(d1)} samples into {path}')
    _write(dense, es, f'{path_handle}_dense_bm25.txt')
    _write(hash, es, f'{path_handle}_hash_bm25.txt')
    _write(dense, hash, f'{path_handle}_dense_hash.txt')

if __name__ == "__main__":
    args = vars(parser_args())
    samples, dataset = args['samples'], args['dataset']
    random.seed(args['seed'])
    
    dense, hash, es = read_dataset(f'generated/{dataset}', samples=samples)
    write_dataset(dense, hash, es, f'generated/{dataset}/{dataset}_{samples}')