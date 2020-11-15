from tqdm import tqdm
import ipdb, json, random, csv, torch

def read_file(path, mode='train'):
    with open(path) as f:
        data = json.load(f)
        if mode == 'train':
            data = data[mode]
        dialogs = []
        for i in data:
            i = [''.join(j.split()) for j in i]
            dialogs.append(i)
    return dialogs

def write_file(dataset, path):
    with open(path, 'w') as f:
        for data in tqdm(dataset):
            data = '\t'.join(data)
            f.write(f'1\t{data}\n')

if __name__ == '__main__':
    seed = 50
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    train_cut_size = 2000000
    dataset = read_file('LCCC-base.json', mode='train')
    dataset = random.sample(dataset, train_cut_size)
    write_file(dataset, 'train.txt')
    dataset = read_file('LCCC-base_test.json', mode='test')
    write_file(dataset, 'test.txt')
