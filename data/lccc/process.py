from tqdm import tqdm
import ipdb, json, random, csv, torch

def read_file(path, mode='train'):
    with open(path) as f:
        data = json.load(f)
        if mode == 'train':
            data = data[mode]
        data = [[''.join(j.split()) for j in i] for i in data]
        responses = [i[-1] for i in data]
        dialogs = []
        for i in tqdm(data):
            i = [''.join(j.split()) for j in i]
            if mode == 'train':
                neg = random.choice(responses)
                dialogs.append((i, i[:-1] + [neg]))
            else:
                neg = [i[:-1] + [j] for j in random.sample(responses, 9)]
                dialogs.append((i, neg))
    return dialogs

def write_file(dataset, path, mode='train'):
    with open(path, 'w') as f:
        for data in tqdm(dataset):
            pos_data, neg_data = data
            pos_data = '\t'.join(pos_data)
            if mode == 'train':
                neg_data = '\t'.join(neg_data)
                f.write(f'1\t{pos_data}\n')
                f.write(f'0\t{neg_data}\n')
            else:
                neg_data = ['\t'.join(i) for i in neg_data]
                f.write(f'1\t{pos_data}\n')
                for i in neg_data:
                    f.write(f'0\t{i}\n')

if __name__ == '__main__':
    seed = 50
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    train_cut_size = 2000000
    dataset = read_file('LCCC-base.json', mode='train')
    dataset = random.sample(dataset, train_cut_size)
    write_file(dataset, 'train.txt', mode='train')
    dataset = read_file('LCCC-base_test.json', mode='test')
    write_file(dataset, 'test.txt', mode='test')
