import torch, sys, argparse, ipdb, joblib
import numpy as np
from tqdm import tqdm

def parser_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', default='zh50w', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--num_nodes', type=int)
    return parser.parse_args()

if __name__ == "__main__":
    args = parser_args()
    args = vars(args)
    
    paths = [f'rest/{args["dataset"]}/{args["model"]}/rest_{i}.pt' for i in range(args['num_nodes'])]
    
    dataset_matrix, dataset_text = [], []
    for path in tqdm(paths):
        matrix, text = torch.load(path)
        dataset_matrix.append(matrix)
        dataset_text.extend(text)
    dataset_matrix = np.concatenate(dataset_matrix)
    assert len(dataset_matrix) == len(dataset_text)
    print(f'[!] collect {len(dataset_text)} samples')
    
    with open(f'rest/{args["dataset"]}/{args["model"]}/rest.pt', 'wb') as f:
        joblib.dump((dataset_matrix, dataset_text), f)
    # torch.save((dataset_matrix, dataset_text), f'rest/{args["dataset"]}/{args["model"]}/rest.pt')
    print(f'[!] reconstruct and save the overall embedding into rest/{args["dataset"]}/{args["model"]}/rest.pt')