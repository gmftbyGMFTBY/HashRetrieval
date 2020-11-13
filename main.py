from header import *
from models import load_model
from dataloader import load_dataset

def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--dataset', default='zh50w', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--seed', type=float, default=30)
    parser.add_argument('--max_len', type=int, default=300)
    parser.add_argument('--multi_gpu', type=str, default=None)
    parser.add_argument('--local_rank', type=int)
    return parser.parse_args()

def main(**args):
    if args['mode'] == 'train':
        torch.cuda.set_device(args['local_rank'])
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        
        train_iter, agent = load_dataset(args), load_model(args)
        
        sum_writer = SummaryWriter(log_dir=f'rest/{args["dataset"]}/{args["model"]}')
        for i in tqdm(range(args['epoch'])):
            train_loss = agent.train_model(
                train_iter, 
                mode='train',
                recoder=sum_writer,
                idx_=i,
            )
            if args['local_rank'] == 0:
                agent.save_model(f'ckpt/{args["dataset"]}/{args["model"]}/best.pt')
        sum_writer.close()
    elif args['mode'] == 'test':
        test_iter, agent = load_dataset(args), load_model(args)
        agent.load_model(f'ckpt/{args["dataset"]}/{args["model"]}/best.pt')
        agent.test_model(test_iter)
    elif args['mode'] == 'inference':
        torch.cuda.set_device(args['local_rank'])
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        
        iter_, agent = load_dataset(args), load_model(args)
        
        agent.load_model(f'ckpt/{args["dataset"]}/{args["model"]}/best.pt')
        path = f'rest/{args["dataset"]}/{args["model"]}/rest_{args["local_rank"]}.pt'
        agent.inference(iter_, path)
    else:
        raise Exception(f'[!] cannot find the mode: {args["mode"]}')

if __name__ == "__main__":
    args = parser_args()
    args = vars(args)

    random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args['seed'])

    main(**args)
