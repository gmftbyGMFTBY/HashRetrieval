from .dual_bert import *
from .cross_bert import *
from .hash_bert import *
from .bert_ruber import *

def load_model(args):
    model_name = args['model']
    if model_name == 'dual-bert':
        agent = BERTBiEncoderAgent(
            args['multi_gpu'], 
            args['total_steps'], 
            run_mode=args['mode'], 
            local_rank=args['local_rank'],
        )
    elif model_name == 'cross-bert':
        agent = BERTRetrievalAgent(
            args['multi_gpu'],
            args['total_steps'],
            run_mode=args['mode'],
            local_rank=args['local_rank'],
        )
    elif model_name == 'hash-bert':
        agent = HashModelAgent(
            args['multi_gpu'],
            args['total_steps'],
            run_mode=args['mode'],
            local_rank=args['local_rank'],
            path=args['pretrained_path'],
        )
    elif model_name == 'bert-ruber':
        agent = RUBERMetric(
            args['multi_gpu'],
            run_mode=args['mode'],
            local_rank=args['local_rank'],
            ft=False,
        )
    elif model_name == 'bert-ruber-ft':
        agent = RUBERMetric(
            args['multi_gpu'],
            run_mode=args['mode'],
            local_rank=args['local_rank'],
            ft=True,
        )
    else:
        raise Exception(f'[!] cannot find the model {args["model"]}')
    return agent