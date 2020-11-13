from .dual_bert import *
from .cross_bert import *

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
    else:
        pass
    return agent