from .header import *

class RetrievalBaseAgent:

    def __init__(self):
        pass

    def save_model(self, path):
        state_dict = self.model.module.state_dict()
        torch.save(state_dict, path)
        print(f'[!] save model into {path}')
    
    def load_model(self, path):
        state_dict = torch.load(path)
        try:
            self.model.module.load_state_dict(state_dict)
        except:
            self.model.load_state_dict(state_dict)
        print(f'[!] load model from {path}')

    def train_model(self, train_iter, mode='train'):
        raise NotImplementedError

    def test_model(self, test_iter, path):
        raise NotImplementedError