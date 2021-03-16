import torch
from torch.utils.data import Dataset, DataLoader
from dataraw_sampling import decode_key_to_vocab_token
import numpy as np


class GameDatasetFromDataPoints(Dataset):
    
    def __init__(self, raw_data, model_typ, debug=False):
        '''
        raw_data: object returned by sample_dataset()
        '''
        super().__init__()
        self.raw_data = raw_data
        self.model_typ = model_typ
        self.debug = debug
        self.num_attributes = self.raw_data['num_attributes']
        self.num_attr_vals = self.raw_data['num_attr_vals']
        self.num_cards_per_query = self.raw_data['num_cards_per_query']
        self.nest_depth_int = self.raw_data['nest_depth_int']
        self.key_support_size = self.raw_data['key_support_size']
        self.NULL = self.raw_data['NULL']
        self.SEP = self.raw_data['SEP']
        self.SOS = self.raw_data['SOS']
        self.EOS = self.raw_data['EOS']
        self.PAD = self.raw_data['PAD']
        self.PLH = self.raw_data['PLH']

        self.decode_key_to_vocab_token = decode_key_to_vocab_token

    def make_gt_binary(self, gt_idxs):
        gt_binary = torch.zeros(self.key_support_size)
        gt_binary[gt_idxs] = 1
        return gt_binary


class GameDatasetTrainDataset(GameDatasetFromDataPoints):
    
    def __init__(self, raw_data, model_typ, debug=False):
        super().__init__(raw_data, model_typ, debug)
        self.split = 'train'
        
    def __len__(self):
        return len(self.raw_data[self.split + '_key_datapoints'])
            
    def __getitem__(self, idx):

        x_i = self.raw_data[self.split + '_key_datapoints'][idx]
        y_vocab_tokens, x_vocab_tokens = self.raw_data[self.split + '_tokens'][idx]
        
        if self.debug:
            print('query\n', y_vocab_tokens)
            print('key\n', x_i, "\n", x_vocab_tokens)

        x_i_tensor = torch.tensor([x_i]).long()

        if self.model_typ == 'generative':
            return (
                x_i_tensor, # key idx
                torch.tensor([self.SOS] + y_vocab_tokens + [self.SEP] + x_vocab_tokens + [self.EOS]).long(), # X querykey
            )
        else:
            return (
                x_i_tensor, # key idx
                torch.tensor([self.SOS] + y_vocab_tokens + [self.EOS]).long(), # X query
                torch.tensor([self.SOS] + x_vocab_tokens + [self.EOS]).long(), # X key
            )


class GameDatasetValDataset(GameDatasetFromDataPoints):
    
    def __init__(self, raw_data, model_typ, debug=False):
        super().__init__(raw_data, model_typ, debug)
        self.split = 'val'
        
    def __len__(self):
        return len(self.raw_data[self.split + '_key_datapoints'])
            
    def __getitem__(self, idx):

        x_i = self.raw_data[self.split + '_key_datapoints'][idx]
        y_vocab_tokens, x_vocab_tokens = self.raw_data[self.split + '_tokens'][idx]
        # list of integers
        gt_idxs = self.raw_data[self.split + '_gt_idxs'][idx]
        gt_binary = self.make_gt_binary(gt_idxs)

        if self.debug:
            print('query\n', y_vocab_tokens)
            print('key\n', x_i, "\n", x_vocab_tokens)

        x_i_tensor = torch.tensor([x_i]).long()
        gt_binary_tensor = torch.tensor(gt_binary).long()

        if self.model_typ == 'generative':
            return (
                x_i_tensor, # key idx
                torch.tensor([self.SOS] + y_vocab_tokens + [self.SEP] + x_vocab_tokens + [self.EOS]).long(), # X querykey
                gt_binary_tensor, # all gt key ids
            )
        else:
            return (
                x_i_tensor, # key idx
                torch.tensor([self.SOS] + y_vocab_tokens + [self.EOS]).long(), # X query
                torch.tensor([self.SOS] + x_vocab_tokens + [self.EOS]).long(), # X key
                gt_binary_tensor, # all gt key ids
            )


class GameTestFullDataset(GameDatasetFromDataPoints):
    
    def __init__(self, raw_data, model_typ, debug=False):
        super().__init__(raw_data, model_typ, debug)
        self.split = 'test'
        self.dummy_x_vocab_tokens = [self.PLH] * len(raw_data['train_tokens'][0][1])
        
    def __len__(self):
        return len(self.raw_data[self.split + '_key_datapoints'])
    
    def __getitem__(self, idx):

        y_vocab_tokens, _ = self.raw_data[self.split + '_tokens'][idx]
        # list of integers
        gt_idxs = self.raw_data[self.split + '_gt_idxs'][idx]
        gt_binary = self.make_gt_binary(gt_idxs)

        if self.debug:
            print('query\n', y_vocab_tokens)
            gt_vocab_tokens = [self.decode_key_to_vocab_token(
                self.num_attributes, self.num_attr_vals, g_i) for g_i in gt_idxs]
            print('key\n', gt_idxs, "\n", gt_vocab_tokens)

        gt_binary_tensor = torch.tensor(gt_binary).long()

        if self.model_typ == 'generative':
            return (
                torch.tensor([self.SOS] + y_vocab_tokens + [self.SEP] + self.dummy_x_vocab_tokens + [self.EOS]).long(), # X_query_only
                gt_binary_tensor, # all gt key ids
            )
        else:
            return (
                torch.tensor([self.SOS] + y_vocab_tokens + [self.EOS]).long(), # query
                gt_binary_tensor, # all gt key ids
            )  


class RepresentationStudyDataset(GameDatasetFromDataPoints):
    '''For pulling representation.'''

    def __init__(self, raw_data, model_typ, debug=False):
        super().__init__(raw_data, model_typ, debug)
        self.split = 'repr_study'
        
    def __len__(self):
        return len(self.raw_data[self.split + '_tokens'])

    def __getitem__(self, idx):
        y_vocab_tokens, x_vocab_tokens = self.raw_data[self.split + '_tokens'][idx]

        if self.debug:
            print('query\n', y_vocab_tokens)
            print('key\n', x_vocab_tokens)

        if self.model_typ == 'generative':
            return (
                torch.tensor([self.SOS] + y_vocab_tokens + [self.SEP] + x_vocab_tokens + [self.EOS]).long(), # X querykey
            )
        else:
            return (
                torch.tensor([self.SOS] + y_vocab_tokens + [self.EOS]).long(), # query
                torch.tensor([self.SOS] + x_vocab_tokens + [self.EOS]).long(), # X key
            )   


# TODO : test train data on both contrastive and generative to debug