import torch
from torch.utils.data import Dataset
import numpy as np
import random

# Delete later
import operator as op
from functools import reduce
from tqdm import tqdm
from itertools import chain, combinations

class GameDatasetFromDataPoints(Dataset):
    
    def __init__(self, raw_data, embedding_by_property, model_typ, debug=False):
        '''
        raw_data: object returned by sample_dataset()
        '''
        super().__init__()
        self.raw_data = raw_data
        self.embedding_by_property = embedding_by_property
        self.model_typ = model_typ
        self.debug = debug
        self.num_attributes = self.raw_data['num_attributes']
        self.num_attr_vals = self.raw_data['num_attr_vals']
        self.nest_depth_int = self.raw_data['nest_depth_int']
        self.key_support_size = self.raw_data['key_support_size']
        self.OpenP = self.raw_data['(']
        self.CloseP = self.raw_data[')']
        self.NULL = self.raw_data['NULL']
        self.SEP = self.raw_data['SEP']
        self.SOS = self.raw_data['SOS']
        self.EOS = self.raw_data['EOS']
        self.PAD = self.raw_data['PAD']
        self.PLH = self.raw_data['PLH']

    def evaluate_query_idx(self, query_tokens):
        return query_tokens

    def make_gt_binary(self, gt_idxs):
        gt_binary = torch.zeros(self.key_support_size)
        gt_binary[gt_idxs] = 1
        return gt_binary


class GameDatasetTrainDataset(GameDatasetFromDataPoints):
    '''On the fly.'''
    
    def __init__(self, raw_data, embedding_by_property, model_typ, debug=False):
        super().__init__(raw_data, embedding_by_property, model_typ, debug)
        self.split = 'train'
        self.setup_subsets()

    def setup_subsets(self):
        s = [i for i in range(self.key_support_size)]
        power_set = list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))
        self.all_subsets = power_set[1:]
        self.probs = np.array([len(subset) for subset in self.all_subsets]) / np.sum(np.array([len(subset) for subset in self.all_subsets]))
        
    def __len__(self):
        return 10 # dummy
            
    def __getitem__(self, idx):
        '''
        idx: int.
        '''
        y_vocab_tokens = list(np.random.choice(a=self.all_subsets, p=self.probs))
        # bucket = len(y_vocab_tokens)
        x_vocab_tokens = [int(np.random.choice(y_vocab_tokens))]

        # # query, key
        # y_vocab_tokens, x_vocab_tokens = self.raw_data[self.split + '_tokens'][idx]
        
        if self.debug:
            y_vocab_tokens, x_vocab_tokens = self.raw_data[self.split + '_tokens'][idx]
            print('query\n',  y_vocab_tokens)
            print('key\n', x_vocab_tokens)

        if self.model_typ == 'generative':
            return (
                # shape(2 + 2*num attributes,)
                torch.tensor([self.SOS] + y_vocab_tokens + [self.SEP] + x_vocab_tokens + [self.EOS]).long(), # X querykey
            )
        else:
            return (
                # shape(2 + 2*num attributes,)
                torch.tensor([self.SOS] + y_vocab_tokens + [self.EOS]).long(), # X query
                # shape(1 + num attributes,)
                torch.tensor(x_vocab_tokens).long(), # X key
            )

# Use this one?
# class GameDatasetTrainDataset(GameDatasetFromDataPoints):
#     '''Sample from Presampled Datapoints. Better for Sparse distribution matrix.'''
    
#     def __init__(self, raw_data, embedding_by_property, model_typ, debug=False):
#         super().__init__(raw_data, embedding_by_property, model_typ, debug)
#         self.split = 'val'
        
#     def __len__(self):
#         return len(self.raw_data[self.split + '_tokens'])
            
#     def __getitem__(self, idx):
#         '''
#         idx: int.
#         '''
#         # query, key
#         y_vocab_tokens, x_vocab_tokens = self.raw_data[self.split + '_tokens'][idx]
        
#         if self.debug:
#             y_vocab_tokens, x_vocab_tokens = self.raw_data[self.split + '_tokens'][idx]
#             print('query\n',  y_vocab_tokens)
#             print('key\n', x_vocab_tokens)

#         if self.model_typ == 'generative':
#             return (
#                 # shape(2 + 2*num attributes,)
#                 torch.tensor([self.SOS] + y_vocab_tokens + [self.SEP] + x_vocab_tokens + [self.EOS]).long(), # X querykey
#             )
#         else:
#             return (
#                 # shape(2 + 2*num attributes,)
#                 torch.tensor([self.SOS] + y_vocab_tokens + [self.EOS]).long(), # X query
#                 # shape(1 + num attributes,)
#                 torch.tensor(x_vocab_tokens).long(), # X key
#             )


class GameDatasetValDataset(GameDatasetFromDataPoints):
    '''Sample from Presampled Datapoints. Better for Sparse distribution matrix.'''
    
    def __init__(self, raw_data, embedding_by_property, model_typ, debug=False):
        super().__init__(raw_data, embedding_by_property, model_typ, debug)
        self.split = 'val'

        
    def __len__(self):
        return len(self.raw_data[self.split + '_tokens'])
            
    def __getitem__(self, idx):
        '''
        idx: int.
        '''
        y_vocab_tokens, x_vocab_tokens = self.raw_data[self.split + '_tokens'][idx]

        gt_key_list = self.evaluate_query_idx(
            y_vocab_tokens)
        gt_idxs = gt_key_list
        gt_binary = self.make_gt_binary(gt_idxs)
        
        if self.debug:
            y_vocab_tokens, x_vocab_tokens = self.raw_data[self.split + '_tokens'][idx]
            print('query\n', y_vocab_tokens)
            print('key\n', x_vocab_tokens)

        gt_binary_tensor = torch.tensor(gt_binary).long()

        if self.model_typ == 'generative':
            return (
                torch.tensor([self.SOS] + y_vocab_tokens + [self.SEP] + x_vocab_tokens + [self.EOS]).long(), # X querykey
                gt_binary_tensor, # all gt key ids
            )
        else:
            return (
                torch.tensor([self.SOS] + y_vocab_tokens + [self.EOS]).long(), # X query
                torch.tensor(x_vocab_tokens).long(), # X key
                gt_binary_tensor, # all gt key ids
            )



# TODO test dataset