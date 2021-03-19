import torch
from torch.utils.data import Dataset
import numpy as np
import random

from dataraw_sampling_SETShatter import (
    sample_one_training_datapoint, 
    construct_cardpair_answer_lookup, derive_shatter_bucket_probs, sample_shattering_bucket, sample_keys_column)

# Delete later
import operator as op
from functools import reduce
from tqdm import tqdm
from itertools import chain, combinations

class GameDatasetFromDataPoints(Dataset):
    
    def __init__(self, hparams):
        '''
        raw_data: object returned by sample_dataset()
        '''
        super().__init__()
        self.embedding_by_property = hparams['embedding_by_property']
        self.model_typ = hparams['model']
        self.debug = hparams['debug']
        self.num_attributes = hparams['num_attributes']
        self.num_attr_vals = hparams['num_attr_vals']
        self.nest_depth_int = hparams['nest_depth_int']
        self.key_support_size = hparams['key_support_size']
        self.OpenP = hparams['(']
        self.CloseP = hparams[')']
        self.NULL = hparams['NULL']
        self.SEP = hparams['SEP']
        self.SOS = hparams['SOS']
        self.EOS = hparams['EOS']
        self.PAD = hparams['PAD']
        self.PLH = hparams['PLH']

    def evaluate_query_idx(self, query_tokens):
        return query_tokens

    def make_gt_binary(self, gt_idxs):
        gt_binary = torch.zeros(self.key_support_size)
        gt_binary[gt_idxs] = 1
        return gt_binary


# class GameDatasetTrainDataset(GameDatasetFromDataPoints):
#     '''On the fly SET shatter'''
    
#     def __init__(self, hparams):
#         super().__init__(hparams)
#         self.split = 'train'

#         self.cardpair_answer_lookup = construct_cardpair_answer_lookup(
#             self.num_attributes, self.num_attr_vals)

#         self.symbol_vocab_token_lookup = {
#             '(': hparams['('],
#             ')': hparams[')'],
#             'NULL': hparams['NULL'],
#             'SEP': hparams['SEP'],
#             'SOS': hparams['SOS'],
#             'EOS': hparams['EOS'],
#             'PAD': hparams['PAD'],
#             'PLH': hparams['PLH'],
#             '&': hparams['&'],
#             '|': hparams['|'],
#         }

#         self.query_length_multiplier = hparams['query_length_multiplier']
#         self.multiple_OR_sets_bool = hparams['multiple_OR_sets_bool']
#         self.sampler_weights = derive_shatter_bucket_probs(self.key_support_size)
        
#     def __len__(self):
#         return self.key_support_size
            
#     def __getitem__(self, idx):

#         # sampled in dataloader sampler
#         # bucket_int = idx
#         bucket_int = sample_shattering_bucket(self.num_attributes, self.num_attr_vals, self.sampler_weights)

#         # list, int
#         y_vocab_tokens, x_vocab_tokens = sample_one_training_datapoint(
#             bucket=bucket_int, 
#             num_attributes=self.num_attributes, 
#             num_attr_vals=self.num_attr_vals, 
#             query_length_multiplier=self.query_length_multiplier, 
#             nest_depth_int=self.nest_depth_int, 
#             multiple_OR_sets_bool=self.multiple_OR_sets_bool,
#             cardpair_answer_lookup=self.cardpair_answer_lookup,
#             symbol_vocab_token_lookup=self.symbol_vocab_token_lookup,
#             validate=False,
#             debug=False
#             )
        
#         if self.debug:
#             print('query\n', y_vocab_tokens)
#             print('key\n', x_vocab_tokens)

#         if self.model_typ == 'generative':
#             return (
#                 torch.tensor([self.SOS] + y_vocab_tokens + [self.SEP] + [x_vocab_tokens] + [self.EOS]).long(), # X querykey
#             )
#         else:
#             return (
#                 torch.tensor([self.SOS] + y_vocab_tokens + [self.EOS]).long(), # X query
#                 torch.tensor([x_vocab_tokens]).long(), # X key
#             )

class GameDatasetTrainDataset(GameDatasetFromDataPoints):
    '''On the fly simple shatter. Sample from Bucket, Sample from Column.'''
    
    def __init__(self, hparams):
        super().__init__(hparams)
        self.split = 'train'
        self.setup_buckets()
        self.simple_shatter_query_order = hparams['simple_shatter_query_order']
        assert self.simple_shatter_query_order in ('random', 'ascending', 'descending')

    def setup_buckets(self):
        self.bucket_probs = derive_shatter_bucket_probs(self.key_support_size)
        
    def __len__(self):
        return 10 # dummy
            
    def __getitem__(self, idx):
        '''
        idx: int.
        '''
        y_vocab_tokens, bucket = sample_keys_column(self.num_attributes, self.num_attr_vals, self.bucket_probs)

        if self.simple_shatter_query_order == 'ascending':
            y_vocab_tokens = sorted(y_vocab_tokens)
        elif self.simple_shatter_query_order == 'descending':
            y_vocab_tokens = sorted(y_vocab_tokens, reverse=True)

        # bucket = len(y_vocab_tokens)
        x_vocab_tokens = [int(np.random.choice(y_vocab_tokens))]
        
        if self.debug:
            y_vocab_tokens, x_vocab_tokens = self.raw_data[self.split + '_tokens'][idx]
            print('bucket:', bucket)
            print('query:',  y_vocab_tokens)
            print('key:', x_vocab_tokens)

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

# class GameDatasetTrainDataset(GameDatasetFromDataPoints):
#     '''On the fly simple shatter. Sample from all subsets.'''
    
#     def __init__(self, hparams):
#         super().__init__(hparams)
#         self.split = 'train'
#         self.setup_subsets()

#     def setup_subsets(self):
#         s = [i for i in range(self.key_support_size)]
#         power_set = list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))
#         self.all_subsets = power_set[1:]
#         self.probs = np.array([len(subset) for subset in self.all_subsets]) / np.sum(np.array([len(subset) for subset in self.all_subsets]))
        
#     def __len__(self):
#         return 10 # dummy
            
#     def __getitem__(self, idx):
#         '''
#         idx: int.
#         '''
#         y_vocab_tokens = list(np.random.choice(a=self.all_subsets, p=self.probs))
#         # bucket = len(y_vocab_tokens)
#         x_vocab_tokens = [int(np.random.choice(y_vocab_tokens))]
        
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

# class GameDatasetTrainDataset(GameDatasetFromDataPoints):
#     '''From disk simple shatter.'''
    
#     def __init__(self, hparams):
#         super().__init__(hparams)
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
    
    def __init__(self, hparams, raw_data):
        super().__init__(hparams)
        self.split = 'val'
        self.raw_data = raw_data
        
    def __len__(self):
        return len(self.raw_data[self.split + '_tokens'])
            
    def __getitem__(self, idx):
        '''
        idx: int.
        '''
        y_vocab_tokens, x_vocab_tokens = self.raw_data[self.split + '_tokens'][idx]

        # list of integers
        gt_idxs = self.raw_data[self.split + '_gt_idxs'][idx]
        gt_binary_tensor = self.make_gt_binary(gt_idxs)
        
        if self.debug:
            print('query\n', y_vocab_tokens)
            print('key\n', x_vocab_tokens)


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



class GameTestFullDataset(GameDatasetFromDataPoints):
    
    def __init__(self, hparams, raw_data):
        super().__init__(hparams)
        self.split = 'test'
        self.raw_data = raw_data
        
    def __len__(self):
        return len(self.raw_data[self.split + '_tokens'])
    
    def __getitem__(self, idx):

        y_vocab_tokens, _ = self.raw_data[self.split + '_tokens'][idx]
        # list of integers
        gt_idxs = self.raw_data[self.split + '_gt_idxs'][idx]
        gt_binary_tensor = self.make_gt_binary(gt_idxs)

        if self.debug:
            print('query\n', y_vocab_tokens)
            print('key\n', gt_idxs)

        if self.model_typ == 'generative':
            return (
                torch.tensor([self.SOS] + y_vocab_tokens + [self.SEP] + [self.PLH] + [self.EOS]).long(), # X_query_only
                gt_binary_tensor, # all gt key ids
            )
        else:
            return (
                torch.tensor([self.SOS] + y_vocab_tokens + [self.EOS]).long(), # query
                gt_binary_tensor, # all gt key ids
            )  