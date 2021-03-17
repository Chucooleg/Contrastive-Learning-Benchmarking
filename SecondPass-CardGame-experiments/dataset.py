import torch
from torch.utils.data import Dataset, DataLoader
from dataraw_sampling import (
    sample_one_training_datapoint, sample_one_training_datapoint_simple_shatter, 
    construct_cardpair_answer_lookup, derive_shatter_bucket_probs, sample_shattering_bucket)
import numpy as np


class GameDatasetFromDataPoints(Dataset):
    
    def __init__(self, hparams):
        '''
        raw_data: object returned by sample_dataset()
        '''
        super().__init__()
        self.model_typ = hparams['model']
        self.debug = hparams['debug']
        self.num_attributes = hparams['num_attributes']
        self.num_attr_vals = hparams['num_attr_vals']
        self.nest_depth_int = hparams['nest_depth_int']
        self.key_support_size = hparams['key_support_size']
        self.query_length_multiplier = hparams['query_length_multiplier']
        self.multiple_OR_sets_bool = hparams['multiple_OR_sets_bool']

        self.SEP = hparams['SEP']
        self.SOS = hparams['SOS']
        self.EOS = hparams['EOS']
        self.PLH = hparams['PLH']

    def make_gt_binary(self, gt_idxs):
        gt_binary = torch.zeros(self.key_support_size).long()
        gt_binary[gt_idxs] = 1
        return gt_binary


# class GameDatasetTrainDataset(GameDatasetFromDataPoints):
#     '''Sample training data on the fly'''
    
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

#         self.sampler_weights = derive_shatter_bucket_probs(self.key_support_size)
        
#     def __len__(self):
#         return self.key_support_size
            
#     def __getitem__(self, idx):

#         # sampled in dataloader sampler
#         # bucket_int = idx
#         bucket_int = sample_shattering_bucket(self.num_attributes, self.num_attr_vals, self.sampler_weights)

#         # list, int
#         y_vocab_tokens, x_vocab_tokens = sample_one_training_datapoint_simple_shatter(
#             bucket=bucket_int, 
#             num_attributes=self.num_attributes, 
#             num_attr_vals=self.num_attr_vals, 
#             query_length_multiplier=self.query_length_multiplier, 
#             nest_depth_int=self.nest_depth_int, 
#             multiple_OR_sets_bool=self.multiple_OR_sets_bool,
#             cardpair_answer_lookup=self.cardpair_answer_lookup,
#             symbol_vocab_token_lookup=self.symbol_vocab_token_lookup,
#             )

#         # # list, int
#         # y_vocab_tokens, x_vocab_tokens = self.raw_data[self.split + '_tokens'][idx]
        
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
    
    def __init__(self, hparams, raw_data):
        super().__init__(hparams)
        self.split = 'val'
        self.raw_data = raw_data
        
    def __len__(self):
        return len(self.raw_data[self.split + '_tokens'])
            
    def __getitem__(self, idx):
        # list, int
        y_vocab_tokens, x_vocab_tokens = self.raw_data[self.split + '_tokens'][idx]
        # list of integers
        gt_idxs = self.raw_data[self.split + '_gt_idxs'][idx]

        if self.debug:
            print('query\n', y_vocab_tokens)
            print('key\n', x_vocab_tokens)

        if self.model_typ == 'generative':
            return (
                torch.tensor([self.SOS] + y_vocab_tokens + [self.SEP] + [x_vocab_tokens] + [self.EOS]).long(), # X querykey
            )
        else:
            return (
                torch.tensor([self.SOS] + y_vocab_tokens + [self.EOS]).long(), # X query
                torch.tensor([x_vocab_tokens]).long(), # X key
            )

class GameDatasetValDataset(GameDatasetFromDataPoints):
    
    def __init__(self, hparams, raw_data):
        super().__init__(hparams)
        self.split = 'val'
        self.raw_data = raw_data
        
    def __len__(self):
        return len(self.raw_data[self.split + '_tokens'])
            
    def __getitem__(self, idx):
        # list, int
        y_vocab_tokens, x_vocab_tokens = self.raw_data[self.split + '_tokens'][idx]
        # list of integers
        gt_idxs = self.raw_data[self.split + '_gt_idxs'][idx]
        gt_binary_tensor = self.make_gt_binary(gt_idxs)

        if self.debug:
            print('query\n', y_vocab_tokens)
            print('key\n', x_vocab_tokens)

        if self.model_typ == 'generative':
            return (
                torch.tensor([self.SOS] + y_vocab_tokens + [self.SEP] + [x_vocab_tokens] + [self.EOS]).long(), # X querykey
                gt_binary_tensor, # all gt key ids
            )
        else:
            return (
                torch.tensor([self.SOS] + y_vocab_tokens + [self.EOS]).long(), # X query
                torch.tensor([x_vocab_tokens]).long(), # X key
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


class RepresentationStudyDataset(GameDatasetFromDataPoints):
    '''For pulling representation.'''

    def __init__(self, hparams, raw_data):
        super().__init__(hparams)
        self.split = 'repr_study'
        self.raw_data = raw_data
        
    def __len__(self):
        return len(self.raw_data[self.split + '_tokens'])

    def __getitem__(self, idx):
        # list, int
        y_vocab_tokens, x_vocab_tokens = self.raw_data[self.split + '_tokens'][idx]

        if self.debug:
            print('query\n', y_vocab_tokens)
            print('key\n', x_vocab_tokens)

        if self.model_typ == 'generative':
            return (
                torch.tensor([self.SOS] + y_vocab_tokens + [self.SEP] + [x_vocab_tokens] + [self.EOS]).long(), # X querykey
            )
        else:
            return (
                torch.tensor([self.SOS] + y_vocab_tokens + [self.EOS]).long(), # query
                torch.tensor([x_vocab_tokens]).long(), # X key
            )   
