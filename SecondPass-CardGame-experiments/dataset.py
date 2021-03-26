import torch
from torch.utils.data import Dataset
import numpy as np
import random


from dataraw_sampling import (
    sample_one_training_datapoint, 
    construct_card_idx_lookup,
    construct_all_query_and_keys
    )

# from dataraw_sampling_SimpleSET_prop_wildcard import (
#     sample_one_training_datapoint, 
#     construct_cardpair_answer_lookup)

# from dataraw_sampling_SimpleSET_prop import (
#     sample_one_training_datapoint, 
#     construct_cardpair_answer_lookup)

# from dataraw_sampling_SimpleSET_idx import (
#     sample_one_training_datapoint, 
#     construct_cardpair_answer_lookup)


# from dataraw_sampling_SETShatter import (
#     sample_one_training_datapoint, 
#     construct_cardpair_answer_lookup, derive_shatter_bucket_probs, sample_shattering_bucket, sample_keys_column)

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
        self.model_typ = hparams['model']
        self.debug = hparams['debug']
        self.num_attributes = hparams['num_attributes']
        self.num_attr_vals = hparams['num_attr_vals']
        self.nest_depth_int = hparams['nest_depth_int']
        self.key_support_size = hparams['key_support_size']
        self.vocab_by_property = hparams['vocab_by_property']
        self.OpenP = hparams['(']
        self.CloseP = hparams[')']
        self.NULL = hparams['NULL']
        self.SEP = hparams['SEP']
        self.SOS = hparams['SOS']
        self.EOS = hparams['EOS']
        self.PAD = hparams['PAD']
        self.PLH = hparams['PLH']

    def make_gt_binary(self, gt_idxs):
        gt_binary = torch.zeros(self.key_support_size)
        gt_binary[gt_idxs] = 1
        return gt_binary


class GameDatasetTrainDataset(GameDatasetFromDataPoints):
    '''On the fly WildCard SET sample from all'''
    
    def __init__(self, hparams):
        super().__init__(hparams)
        self.split = 'train'

        self.card2idx_lookup, _ = construct_card_idx_lookup(
            self.num_attributes, self.num_attr_vals)

        self.symbol_vocab_token_lookup = {
            '(': hparams['('],
            ')': hparams[')'],
            'NULL': hparams['NULL'],
            'SEP': hparams['SEP'],
            'SOS': hparams['SOS'],
            'EOS': hparams['EOS'],
            'PAD': hparams['PAD'],
            'PLH': hparams['PLH'],
            '&': hparams['&'],
            '|': hparams['|'],
        }

        self.batch_size = hparams['batch_size']
        card2idx_lookup, idx2card_lookup = construct_card_idx_lookup(self.num_attributes, self.num_attr_vals)
        self.all_query_and_keys = construct_all_query_and_keys(self.num_attributes, self.num_attr_vals, card2idx_lookup, idx2card_lookup)

        
    def __len__(self):
        return self.batch_size * 2
            
    def __getitem__(self, idx):

        # list, list if vocab_by_property else int 
        y_vocab_tokens, x_vocab_tokens, gt_idxs = sample_one_training_datapoint(
            self.all_query_and_keys
        )

        # list of integers
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
            if self.vocab_by_property:
                return (
                    torch.tensor([self.SOS] + y_vocab_tokens + [self.EOS]).long(), # X query
                    torch.tensor([self.SOS] + x_vocab_tokens + [self.EOS]).long(), # X key
                    gt_binary_tensor, # all gt key ids
                )    
            else:
                return (
                    torch.tensor([self.SOS] + y_vocab_tokens + [self.EOS]).long(), # X query
                    torch.tensor(x_vocab_tokens).long(), # X key
                    gt_binary_tensor, # all gt key ids
                )


# class GameDatasetTrainDataset(GameDatasetFromDataPoints):
#     '''On the fly WildCard SET rejection sampling'''
    
#     def __init__(self, hparams):
#         super().__init__(hparams)
#         self.split = 'train'

#         self.card2idx_lookup, _ = construct_card_idx_lookup(
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

#         self.batch_size = hparams['batch_size']

        
#     def __len__(self):
#         return self.batch_size * 2
            
#     def __getitem__(self, idx):

#         # list, list if vocab_by_property else int 
#         y_vocab_tokens, x_vocab_tokens, gt_idxs = sample_one_training_datapoint(
#             num_attributes = self.num_attributes, 
#             num_attr_vals = self.num_attr_vals, 
#             card2idx_lookup = self.card2idx_lookup,
#         )

#         # list of integers
#         gt_binary_tensor = self.make_gt_binary(gt_idxs)

#         if self.debug:
#             print('query\n', y_vocab_tokens)
#             print('key\n', x_vocab_tokens)

#         if self.model_typ == 'generative':
#             return (
#                 torch.tensor([self.SOS] + y_vocab_tokens + [self.SEP] + x_vocab_tokens + [self.EOS]).long(), # X querykey
#                 gt_binary_tensor, # all gt key ids
#             )
#         else:
#             if self.vocab_by_property:
#                 return (
#                     torch.tensor([self.SOS] + y_vocab_tokens + [self.EOS]).long(), # X query
#                     torch.tensor([self.SOS] + x_vocab_tokens + [self.EOS]).long(), # X key
#                     gt_binary_tensor, # all gt key ids
#                 )    
#             else:
#                 return (
#                     torch.tensor([self.SOS] + y_vocab_tokens + [self.EOS]).long(), # X query
#                     torch.tensor(x_vocab_tokens).long(), # X key
#                     gt_binary_tensor, # all gt key ids
#                 )


# class GameDatasetTrainDataset(GameDatasetFromDataPoints):
#     '''On the fly Simple SET'''
    
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
        
#     def __len__(self):
#         return self.key_support_size
            
#     def __getitem__(self, idx):

#         # list, list if vocab_by_property else int 
#         y_vocab_tokens, x_vocab_tokens, _ = sample_one_training_datapoint(
#             num_keys = self.key_support_size, 
#             num_attributes = self.num_attributes, 
#             num_attr_vals = self.num_attr_vals, 
#             cardpair_answer_lookup = self.cardpair_answer_lookup, 
#             symbol_vocab_token_lookup = self.symbol_vocab_token_lookup,
#             return_gt=False)
        
#         if self.debug:
#             print('query\n', y_vocab_tokens)
#             print('key\n', x_vocab_tokens)

#         if self.model_typ == 'generative':
#             return (
#                 torch.tensor([self.SOS] + y_vocab_tokens + [self.SEP] + x_vocab_tokens + [self.EOS]).long(), # X querykey
#             )
#         else:
#             if self.vocab_by_property:
#                 return (
#                     torch.tensor([self.SOS] + y_vocab_tokens + [self.EOS]).long(), # X query
#                     torch.tensor([self.SOS] + x_vocab_tokens + [self.EOS]).long(), # X key
#                 )    
#             else:
#                 return (
#                     torch.tensor([self.SOS] + y_vocab_tokens + [self.EOS]).long(), # X query
#                     torch.tensor(x_vocab_tokens).long(), # X key
#                 )


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
#         bucket_int = sample_shattering_bucket(self.num_attributes, self.num_attr_vals, self.sampler_weights)

#         # list, list if vocab_by_property else int 
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
#                 torch.tensor([self.SOS] + y_vocab_tokens + [self.SEP] + x_vocab_tokens + [self.EOS]).long(), # X querykey
#             )
#         else:
#             if self.vocab_by_property:
#                 return (
#                     torch.tensor([self.SOS] + y_vocab_tokens + [self.EOS]).long(), # X query
#                     torch.tensor([self.SOS] + x_vocab_tokens + [self.EOS]).long(), # X key
#                 )    
#             else:
#                 return (
#                     torch.tensor([self.SOS] + y_vocab_tokens + [self.EOS]).long(), # X query
#                     torch.tensor(x_vocab_tokens).long(), # X key
#                 )

# class GameDatasetTrainDataset(GameDatasetFromDataPoints):
#     '''On the fly simple shatter. Sample from Bucket, Sample from Column.'''
    
#     def __init__(self, hparams):
#         super().__init__(hparams)
#         self.split = 'train'
#         self.setup_buckets()

#     def setup_buckets(self):
#         self.bucket_probs = derive_shatter_bucket_probs(self.key_support_size)
        
#     def __len__(self):
#         return 10 # dummy
            
#     def __getitem__(self, idx):
#         '''
#         idx: int.
#         '''
#         y_vocab_tokens, bucket = sample_keys_column(self.num_attributes, self.num_attr_vals, self.bucket_probs)

#         # bucket = len(y_vocab_tokens)
#         x_vocab_tokens = [int(np.random.choice(y_vocab_tokens))]
        
#         if self.debug:
#             y_vocab_tokens, x_vocab_tokens = self.raw_data[self.split + '_tokens'][idx]
#             print('bucket:', bucket)
#             print('query:',  y_vocab_tokens)
#             print('key:', x_vocab_tokens)

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
        # list, list
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
            if self.vocab_by_property:
                return (
                    torch.tensor([self.SOS] + y_vocab_tokens + [self.EOS]).long(), # X query
                    torch.tensor([self.SOS] + x_vocab_tokens + [self.EOS]).long(), # X key
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

        # PLH
        x_vocab_tokens_len = len(self.raw_data[self.split + '_tokens'][0][1])
        self.x_vocab_tokens_place_holder = ([self.PLH] * x_vocab_tokens_len) if self.vocab_by_property else [self.PLH]
        
    def __len__(self):
        return len(self.raw_data[self.split + '_tokens'])
    
    def __getitem__(self, idx):

        # list
        y_vocab_tokens, _ = self.raw_data[self.split + '_tokens'][idx]
        # list of integers
        gt_idxs = self.raw_data[self.split + '_gt_idxs'][idx]
        gt_binary_tensor = self.make_gt_binary(gt_idxs)

        if self.debug:
            print('query\n', y_vocab_tokens)
            print('key\n', gt_idxs)

        if self.model_typ == 'generative':
            return (
                torch.tensor([self.SOS] + y_vocab_tokens + [self.SEP] + self.x_vocab_tokens_place_holder + [self.EOS]).long(), # X_query_only
                gt_binary_tensor, # all gt key ids
            )
        else:
            return (
                torch.tensor([self.SOS] + y_vocab_tokens + [self.EOS]).long(), # query
                gt_binary_tensor, # all gt key ids
            )  