import torch
from torch.utils.data import Dataset
import numpy as np
import random


from dataraw_sampling import (
    sample_one_training_datapoint, 
    construct_card_idx_lookup,
    draw_N_cardpairs_union_only,
    draw_N_cardpairs_union_symdiff
    )

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
        self.key_support_size = hparams['key_support_size']
        self.vocab_by_property = hparams['vocab_by_property']
        self.N_pairs = hparams['N_pairs']
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
            '|': hparams['|'],
            '!': hparams['!'],
        }

        self.batch_size = hparams['batch_size']
        self.card2idx_lookup, _ = construct_card_idx_lookup(self.num_attributes, self.num_attr_vals)
        self.card2idx_lookup = {**self.card2idx_lookup, **self.symbol_vocab_token_lookup}
        self.sample_one_training_datapoint = sample_one_training_datapoint
        if hparams['union_only']:
            self.draw_N_pairs_fn = draw_N_cardpairs_union_only
        else:
            self.draw_N_pairs_fn = draw_N_cardpairs_union_symdiff


    def __len__(self):
        return self.batch_size * 2
            
    def __getitem__(self, idx):

        # list, list if vocab_by_property else int 
        y_vocab_tokens, x_vocab_tokens, gt_idxs = self.sample_one_training_datapoint(
            num_attributes=self.num_attributes, 
            num_attr_vals=self.num_attr_vals, 
            card2idx_lookup=self.card2idx_lookup, 
            N_pairs=self.N_pairs,
            draw_N_pairs_fn=self.draw_N_pairs_fn
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
        if hparams['mode'] == 'test_marginal':
            self.split = 'test_marginal'
        else:
            self.split = 'test'
        self.raw_data = raw_data

        # PLH
        x_vocab_tokens_len = 1
        self.x_vocab_tokens_place_holder = ([self.PLH] * x_vocab_tokens_len) if self.vocab_by_property else [self.PLH]
        
    def __len__(self):
        return len(self.raw_data[self.split + '_tokens'])
    
    def __getitem__(self, idx):

        # list
        y_vocab_tokens = self.raw_data[self.split + '_tokens'][idx][0]
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
