import torch
from torch.utils.data import Dataset, DataLoader
from dataraw_sampling import (
    decode_key_to_vocab_token, 
    decode_query_to_vocab_token)
import numpy as np

class GameDatasetFromDataPoints(Dataset):
    
    def __init__(self, raw_data, embedding_by_property, debug=False):
        '''
        raw_data: object returned by sample_dataset()
        '''
        super().__init__()
        self.raw_data = raw_data
        self.embedding_by_property = embedding_by_property
        self.debug = debug
        self.num_attributes = self.raw_data['num_attributes']
        self.num_attr_vals = self.raw_data['num_attr_vals']
        self.num_cards_per_query = self.raw_data['num_cards_per_query']
        self.nest_depth_int = self.raw_data['nest_depth_int']
        self.key_support_size = self.raw_data['key_support_size']
        self.query_support_size = self.raw_data['query_support_size']
        self.OpenP = self.raw_data['(']
        self.CloseP = self.raw_data[')']
        self.NULL = self.raw_data['NULL']
        self.SEP = self.raw_data['SEP']
        self.SOS = self.raw_data['SOS']
        self.EOS = self.raw_data['EOS']
        self.PAD = self.raw_data['PAD']
        self.hold_out = self.raw_data['hold_out']

        self.decode_key_to_vocab_token = decode_key_to_vocab_token
        self.decode_query_to_vocab_token = decode_query_to_vocab_token

    def evaluate_query_idx(self, query_tokens):
        return query_tokens

    def make_gt_binary(self, gt_idxs):
        gt_binary = torch.zeros(self.key_support_size)
        gt_binary[gt_idxs] = 1
        return gt_binary


class GameDatasetTrainDataset(GameDatasetFromDataPoints):
    '''Sample from Presampled Datapoints. Better for Sparse distribution matrix.'''
    
    def __init__(self, raw_data, embedding_by_property, debug=False):
        super().__init__(raw_data, embedding_by_property, debug)
        self.split = 'train'
        
    def __len__(self):
        return len(self.raw_data[self.split + '_datapoints'])
            
    def __getitem__(self, idx):
        '''
        idx: int.
        '''
        # query, key
        y_j, x_i = self.raw_data[self.split + '_datapoints'][idx]
        if self.embedding_by_property:
            y_vocab_tokens, x_vocab_tokens = self.raw_data[self.split + '_tokens'][idx]
        
        if self.debug:
            y_vocab_tokens, x_vocab_tokens = self.raw_data[self.split + '_tokens'][idx]
            print('query\n', y_j, "\n", y_vocab_tokens)
            print('key\n', x_i, "\n", x_vocab_tokens)

        y_j_tensor = torch.tensor([y_j]).long()
        x_i_tensor = torch.tensor([x_i]).long()

        if self.embedding_by_property:
            return (
                y_j_tensor, # query
                x_i_tensor, # gt key
                # shape(2 + 2*num attributes,)
                torch.tensor([self.SOS] + y_vocab_tokens).long(), # query
                # shape(1 + num attributes,)
                torch.tensor([self.SOS] + x_vocab_tokens).long(), # gt key
                # shape(key_support_size,)
            )
        else:
            return (
                y_j_tensor, # query
                x_i_tensor, # gt key
                y_j_tensor, # query
                x_i_tensor, # gt key
            )

class GameDatasetValDataset(GameDatasetFromDataPoints):
    '''Sample from Presampled Datapoints. Better for Sparse distribution matrix.'''
    
    def __init__(self, raw_data, embedding_by_property, debug=False):
        super().__init__(raw_data, embedding_by_property, debug)
        if self.hold_out:
            self.split = 'val'
        else:
            self.split = 'train'
        
    def __len__(self):
        return len(self.raw_data[self.split + '_datapoints'])
            
    def __getitem__(self, idx):
        '''
        idx: int.
        '''
        # query, key
        y_j, x_i = self.raw_data[self.split + '_datapoints'][idx]

        y_vocab_tokens, _ = self.raw_data[self.split + '_tokens'][idx]

        gt_key_list = self.evaluate_query_idx(
            y_vocab_tokens)
        gt_idxs = gt_key_list
        gt_binary = self.make_gt_binary(gt_idxs)

        # if self.embedding_by_property:
        #     y_vocab_tokens, x_vocab_tokens = self.raw_data[self.split + '_tokens'][idx]
        
        if self.debug:
            y_vocab_tokens, x_vocab_tokens = self.raw_data[self.split + '_tokens'][idx]
            print('query\n', y_j, "\n", y_vocab_tokens)
            print('key\n', x_i, "\n", x_vocab_tokens)

        y_j_tensor = torch.tensor([y_j]).long()
        x_i_tensor = torch.tensor([x_i]).long()
        gt_binary_tensor = torch.tensor(gt_binary).long()

        if self.embedding_by_property:
            return (
                y_j_tensor, # query idx
                x_i_tensor, # key idx
                torch.tensor([self.SOS] + y_vocab_tokens).long(), # X query
                torch.tensor([self.SOS] + x_vocab_tokens).long(), # X key
                gt_binary_tensor, # all gt key ids
            )
        else:
            return (
                y_j_tensor, # query idx
                x_i_tensor, # key idx
                y_j_tensor, # X query
                x_i_tensor, # X key
                gt_binary_tensor, # all gt key ids
            )


class GameTestFullDataset(GameDatasetFromDataPoints):
    
    def __init__(self, raw_data, embedding_by_property, debug=False):
        super().__init__(raw_data, embedding_by_property, debug)
        
    def __len__(self):
        return self.raw_data['query_support_size']
    
    def __getitem__(self, idx):
        '''
        key_idx: int. 0 to query_support_size-1
        '''
        y_j = idx

        y_vocab_tokens = self.decode_query_to_vocab_token(
                self.num_attributes, self.num_attr_vals, self.num_cards_per_query, y_j, self.nest_depth_int)

        gt_key_list = self.evaluate_query_idx(
            y_vocab_tokens)
        gt_idxs = gt_key_list
        gt_binary = self.make_gt_binary(gt_idxs)

        # if self.embedding_by_property:
        #     y_vocab_tokens = self.decode_query_to_vocab_token(
        #         self.num_attributes, self.num_attr_vals, self.num_cards_per_query, y_j, self.nest_depth_int)

        if self.debug:
            y_vocab_tokens = self.decode_query_to_vocab_token(
                self.num_attributes, self.num_attr_vals, self.num_cards_per_query, y_j, self.nest_depth_int)
            gt_vocab_tokens = [[self.decode_key_to_vocab_token(
                self.num_attributes, self.num_attr_vals, g_i)] for g_i in gt_idxs]
            print('query\n', y_j, "\n", y_vocab_tokens)
            print('key\n', gt_idxs, "\n", gt_vocab_tokens)

        y_j_tensor = torch.tensor([y_j]).long()
        gt_binary_tensor = torch.tensor(gt_binary).long()

        if self.embedding_by_property:
            return (
                y_j_tensor, # query idx
                torch.tensor([self.SOS] + y_vocab_tokens).long(), # query
                gt_binary_tensor, # all gt key ids
            )  
        else:
            return (
                y_j_tensor, # query idx
                y_j_tensor, # query
                gt_binary_tensor, # all gt key ids
            )
