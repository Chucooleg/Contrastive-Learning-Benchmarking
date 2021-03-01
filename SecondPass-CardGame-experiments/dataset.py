import torch
from torch.utils.data import Dataset, DataLoader
from dataraw_sampling import (
    decode_key_idx, decode_query_idx, decode_key_to_vocab_token, 
    decode_key_properties_to_vocab_token, encode_key_idx, encode_query_idx, 
    evaluate_query_idx, decode_query_to_vocab_token)
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
        self.decode_key_idx = decode_key_idx
        self.decode_query_idx = decode_query_idx
        self.evaluate_query_idx = evaluate_query_idx
        self.encode_key_idx = encode_key_idx
        


class GameDatasetTrainDataset(GameDatasetFromDataPoints):
    '''Sample from Presampled Datapoints. Better for Sparse distribution matrix.'''
    
    def __init__(self, raw_data, embedding_by_property, split, debug=False):
        assert split in ('train', 'val')
        super().__init__(raw_data, embedding_by_property, debug)
        if not self.hold_out:
            self.split = 'train'
        else:
            self.split = split
        
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
        gt_key = self.evaluate_query_idx(
            self.num_attributes, self.num_attr_vals, self.num_cards_per_query, y_j, self.nest_depth_int)
        x_i = self.encode_key_idx(self.num_attributes, self.num_attr_vals, gt_key)

        if self.embedding_by_property:
            y_vocab_tokens = self.decode_query_to_vocab_token(
                self.num_attributes, self.num_attr_vals, self.num_cards_per_query, y_j, self.nest_depth_int)
            x_vocab_tokens = self.decode_key_to_vocab_token(
                self.num_attributes, self.num_attr_vals, x_i)

        if self.debug:
            y_vocab_tokens = self.decode_query_to_vocab_token(
                self.num_attributes, self.num_attr_vals, self.num_cards_per_query, y_j, self.nest_depth_int)
            x_vocab_tokens = self.decode_key_to_vocab_token(
                self.num_attributes, self.num_attr_vals, x_i)
            print('query\n', y_j, "\n", y_vocab_tokens)
            print('key\n', x_i, "\n", x_vocab_tokens)

        y_j_tensor = torch.tensor([y_j]).long()
        x_i_tensor = torch.tensor([x_i]).long()

        if self.embedding_by_property:
            return (
                y_j_tensor, # query
                x_i_tensor, # dummy key
                # shape(2 + 2*num attributes,)
                torch.tensor([self.SOS] + y_vocab_tokens).long(), # query
                # shape(1 + num attributes,)
                torch.tensor([self.SOS] + x_vocab_tokens).long(), # gt key
                # shape(key_support_size,)
            )  
        else:
            return (
                y_j_tensor, # query
                x_i_tensor, # dummy key
                y_j_tensor, # query
                x_i_tensor, # dummy key
            )

# TODO need to rewrite
# class PropertyBatchFetcher(GameDatasetFromDataPoints):
    
#     def __init__(self, raw_data, embedding_by_property, debug=False):
#         '''
#         raw_data: object returned by sample_dataset()
#         '''
#         super().__init__(raw_data, embedding_by_property, debug)
#         self.decode_key_properties_to_vocab_token = decode_key_properties_to_vocab_token
#         self.encode_key_idx = encode_key_idx
#         self.encode_query_idx = encode_query_idx

#     def make_batch_from_propertylist(self, queries, keys):
#         '''
#         queries: property list, of len b, each a list of 2 lists
#         keys: property list, of len b, each a list
#         '''
#         assert queries or keys
#         if queries:
#             b_q = len(queries)
#         if keys:
#             b_k = len(keys)

#         if queries:
#             SOSs = np.array([self.SOS] * b_q).reshape(-1, 1)
#             SEPs = np.array([self.SEP] * b_q).reshape(-1, 1)

#             queries = np.array(queries)
#             # len b
#             y_j1s = [self.encode_key_idx(self.num_attributes, self.num_attr_vals, q[0]) for q in queries]
#             y_j2s = [self.encode_key_idx(self.num_attributes, self.num_attr_vals, q[1]) for q in queries]
#             y_js = [self.encode_query_idx(self.num_attributes, self.num_attr_vals, j1, j2) for j1, j2 in zip(y_j1s, y_j2s)]
#             # shape (b, 1)
#             y_j_tensors = torch.tensor(y_js).long().unsqueeze(-1)
#             if self.embedding_by_property:
#                 # shape (b, num attr)
#                 y_j1_vocab_tokens = np.array([self.decode_key_properties_to_vocab_token(self.num_attributes, self.num_attr_vals, q[0], self.NULL) for q in queries])
#                 # shape (b, num attr)
#                 y_j2_vocab_tokens = np.array([self.decode_key_properties_to_vocab_token(self.num_attributes, self.num_attr_vals, q[1], self.NULL) for q in queries])
#                 # shape (b, 2*num attr + 2)
#                 y_jout = torch.tensor(np.concatenate([SOSs, y_j1_vocab_tokens, SEPs, y_j2_vocab_tokens], axis=-1)).long()
#             else:
#                 y_jout = y_j_tensors
#         else:
#             y_j_tensors = None
#             y_jout = None

#         if keys:
#             SOSs = np.array([self.SOS] * b_k).reshape(-1, 1)
#             SEPs = np.array([self.SEP] * b_k).reshape(-1, 1)

#             x_is = [self.encode_key_idx(self.num_attributes, self.num_attr_vals, k) for k in keys]
#             # shape (b, 1)
#             x_i_tensors = torch.tensor(x_is).long().unsqueeze(-1)
#             if self.embedding_by_property:
#                 # shape (b, num attr)
#                 x_i_vocab_tokens = np.array([self.decode_key_properties_to_vocab_token(self.num_attributes, self.num_attr_vals, np.array(k), self.NULL) for k in keys])
#                 # shape (b, num attr + 1)
#                 x_i_out = torch.tensor(np.concatenate([SOSs, x_i_vocab_tokens], axis=-1)).long()
#             else:
#                 x_i_out = x_i_tensors
#         else:
#             x_i_tensors = None
#             x_i_out = None
            
#         return (
#             y_j_tensors,
#             x_i_tensors,
#             y_jout, 
#             x_i_out,
#             )