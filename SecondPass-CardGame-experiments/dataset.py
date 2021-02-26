import torch
from torch.utils.data import Dataset, DataLoader
from dataraw_full_matrix import gen_card_data
from dataraw_sampling import (
    check_q1q2k_match, decode_key_idx, decode_query_idx, decode_key_to_vocab_token, 
    decode_key_properties_to_vocab_token, encode_key_idx, encode_query_idx)
import numpy as np

class GameDatasetFromFullMatrix():
    '''
    Directly sample from full distribution matrix.
    Only support 1 embed per query, 1 embed per key
    '''
    
    def __init__(self, raw_data, debug=False):
        '''
        raw_data: object returned by gen_card_data.
        '''
        super().__init__()
        self.raw_data = raw_data
        self.debug = debug
        self.num_attrs = self.raw_data['num_attributes']
        self.num_attr_vals = self.raw_data['num_attr_vals']
        self.query_support_size = self.raw_data['query_support_size'] # y
        self.key_support_size = self.raw_data['key_support_size'] # x
        
    def __len__(self):
        return self.query_support_size * self.key_support_size
    
    def __getitem__(self, idx):
        '''
        key_idx: (xy_i) * (xy.shape[1]=self.query_support_size) + (xy_j)
        '''
        x_i, y_j = idx//self.query_support_size, idx%self.query_support_size
        all_matches = list(self.raw_data['query_to_keys'].get(y_j, {}).keys())
        gt = np.zeros(self.key_support_size)
        gt[all_matches] = 1.0
        
        if self.debug:
            yj1, yj2 = self.raw_data['idx_to_query'][y_j]
            print('query\n', y_j,":", yj1, yj2, self.raw_data['idx_to_key'][yj1], self.raw_data['idx_to_key'][yj2])
            print('key\n', x_i, self.raw_data['idx_to_key'][x_i])
            print('all matches \n', [self.raw_data['idx_to_key'][m] for m in all_matches])
        
        return (
            idx, 
            torch.tensor([y_j]).long(), # query
            torch.tensor([x_i]).long(), # gt key
            torch.tensor(gt).long()     # all gt keys 1s and 0s, for metrics computation
        )

############################################################################

class GameDatasetFromDataPoints(Dataset):
    
    def __init__(self, raw_data, embedding_by_property, debug=False):
        '''
        raw_data: object returned by sample_dataset()
        '''
        super().__init__()
        self.raw_data = raw_data
        self.embedding_by_property = embedding_by_property
        self.debug = debug
        self.num_attrs = self.raw_data['num_attributes']
        self.num_attr_vals = self.raw_data['num_attr_vals']
        self.query_support_size = self.raw_data['query_support_size']
        self.key_support_size = self.raw_data['key_support_size']
    
        self.NULL = self.num_attr_vals * self.num_attrs # idx for <NULL>
        self.SEP = self.num_attr_vals * self.num_attrs + 1 # idx for <SEP>
        self.SOS = self.num_attr_vals * self.num_attrs + 2 # idx for <SOS>
        self.EOS = self.num_attr_vals * self.num_attrs + 3 # idx for <EOS>
        self.PAD = self.num_attr_vals * self.num_attrs + 4 # idx for <PAD>

        self.decode_key_to_vocab_token = decode_key_to_vocab_token
        self.decode_key_idx = decode_key_idx
        self.decode_query_idx = decode_query_idx

    def compute_gt(self, query_idx):
        # TODO more efficiently
        q1_idx, q2_idx = self.decode_query_idx(self.num_attrs, self.num_attr_vals, query_idx)
        query1 = self.decode_key_idx(self.num_attrs, self.num_attr_vals, q1_idx)
        query2 = self.decode_key_idx(self.num_attrs, self.num_attr_vals, q2_idx)
        
        gt = [
            float(check_q1q2k_match(self.num_attrs, self.num_attr_vals, query1, query2, k_idx) > 0) \
            for k_idx in range(self.key_support_size)
        ]
        return np.array(gt)


class GameDatasetTrainDataset(GameDatasetFromDataPoints):
    '''Sample from Presampled Datapoints. Better for Sparse distribution matrix.'''
    
    def __init__(self, raw_data, embedding_by_property, split, debug=False):
        assert split in ('train', 'val')
        super().__init__(raw_data, embedding_by_property, debug)
        self.split = split
        
    def __len__(self):
        return len(self.raw_data[self.split + '_datapoints'])
            
    def __getitem__(self, idx):
        '''
        idx: int.
        '''
        # query, key
        y_j, x_i = self.raw_data[self.split + '_datapoints'][idx]
        gt = self.compute_gt(y_j)
           
        if self.embedding_by_property:
            yj1, yj2 = self.decode_query_idx(self.num_attrs, self.num_attr_vals, y_j)
            yj1_vocab_tokens = self.decode_key_to_vocab_token(self.num_attrs, self.num_attr_vals, yj1, self.NULL)
            yj2_vocab_tokens = self.decode_key_to_vocab_token(self.num_attrs, self.num_attr_vals, yj2, self.NULL)
            x_i_vocab_tokens = self.decode_key_to_vocab_token(self.num_attrs, self.num_attr_vals, x_i, self.NULL)
        
        if self.debug:
            yj1, yj2 = self.decode_query_idx(self.num_attrs, self.num_attr_vals, y_j)
            print(
                'query\n', y_j, "\n", yj1, yj2, "\n",
                self.decode_key_idx(self.num_attrs, self.num_attr_vals, yj1), 
                self.decode_key_idx(self.num_attrs, self.num_attr_vals, yj2), "\n",
                self.decode_key_to_vocab_token(self.num_attrs, self.num_attr_vals, yj1, self.NULL),
                self.decode_key_to_vocab_token(self.num_attrs, self.num_attr_vals, yj2, self.NULL)
            )
            print('key\n', x_i, self.decode_key_idx(self.num_attrs, self.num_attr_vals, x_i), self.decode_key_to_vocab_token(self.num_attrs, self.num_attr_vals, x_i, self.NULL))
            print('all matches \n', [self.decode_key_idx(self.num_attrs, self.num_attr_vals, i) for i,v in enumerate(gt) if v == 1.])

        y_j_tensor = torch.tensor([y_j]).long()
        x_i_tensor = torch.tensor([x_i]).long()

        if self.embedding_by_property:
            return (
                y_j_tensor, # query
                x_i_tensor, # gt key
                # shape(2 + 2*num attributes,)
                torch.tensor(np.concatenate([[self.SOS], yj1_vocab_tokens, [self.SEP], yj2_vocab_tokens])).long(), # query
                # shape(1 + num attributes,)
                torch.tensor(np.concatenate([[self.SOS], x_i_vocab_tokens])).long(), # gt key
                # shape(key_support_size,)
                torch.tensor(gt).long()     # all gt keys 1s and 0s, for metrics computation
            )
        else:
            return (
                y_j_tensor, # query
                x_i_tensor, # gt key
                y_j_tensor, # query
                x_i_tensor, # gt key
                torch.tensor(gt).long()     # all gt keys 1s and 0s, for metrics computation
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
        x_i = torch.empty(1) # just a meaningless value
        gt = self.compute_gt(y_j)

        if self.embedding_by_property:
            yj1, yj2 = self.decode_query_idx(self.num_attrs, self.num_attr_vals, y_j)
            yj1_vocab_tokens = self.decode_key_to_vocab_token(self.num_attrs, self.num_attr_vals, yj1, self.NULL)
            yj2_properties = self.decode_key_to_vocab_token(self.num_attrs, self.num_attr_vals, yj2, self.NULL)

        if self.debug:
            yj1, yj2 = self.decode_query_idx(self.num_attrs, self.num_attr_vals, y_j)
            print(
                'query\n', y_j, "\n", yj1, yj2, "\n",
                self.decode_key_idx(self.num_attrs, self.num_attr_vals, yj1), 
                self.decode_key_idx(self.num_attrs, self.num_attr_vals, yj2), "\n",
                self.decode_key_to_vocab_token(self.num_attrs, self.num_attr_vals, yj1, self.NULL),
                self.decode_key_to_vocab_token(self.num_attrs, self.num_attr_vals, yj2, self.NULL)
            )
            print('all matches \n', [self.decode_key_idx(self.num_attrs, self.num_attr_vals, i) for i,v in enumerate(gt) if v == 1.])

        y_j_tensor = torch.tensor([y_j]).long()
        x_i_tensor = torch.tensor([x_i]).long()

        if self.embedding_by_property:
            return (
                y_j_tensor, # query
                x_i_tensor, # dummy key
                # shape(2 + 2*num attributes,)
                torch.tensor(np.concatenate([[self.SOS], yj1_vocab_tokens, [self.SEP], yj2_properties])).long(), # query
                x_i_tensor, # dummy key
                # shape(key_support_size,)
                torch.tensor(gt).long()     # all gt keys 1s and 0s, for metrics computation
            )  
        else:
            return (
                y_j_tensor, # query
                x_i_tensor, # dummy key
                y_j_tensor, # query
                x_i_tensor, # dummy key
                torch.tensor(gt).long()     # all gt keys 1s and 0s, for metrics computation
            )


class PropertyBatchFetcher(GameDatasetFromDataPoints):
    
    def __init__(self, raw_data, embedding_by_property, debug=False):
        '''
        raw_data: object returned by sample_dataset()
        '''
        super().__init__(raw_data, embedding_by_property, debug)
        self.decode_key_properties_to_vocab_token = decode_key_properties_to_vocab_token
        self.encode_key_idx = encode_key_idx
        self.encode_query_idx = encode_query_idx

    def make_batch_from_propertylist(self, queries, keys):
        '''
        queries: property list, of len b, each a list of 2 lists
        keys: property list, of len b, each a list
        '''
        assert queries or keys
        if queries:
            b_q = len(queries)
        if keys:
            b_k = len(keys)

        if queries:
            SOSs = np.array([self.SOS] * b_q).reshape(-1, 1)
            SEPs = np.array([self.SEP] * b_q).reshape(-1, 1)

            queries = np.array(queries)
            # len b
            y_j1s = [self.encode_key_idx(self.num_attrs, self.num_attr_vals, q[0]) for q in queries]
            y_j2s = [self.encode_key_idx(self.num_attrs, self.num_attr_vals, q[1]) for q in queries]
            y_js = [self.encode_query_idx(self.num_attrs, self.num_attr_vals, j1, j2) for j1, j2 in zip(y_j1s, y_j2s)]
            # shape (b, 1)
            y_j_tensors = torch.tensor(y_js).long().unsqueeze(-1)
            gts = torch.tensor([self.compute_gt(y_j) for y_j in y_js]).long()
            if self.embedding_by_property:
                # shape (b, num attr)
                y_j1_vocab_tokens = np.array([decode_key_properties_to_vocab_token(self.num_attrs, self.num_attr_vals, q[0], self.NULL) for q in queries])
                # shape (b, num attr)
                y_j2_vocab_tokens = np.array([decode_key_properties_to_vocab_token(self.num_attrs, self.num_attr_vals, q[1], self.NULL) for q in queries])
                # shape (b, 2*num attr + 2)
                y_jout = torch.tensor(np.concatenate([SOSs, y_j1_vocab_tokens, SEPs, y_j2_vocab_tokens], axis=-1)).long()
            else:
                y_jout = y_j_tensors
        else:
            y_j_tensors = None
            gts = None
            y_jout = None

        if keys:
            SOSs = np.array([self.SOS] * b_k).reshape(-1, 1)
            SEPs = np.array([self.SEP] * b_k).reshape(-1, 1)

            x_is = [self.encode_key_idx(self.num_attrs, self.num_attr_vals, k) for k in keys]
            # shape (b, 1)
            x_i_tensors = torch.tensor(x_is).long().unsqueeze(-1)
            if self.embedding_by_property:
                # shape (b, num attr)
                x_i_vocab_tokens = np.array([decode_key_properties_to_vocab_token(self.num_attrs, self.num_attr_vals, np.array(k), self.NULL) for k in keys])
                # shape (b, num attr + 1)
                x_i_out = torch.tensor(np.concatenate([SOSs, x_i_vocab_tokens], axis=-1)).long()
            else:
                x_i_out = x_i_tensors
        else:
            x_i_tensors = None
            x_i_out = None
            
        return (
            y_j_tensors,
            x_i_tensors,
            y_jout, 
            x_i_out,
            gts     # all gt keys 1s and 0s, for metrics computation
            )