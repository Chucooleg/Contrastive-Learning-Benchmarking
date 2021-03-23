'''
dataraw_sampling_SimpleSET_prop
'''


from collections import defaultdict, Counter
import itertools
import copy
import random
import json
import operator as op
from functools import reduce
from tqdm import tqdm
from itertools import chain, combinations
import numpy as np
import time

####################################################################################

def decode_key_to_vocab_token(num_attributes, num_attr_vals, key_idx):
    '''
    convert key_idx into vocab tokens for training
    '''
    # per attribute e.g. for 4 attr, 16 attr vals (10, 15, 5, 15)
    key_properties = decode_key_idx(num_attributes, num_attr_vals, key_idx)
    return decode_key_properties_to_vocab_token(num_attributes, num_attr_vals, key_properties)

def encode_vocab_token_to_key(num_attributes, num_attr_vals, vocab_tokens):
    '''
    reverse decode_key_to_vocab_token()
    '''
    key_properties = encode_vocab_token_to_key_properties(num_attributes, num_attr_vals, vocab_tokens)
    key_idx = encode_key_idx(num_attributes, num_attr_vals, key_properties)
    return key_idx

def decode_key_properties_to_vocab_token(num_attributes, num_attr_vals, key_properties):
    '''
    key_properties: np array. e.g. for 4 attr, 16 attr vals (10, 15, 5, 15)
    '''
    # (0*16, 1*16, 2*16, 3*16)
    key_range_indices = np.arange(num_attributes) * num_attr_vals
    # according to vocab e.g. for 4 attr, 16 attr vals (0*16+10=10, 1*16+15=31, 2*16+5=37, 3*16+15=63)
    vocab_tokens = key_range_indices + key_properties
    return vocab_tokens.tolist()

def encode_vocab_token_to_key_properties(num_attributes, num_attr_vals, vocab_tokens):
    '''
    reverse of decode_key_properties_to_vocab_token()
    '''
    # vocab_tokens (10, 31, 37, 63)
    if isinstance(vocab_tokens, list):
        vocab_tokens = np.array(vocab_tokens)
    # (0*16, 1*16, 2*16, 3*16)
    key_range_indices = np.arange(num_attributes) * num_attr_vals
    # (10, 15, 5, 15)
    key_properties = vocab_tokens - key_range_indices
    return key_properties

def decode_key_idx(num_attributes, num_attr_vals, key_idx):
    '''
    decode card idx into attr val indices
    '''
    assert key_idx <= num_attr_vals ** num_attributes, (key_idx)
    key_properties = []
    if key_idx == num_attr_vals ** num_attributes:
        return np.array([])
    else:
        card_idx_copy = key_idx
        for i in range(num_attributes):
            digit = card_idx_copy % num_attr_vals
            key_properties = [digit] + key_properties
            card_idx_copy = card_idx_copy // num_attr_vals
        assert len(key_properties) == num_attributes
        return np.array(key_properties)
    
def encode_key_idx(num_attributes, num_attr_vals, key_properties):
    '''
    reverse of decode_key_idx()
    '''

    key_properties_copy = key_properties.tolist()

    if not key_properties_copy:
        return num_attr_vals ** num_attributes
    else:
        key_idx = 0
        digit = 0
        while key_properties_copy:
            attr_val = key_properties_copy.pop()
            key_idx += attr_val * (num_attr_vals**digit)
            digit += 1

    return int(key_idx)

################################################################################################
def SET_derive_property(prop1, prop2):
    '''prop1, prop2 are integers'''
    props = {0,1,2}
    assert prop1 in props and prop2 in props    
    if prop1 == prop2: 
        return prop1
    else:
        return list(props - {prop1, prop2})[0]

def SET_cardpair_resolve_fn(card1, card2):
    assert card1.shape == card2.shape
    return np.array([SET_derive_property(card1[i], card2[i]) for i in range(len(card1))])

def eval_cardpair_by_idx(num_attributes, num_attr_vals, querycard1_idx, querycard2_idx, debug=False):
    qc1_properties = decode_key_idx(num_attributes, num_attr_vals, querycard1_idx)
    qc2_properties = decode_key_idx(num_attributes, num_attr_vals, querycard2_idx)
    if debug: print(qc1_properties, ' | ', qc2_properties)
    key_properties = SET_cardpair_resolve_fn(qc1_properties, qc2_properties)
    key_idx = encode_key_idx(num_attributes, num_attr_vals, key_properties)
    return key_idx, key_properties

def construct_cardpair_answer_lookup(num_attributes, num_attr_vals, debug=False):
    '''Simple SET'''
    num_cards = num_attr_vals**num_attributes
    lookup = {}
    for card1, card2 in itertools.combinations(list(range(num_cards)), 2):
        card1_properties = decode_key_idx(num_attributes, num_attr_vals, card1)
        card2_properties = decode_key_idx(num_attributes, num_attr_vals, card2)
        card3, card3_properties = eval_cardpair_by_idx(num_attributes, num_attr_vals, card1, card2, debug=debug)
        if debug: print (card1_properties, card2_properties, card3_properties, card1, card2, card3)
        lookup[(card1, card2)] = card3
    for card in range(num_cards):
        lookup[(card, card)] = card
    return lookup

####################################################################################

def sample_one_training_datapoint(num_keys, num_attributes, num_attr_vals, cardpair_answer_lookup, symbol_vocab_token_lookup, return_gt=False):
    '''Simple SET'''
    card1_idx = np.random.choice(num_keys)
    card2_idx = np.random.choice(num_keys)

    q_vocab_tokens = decode_key_to_vocab_token(num_attributes, num_attr_vals, card1_idx) + [symbol_vocab_token_lookup['&']] + decode_key_to_vocab_token(num_attributes, num_attr_vals, card2_idx)
    
    # int
    cardk_idx = cardpair_answer_lookup[(min(card1_idx, card2_idx), max(card1_idx, card2_idx))]
    k_vocab_tokens = list(decode_key_to_vocab_token(num_attributes, num_attr_vals, cardk_idx))

    if return_gt:
        gt_ks_idx = [cardk_idx]
    else:
        gt_ks_idx = None

    # list, list
    return q_vocab_tokens, k_vocab_tokens, gt_ks_idx


def sample_queries(num_attributes, num_attr_vals, N_train, N_val, N_test):
    '''Simple SET'''

    num_keys = num_attr_vals**num_attributes
    N = N_train + N_val + N_test
    
    tokens = []
    gt_idxs = []

    cardpair_answer_lookup = construct_cardpair_answer_lookup(num_attributes, num_attr_vals)
    base_vocab_size =  num_attr_vals*num_attributes
    symbol_vocab_token_lookup = {
        '(': base_vocab_size,
        ')': base_vocab_size + 1,
        'NULL': base_vocab_size + 2,
        'SEP': base_vocab_size + 3,
        'SOS': base_vocab_size + 4,
        'EOS': base_vocab_size + 5,
        'PAD': base_vocab_size + 6,
        'PLH': base_vocab_size + 7,
        '&': base_vocab_size + 8,
        '|': base_vocab_size + 9,
    }
    
    max_len_q = 2
    for i in tqdm(range(N)):
        q_vocab_tokens, k_vocab_tokens, gt_ks_idx = sample_one_training_datapoint(
            num_keys, 
            num_attributes, 
            num_attr_vals, 
            cardpair_answer_lookup, 
            symbol_vocab_token_lookup, 
            return_gt=True)

        tokens.append((q_vocab_tokens, k_vocab_tokens))
        gt_idxs.append(gt_ks_idx)

        print((q_vocab_tokens, k_vocab_tokens))
        
        # stats
        max_len_q = max(max_len_q, len(q_vocab_tokens))

    data = {
        'num_attributes':num_attributes,
        'num_attr_vals':num_attr_vals,
        'nest_depth_int': None,
        'key_support_size': num_keys,
        'multiple_OR_sets_bool': None,

        'query_length_multiplier': None,
        'max_len_q': max_len_q,
        'len_k': num_attributes,
        
        #################################        
        'train_gt_idxs': gt_idxs[:N_train],
        'val_gt_idxs': gt_idxs[N_train:N_train+N_val],
        'test_gt_idxs': gt_idxs[N_train+N_val:],
        
        'train_tokens': tokens[:N_train],
        'val_tokens': tokens[N_train:N_train+N_val],
        'test_tokens': tokens[N_train+N_val:],
        
        #################################

        'vocab_size': base_vocab_size + len(symbol_vocab_token_lookup),
        'symbol_vocab_token_lookup': symbol_vocab_token_lookup,
        'vocab_by_property': True,

        #################################
    }

    stats = {}

    return data, stats