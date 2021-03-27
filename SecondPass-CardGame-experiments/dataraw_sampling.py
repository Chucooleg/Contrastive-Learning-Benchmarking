'''
dataraw_sampling_SimpleSET_idx_wildcard_expand
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
    raise NotImplmentedError

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
def resolve_prop(prop1, prop2, all_vals_tuple, all_vals_set):
    if prop1 == len(all_vals_tuple) or prop2 == len(all_vals_tuple): # '*' symbol
        return all_vals_tuple
    elif prop1 == prop2:
        return (prop1,)
    else:
        return tuple(all_vals_set - set([prop1, prop2]))
    
def resolve(card1_prop, card2_prop, all_vals_list, all_vals_set):
    '''
    card1_prop/card2_prop: [1, 0 ,*], [2,0,1]
    return answer cards
    '''
    ind_ans = [resolve_prop(prop1, prop2, all_vals_list, all_vals_set) for prop1, prop2 in zip(card1_prop, card2_prop)]
    return list(itertools.product(*ind_ans))

def draw_cardpair_props(num_attributes, num_attr_vals):
    all_vals_tuple = (0,1,2) 
    all_vals_set = set(all_vals_tuple)
    
    card1_prop = np.random.choice(a=num_attr_vals, size=num_attributes, replace=True) # 0, 1, 2
    card2_prop = np.random.choice(a=num_attr_vals+1, size=num_attributes, replace=True) # 0, 1, 2, 3
    keys_prop = resolve(card1_prop, card2_prop, all_vals_tuple, all_vals_set)

    return card1_prop, card2_prop, keys_prop

def draw_N_cardpairs(num_attributes, num_attr_vals, N):
    # union answers from N pairs
    all_key_subsets = []  # list of tuples, each one card
    all_query_pairs = []
    for i in range(N):
        card1_prop, card2_prop, keys_prop = draw_cardpair_props(num_attributes, num_attr_vals)
        all_query_pairs += [card1_prop, card2_prop]
        all_key_subsets += keys_prop
    return all_query_pairs, set(all_key_subsets), len(all_key_subsets)    

def construct_card_idx_lookup(num_attributes, num_attr_vals):
    
    all_card_props = list(itertools.product(*[list(range(4)) for _ in range(num_attributes)]))
    
    card2idx_lookup = {}
    idx2card_lookup = {}
    
    curr_wild_card_idx = num_attr_vals ** num_attributes # first card without *

    for prop in all_card_props:
        if 3 in prop:
            # cards with * start after all key indices
            card2idx_lookup[prop] = curr_wild_card_idx
            idx2card_lookup[curr_wild_card_idx] = prop
            curr_wild_card_idx += 1
        else:
            # key indices are still valid
            card_idx = encode_key_idx(num_attributes, num_attr_vals, np.array(prop))
            card2idx_lookup[prop] = card_idx
            idx2card_lookup[card_idx] = prop
            
    return card2idx_lookup, idx2card_lookup


def construct_all_query_and_keys(num_attributes, num_attr_vals, card2idx_lookup, idx2card_lookup):

    all_query_and_keys = []

    for card1_idx, card2_idx in itertools.product(*[idx2card_lookup.keys(), idx2card_lookup.keys()]):
        
        card1_prop = idx2card_lookup[card1_idx]
        card2_prop = idx2card_lookup[card2_idx]
        keys_prop = resolve(card1_prop, card2_prop, (0,1,2,), set((0,1,2,)))
        gt_ks_idx = [card2idx_lookup[kp] for kp in keys_prop]
        
        for k_id in gt_ks_idx:
            all_query_and_keys.append(([card1_idx, card2_idx], [k_id], gt_ks_idx))
        
    return all_query_and_keys
####################################################################################

def sample_one_testing_query_from_marginal(num_attributes, num_attr_vals, card2idx_lookup):
    '''Wrong sampling props for training.'''

    # card1_prop, card2_prop, keys_prop = draw_cardpair_props(num_attributes, num_attr_vals)
    query_props, key_props, keys_len = draw_N_cardpairs(num_attributes, num_attr_vals, N=3)

    # 6 tokens
    q_vocab_tokens = [card2idx_lookup[tuple(qp)] for qp in query_props]

    gt_ks_idx = [card2idx_lookup[kp] for kp in key_props]

    # list, list, list
    return q_vocab_tokens, None, gt_ks_idx


def sample_one_training_datapoint(num_attributes, num_attr_vals, card2idx_lookup):
    '''Rejection Sampling'''

    success = False 

    while not success:

        query_props, key_props, keys_len = draw_N_cardpairs(num_attributes, num_attr_vals, N=3)
        if random.random() <= (len(ans_subset) *1.0 / (num_attr_vals**num_attributes)):
            success = True

    # 6 tokens
    q_vocab_tokens = [card2idx_lookup[tuple(qp)] for qp in query_props]

    gt_ks_idx = [card2idx_lookup[kp] for kp in key_props]
    k_vocab_tokens = random.choice(gt_ks_idx)

    # list, list, list
    return q_vocab_tokens, [k_vocab_tokens], gt_ks_idx


# def sample_one_training_datapoint(all_query_and_keys):
#     '''from pxy'''

#     idx = np.random.choice(len(all_query_and_keys))
#     q_vocab_tokens = all_query_and_keys[idx][0]
#     k_vocab_tokens = all_query_and_keys[idx][1]
#     gt_ks_idx = all_query_and_keys[idx][2]

#     # list, list, list
#     return q_vocab_tokens, k_vocab_tokens, gt_ks_idx


def sample_queries(num_attributes, num_attr_vals, N_train, N_val, N_test):
    '''Simple SET'''

    num_keys = num_attr_vals**num_attributes
    num_any_keys = (num_attr_vals+1)**num_attributes
    N = N_train + N_val + N_test
    
    tokens = []
    gt_idxs = []

    test_marginal_tokens = []
    test_marginal_gt_idxs = []

    start_time = time.time()
    card2idx_lookup, idx2card_lookup = construct_card_idx_lookup(num_attributes, num_attr_vals)
    print('Time to build cardpair_answer_lookup:', time.time()-start_time, 'seconds')

    start_time = time.time()
    all_query_and_keys = construct_all_query_and_keys(num_attributes, num_attr_vals, card2idx_lookup, idx2card_lookup)
    print('Time to build all_query_and_keys:', time.time()-start_time, 'seconds')

    base_vocab_size =  (num_attr_vals+1)**num_attributes # include * cards
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
            all_query_and_keys
        )

        tokens.append((q_vocab_tokens, k_vocab_tokens))
        gt_idxs.append(gt_ks_idx)
        
        # stats
        max_len_q = max(max_len_q, len(q_vocab_tokens))


    for i in tqdm(range(N_test)):

        q_vocab_tokens, _, gt_ks_idx = sample_one_testing_query_from_marginal(
            num_attributes=num_attributes, 
            num_attr_vals=num_attr_vals, 
            card2idx_lookup=card2idx_lookup, 
        )

        test_marginal_tokens.append((q_vocab_tokens,))
        test_marginal_gt_idxs.append(gt_ks_idx)       


    data = {
        'num_attributes':num_attributes,
        'num_attr_vals':num_attr_vals,
        'nest_depth_int': None,
        'key_support_size': num_keys,
        'multiple_OR_sets_bool': None,

        'query_length_multiplier': None,
        'max_len_q': max_len_q,
        'len_k': 1,
        
        #################################        
        'train_gt_idxs': gt_idxs[:N_train],
        'val_gt_idxs': gt_idxs[N_train:N_train+N_val],
        'test_gt_idxs': gt_idxs[N_train+N_val:],
        
        'train_tokens': tokens[:N_train],
        'val_tokens': tokens[N_train:N_train+N_val],
        'test_tokens': tokens[N_train+N_val:],

        'test_marginal_gt_idxs': test_marginal_gt_idxs,
        'test_marginal_tokens': test_marginal_tokens,

        #################################

        'vocab_size': base_vocab_size + len(symbol_vocab_token_lookup),
        'symbol_vocab_token_lookup': symbol_vocab_token_lookup,
        'vocab_by_property': False,

        #################################
    }

    stats = {}

    return data, stats