import time
import numpy as np
import random
from util_distribution import get_distribution, plot_distribution
from tqdm import tqdm


def decode_key_to_vocab_token(num_attrs, num_attr_vals, key_idx, null_vocab_idx):
    '''
    convert key_idx into vocab tokens for training
    '''
    # per attribute e.g. for 4 attr, 16 attr vals (10, 15, 5, 15)
    key_properties = decode_key_idx(num_attrs, num_attr_vals, key_idx)
    return decode_key_properties_to_vocab_token(num_attrs, num_attr_vals, key_properties, null_vocab_idx)

def encode_vocab_token_to_key(num_attrs, num_attr_vals, vocab_tokens, null_vocab_idx):
    '''
    reverse decode_key_to_vocab_token()
    '''
    key_properties = encode_vocab_token_to_key_properties(num_attrs, num_attr_vals, vocab_tokens, null_vocab_idx)
    key_idx = encode_key_idx(num_attrs, num_attr_vals, key_properties)
    return key_idx

def decode_key_properties_to_vocab_token(num_attrs, num_attr_vals, key_properties, null_vocab_idx):
    '''
    key_properties: np array. e.g. for 4 attr, 16 attr vals (10, 15, 5, 15)
    '''
    # (0*16, 1*16, 2*16, 3*16)
    key_range_indices = np.arange(num_attrs) * num_attr_vals
    if key_properties.size == 0:
        # null card
        vocab_tokens = np.array([null_vocab_idx] * num_attrs)
    else:
        # according to vocab e.g. for 4 attr, 16 attr vals (0*16+10=10, 1*16+15=31, 2*16+5=37, 3*16+15=63)
        vocab_tokens = key_range_indices + key_properties
    return vocab_tokens

def encode_vocab_token_to_key_properties(num_attrs, num_attr_vals, vocab_tokens, null_vocab_idx):
    '''
    reverse of decode_key_properties_to_vocab_token()
    '''
    assert isinstance(vocab_tokens, np.ndarray), 'vocab tokens should be type np.ndarray'
    # (0*16, 1*16, 2*16, 3*16)
    key_range_indices = np.arange(num_attrs) * num_attr_vals
    if set(vocab_tokens) == {null_vocab_idx}:
        key_properties = np.array([])
    else:
        key_properties = vocab_tokens - key_range_indices
    return key_properties

def decode_key_idx(num_attrs, num_attr_vals, key_idx):
    '''
    decode card idx into attr val indices
    '''
    assert key_idx <= num_attr_vals ** num_attrs, (key_idx)
    key_properties = []
    if key_idx == num_attr_vals ** num_attrs:
        return np.array([])
    else:
        card_idx_copy = key_idx
        for i in range(num_attrs):
            digit = card_idx_copy % num_attr_vals
            key_properties = [digit] + key_properties
            card_idx_copy = card_idx_copy // num_attr_vals
        assert len(key_properties) == num_attrs
        return np.array(key_properties)
    
def encode_key_idx(num_attrs, num_attr_vals, key_properties):
    '''
    reverse of decode_key_idx()
    '''
    key_properties = list(key_properties)

    if not key_properties:
        return num_attr_vals ** num_attrs
    else:
        key_idx = 0
        digit = 0
        while key_properties:
            attr_val = key_properties.pop()
            key_idx += attr_val * (num_attr_vals**digit)
            digit += 1
    return key_idx

##############################

def decode_query_idx_to_key_properties(num_attrs, num_attr_vals, query_idx):
    key1_idx, key2_idx = decode_query_idx(num_attrs, num_attr_vals, query_idx)
    return [decode_key_idx(num_attrs, num_attr_vals, key1_idx), decode_key_idx(num_attrs, num_attr_vals, key2_idx)]

def decode_query_idx(num_attrs, num_attr_vals, query_idx):
    '''decode query_idx into pair of card indices'''
    num_cards = num_attr_vals ** num_attrs
    card1_idx = query_idx // num_cards
    card2_idx = query_idx % num_cards
    return card1_idx, card2_idx

def encode_query_idx(num_attrs, num_attr_vals, card1_idx, card2_idx):
    '''reverse decode_query_idx()'''
    num_cards = num_attr_vals ** num_attrs
    query_idx = card1_idx * num_cards + card2_idx
    return query_idx

##################################################################
def check_if_query_key_match(query_part1, query_part2, key, debug=False):
    shared_attr_filter_q1q2 = query_part1 == query_part2
    shared_attr_filter_q1k = query_part1 == key
    matches = (shared_attr_filter_q1q2 & shared_attr_filter_q1k) # F, F , T
    num_matches = np.sum(matches) # 1
    if debug: print(query_part1, query_part2, key, num_matches)
    return num_matches

def queryidx_to_querypair(num_attrs, num_attr_vals, query_idx):  
    q1_idx, q2_idx = decode_query_idx(num_attrs, num_attr_vals, query_idx)
    query_part1 = decode_key_idx(num_attrs, num_attr_vals, q1_idx)
    query_part2 = decode_key_idx(num_attrs, num_attr_vals, q2_idx)
    return query_part1, query_part2
    
def check_if_query_key_match_by_idx(num_attrs, num_attr_vals, query_idx, key_idx):
    '''return an int'''
    query_part1, query_part2 = queryidx_to_querypair(num_attrs, num_attr_vals, query_idx)
    return check_q1q2k_match(num_attrs, num_attr_vals, query_part1, query_part2, key_idx)

def check_q1q2k_match(num_attrs, num_attr_vals, query_part1, query_part2, key_idx):
    if key_idx == num_attr_vals ** num_attrs: # null key card
        if not (query_part1 == query_part2).any():
            return 1
        else:
            return 0
    else: # key card with real values
        key = decode_key_idx(num_attrs, num_attr_vals, key_idx)
        num_matched_attributes = check_if_query_key_match(query_part1, query_part2, key)
        return num_matched_attributes 

def query_has_real_matches(query_part1, query_part2):
    return np.sum(query_part1 == query_part2)

def query_idx_has_real_matches(num_attrs, num_attr_vals, query_idx):
    query_part1, query_part2 = queryidx_to_querypair(num_attrs, num_attr_vals, query_idx)
    return query_has_real_matches(query_part1, query_part2)

##################################################################

def sample_query_key_idx(num_attrs, num_attr_vals):
    num_keys = num_attr_vals ** num_attrs
    num_queries = num_keys * num_keys
    key_idx = random.randrange(num_keys + 1)
    query_idx = random.randrange(num_queries)
    return query_idx, key_idx
    
def sample_valid_query_key_idx(num_attrs, num_attr_vals):
    valid = False
    trials = 0
    while not valid:
        tmp_query_idx, tmp_key_idx = sample_query_key_idx(num_attrs, num_attr_vals)
        valid = check_if_query_key_match_by_idx(num_attrs, num_attr_vals, tmp_query_idx, tmp_key_idx)
        if valid: valids = (tmp_query_idx, tmp_key_idx)
        trials += 1
    return trials, valids

def sample_N_datapoints(num_attrs, num_attr_vals, N):
    all_valids = []
    cts = 0
    for i in tqdm(range(N)):
        trials, valids = sample_valid_query_key_idx(num_attrs, num_attr_vals)
        cts += trials
        all_valids.append(valids)
    sparsity_estimate = 1/(cts / N)
    print('Sparsity Estimate:', sparsity_estimate)
    return all_valids, sparsity_estimate
   
def sample_dataset(num_attrs, num_attr_vals, N_train, N_val):
    start_time = time.time()

    N = N_train + N_val
    datapoints, sparsity_estimate = sample_N_datapoints(num_attrs, num_attr_vals, N)
        
    data = {
        'num_attributes':num_attrs,
        'num_attr_vals':num_attr_vals,
        'key_support_size': num_attr_vals**num_attrs + 1,
        'query_support_size': (num_attr_vals**num_attrs)**2,
        'train_datapoints': datapoints[:N_train],
        'val_datapoints': datapoints[N_train:N_train+N_val],
        'sparsity_estimate': sparsity_estimate,
    }
    print("--- %s seconds ---" % (time.time() - start_time))

    return data

################################################################

def plot_sampled_distribution(data):
    '''
    data: returned by sample_dataset()
    '''
    count_table = np.zeros((data['num_attr_vals']**data['num_attributes']+1, (data['num_attr_vals']**data['num_attributes'])**2))
    for q, k in data['train_datapoints']:
        count_table[k, q] += 1

    xy, xyind, xy_div_xyind = get_distribution(count_table, distribution_epsilon=0.0)
    plot_distribution(xy, xy_div_xyind, 'sampled', figsize=(20,15))