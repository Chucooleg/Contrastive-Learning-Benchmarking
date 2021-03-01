from collections import defaultdict, Counter
import numpy as np
import itertools
import copy
import random

################################################################################################
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
    key_properties = list(key_properties)

    if not key_properties:
        return num_attr_vals ** num_attributes
    else:
        key_idx = 0
        digit = 0
        while key_properties:
            attr_val = key_properties.pop()
            key_idx += attr_val * (num_attr_vals**digit)
            digit += 1
    return int(key_idx)

################################################################################################

def decode_query_to_vocab_token(num_attributes, num_attr_vals, num_cards_per_query, query_idx, nest_depth_int):
    '''
    convert query_idx into vocab tokens for training
    '''
    OpenP = num_attributes * num_attr_vals
    CloseP = num_attributes * num_attr_vals + 1
    _, nested_query_parts_for_input, _ = nest_query_parts_from_query_idx(
        num_attributes, num_attr_vals, num_cards_per_query, query_idx, nest_depth_int
    )
    tokens = []
    for part in nested_query_parts_for_input:
        if part == '(':
            tokens.append(OpenP)
        elif part == ')':
            tokens.append(CloseP)
        else:
            tokens += decode_key_properties_to_vocab_token(num_attributes, num_attr_vals, part)
    return tokens

def decode_query_idx_to_card_properties(num_attributes, num_attr_vals, num_cards_per_query, query_idx):
    query_parts_idx = decode_query_idx(num_attributes, num_attr_vals, num_cards_per_query, query_idx)
    query_properties = [decode_key_idx(num_attributes, num_attr_vals, idx) for idx in query_parts_idx]
    return query_properties
    
def decode_query_idx(num_attributes, num_attr_vals, num_cards_per_query, query_idx):
    '''
    decode query_idx into individual card indices
    '''
    num_cards = num_attr_vals ** num_attributes
    assert query_idx < num_cards ** num_cards_per_query
    query_parts_idx = []
    query_idx_copy = query_idx
    for i in range(num_cards_per_query):
        digit = query_idx_copy % num_cards
        query_parts_idx = [digit] + query_parts_idx
        query_idx_copy = query_idx_copy // num_cards
    assert len(query_parts_idx) == num_cards_per_query
    return np.array(query_parts_idx)

assert (decode_query_idx(num_attributes=4, num_attr_vals=3, num_cards_per_query=2, query_idx=100) == np.array([1,19])).all()
assert (decode_query_idx(num_attributes=4, num_attr_vals=3, num_cards_per_query=3, query_idx=1) == np.array([0, 0, 1])).all()
assert (decode_query_idx(num_attributes=4, num_attr_vals=3, num_cards_per_query=3, query_idx=82) == np.array([0, 1, 1])).all()
assert (decode_query_idx(num_attributes=4, num_attr_vals=3, num_cards_per_query=3, query_idx=531439) == np.array([80, 80, 79])).all()
assert (decode_query_idx(num_attributes=4, num_attr_vals=3, num_cards_per_query=3, query_idx=531440) == np.array([80, 80, 80])).all()

def encode_query_idx(num_attributes, num_attr_vals, num_cards_per_query, query_parts_idx, queryidx_base=None):
    '''
    reverse decode_query_idx()
    query_parts_idx: a list of card indices
    base: np.array([81**2, 81**1, 81**0 ...])
    '''
    assert isinstance(query_parts_idx, np.ndarray)
    assert num_cards_per_query == len(query_parts_idx)
    num_cards = num_attr_vals ** num_attributes
    if not queryidx_base:
        queryidx_base = np.array([num_cards**i for i in reversed(range(num_cards_per_query))])
    query_idx = sum(query_parts_idx * queryidx_base)    
    assert query_idx < num_cards ** num_cards_per_query
    return query_idx

assert (encode_query_idx(num_attributes=4, num_attr_vals=3, num_cards_per_query=2, query_parts_idx=np.array([1,19])) == 100).all()
assert (encode_query_idx(num_attributes=4, num_attr_vals=3, num_cards_per_query=3, query_parts_idx=np.array([0,0,1])) == 1).all()
assert (encode_query_idx(num_attributes=4, num_attr_vals=3, num_cards_per_query=3, query_parts_idx=np.array([0,1,1])) == 82).all()
assert (encode_query_idx(num_attributes=4, num_attr_vals=3, num_cards_per_query=3, query_parts_idx=np.array([80,80,79])) == 531439).all()
assert (encode_query_idx(num_attributes=4, num_attr_vals=3, num_cards_per_query=3, query_parts_idx=np.array([80,80,80])) == 531440).all()

################################################################################################

def split_at_mid(flat_query_parts, depth, max_depth):
    depth += 1
    if len(flat_query_parts) <= 2 or depth == max_depth:
        return flat_query_parts, ['('] + flat_query_parts + [')'], depth
    else:
        mid_idx = len(flat_query_parts) // 2
        L_e, L_i, _ = split_at_mid(flat_query_parts[:mid_idx], depth, max_depth)
        R_e, R_i, depth = split_at_mid(flat_query_parts[mid_idx:], depth, max_depth)
        return [L_e, R_e], ['('] + L_i + R_i + [')'], depth

def nest_query_parts(flat_query_parts, depth_int):
    '''
    flat_query_parts: such as [[array([0, 0, 0, 0]), array([0, 0, 0, 0])], array([0, 0, 0, 0])]]
    '''
    assert flat_query_parts and depth_int > 0
    nested_query_parts_for_eval, nested_query_parts_for_input, depth_int = split_at_mid(flat_query_parts, depth=0, max_depth=depth_int)  
    return nested_query_parts_for_eval, nested_query_parts_for_input, depth_int

nested_query_parts_for_eval, nested_query_parts_for_input, depth_int = nest_query_parts(flat_query_parts=[1,2,3,4,5,6,7,8,9,0], depth_int=3)

assert nested_query_parts_for_eval == [[[1, 2], [3, 4, 5]], [[6, 7], [8, 9, 0]]]
assert nested_query_parts_for_input == ['(', '(', '(', 1, 2, ')', '(', 3, 4, 5, ')', ')', '(', '(', 6, 7, ')', '(', 8, 9, 0, ')', ')', ')']

################################################################################################

def evaluate_nested_query_parts(nested_query_parts, resolve_fn):
    '''
    query_parts: a list of lists. 
    e.g. [array([0, 0, 0, 0]), array([0, 0, 0, 0]), array([0, 0, 0, 0])]
    e.g. [[array([0, 0, 0, 0])], [array([0, 0, 0, 0]), array([0, 0, 0, 0])]]
    ''' 
    stack = []
    nested_query_stack = nested_query_parts[::-1]
    while nested_query_stack:
        part = nested_query_stack.pop()
        if isinstance(part, list):
            part = evaluate_nested_query_parts(part, resolve_fn)
            if stack:
                last_part = stack.pop()
                res = resolve_fn(last_part, part)
            else:
                res = part
            stack.append(res)
        else:
            if not stack:
                stack.append(part)
            else:
                last_part = stack.pop()
                res = resolve_fn(last_part, part)
                stack.append(res)
    assert len(stack) == 1
    return stack[0]

def SET_derive_property(prop1, prop2):
    '''prop1, prop2 are integers'''
    props = {0,1,2}
    assert prop1 in props and prop2 in props    
    if prop1 == prop2: 
        return prop1
    else:
        return list(props - {prop1, prop2})[0]
    
def SET_resolve_fn(card1, card2):
    assert card1.shape == card2.shape
    return np.array([SET_derive_property(card1[i], card2[i]) for i in range(len(card1))])

assert (evaluate_nested_query_parts([np.array([1, 2, 0, 1]), np.array([0, 0, 1, 0])], SET_resolve_fn) == np.array([2, 1, 2, 2])).all()
assert (evaluate_nested_query_parts([np.array([1, 2, 0, 1]), np.array([0, 0, 1, 0]), np.array([1, 2, 0, 1])], SET_resolve_fn) == np.array([0, 0, 1, 0])).all()
assert (evaluate_nested_query_parts([np.array([2, 2, 0, 1]), [np.array([0, 0, 1, 0]), np.array([1, 2, 0, 1])]], SET_resolve_fn) == np.array([2, 0, 1, 0])).all()
assert (evaluate_nested_query_parts([[np.array([2, 2, 0, 1]), np.array([0, 0, 1, 0])], np.array([1, 2, 0, 1])], SET_resolve_fn) == np.array([1, 0, 1, 0])).all()

################################################################################################

def nest_query_parts_from_query_idx(num_attributes, num_attr_vals, num_cards_per_query, query_idx, nest_depth_int):
    flat_query_parts = decode_query_idx_to_card_properties(num_attributes, num_attr_vals, num_cards_per_query, query_idx)
    nested_query_parts_for_eval, nested_query_parts_for_input, depth_int = nest_query_parts(flat_query_parts, nest_depth_int)
    return nested_query_parts_for_eval, nested_query_parts_for_input, depth_int

def evaluate_query_idx(num_attributes, num_attr_vals, num_cards_per_query, query_idx, nest_depth_int):
    nested_query_parts, _, _ = nest_query_parts_from_query_idx(num_attributes, num_attr_vals, num_cards_per_query, query_idx, nest_depth_int)
    gt_key = evaluate_nested_query_parts(nested_query_parts, resolve_fn=SET_resolve_fn)
    return gt_key

def check_if_query_key_match_by_idx(num_attributes, num_attr_vals, num_cards_per_query, query_idx, key_idx, nest_depth_int):
    '''return an int'''
    gt_key = evaluate_query_idx(num_attributes, num_attr_vals, num_cards_per_query, query_idx, nest_depth_int)
    key = decode_key_idx(num_attributes, num_attr_vals, key_idx)
    if (gt_key == key).all():
        return 1
    else:
        return 0

def construct_full_matrix(num_attributes, num_attr_vals, num_cards_per_query, nest_depth_int):
    num_keys = num_attr_vals ** num_attributes
    num_queries = num_keys ** num_cards_per_query
    count_table = np.zeros((num_keys, num_queries))
    for q_idx in range(num_queries):
        gt_key = evaluate_query_idx(num_attributes, num_attr_vals, num_cards_per_query, q_idx, nest_depth_int)
        k_idx = encode_key_idx(num_attributes, num_attr_vals, gt_key)
        count_table[k_idx, q_idx] += 1
    tot_size = num_keys * num_queries
    sparsity = np.sum(count_table) * 1.0 / tot_size
    print('Constructed Full Matrix:')
    print(f'{num_keys} keys by {num_queries} queries')
    print(f'Total size {tot_size}')
    print(f'Sparsity {sparsity}')
    return count_table 

################################################################################################

def gen_full_dataset(num_attributes, num_attr_vals, num_cards_per_query, nest_depth_int):
    num_keys = num_attr_vals ** num_attributes
    num_queries = num_keys ** num_cards_per_query
    datapoints = []
    tokens = []
    for q_idx in range(num_queries):
        gt_key = evaluate_query_idx(num_attributes, num_attr_vals, num_cards_per_query, q_idx, nest_depth_int)
        k_idx = encode_key_idx(num_attributes, num_attr_vals, gt_key)
        q_tokens = decode_query_to_vocab_token(num_attributes, num_attr_vals, num_cards_per_query, q_idx, nest_depth_int)
        k_tokens = decode_key_to_vocab_token(num_attributes, num_attr_vals, k_idx)
        tokens.append((q_tokens, k_tokens))
        datapoints.append((q_idx, k_idx))

    data = {
        'num_attributes':num_attributes,
        'num_attr_vals':num_attr_vals,
        'num_cards_per_query': num_cards_per_query,
        'nest_depth_int': nest_depth_int,
        'key_support_size': num_keys,
        'query_support_size': num_queries,
        'hold_out': False,
        'train_datapoints': datapoints,
        'val_datapoints': None,
        'train_tokens': tokens,
        'val_tokens': None,
        'sparsity_estimate': 1.0 / num_queries,
        'vocab_size': num_attributes * num_attr_vals + 7,
        '(': num_attributes * num_attr_vals,
        ')': num_attributes * num_attr_vals + 1,
        'NULL': num_attributes * num_attr_vals + 2,
        'SEP': num_attributes * num_attr_vals + 3,
        'SOS': num_attributes * num_attr_vals + 4,
        'EOS': num_attributes * num_attr_vals + 5,
        'PAD': num_attributes * num_attr_vals + 6,
    }

    for k in data:
        if not 'datapoints' in k and not 'tokens' in k:
            print(k, data[k])

    return data
    
def gen_random_dataset(num_attributes, num_attr_vals, num_cards_per_query, nest_depth_int, N_train, N_val):
    num_keys = num_attr_vals ** num_attributes
    num_queries = num_keys ** num_cards_per_query
    datapoints = []
    tokens = []
    for _ in range(N_train + N_val):
        q_idx = random.randrange(num_queries)
        gt_key = evaluate_query_idx(num_attributes, num_attr_vals, num_cards_per_query, q_idx, nest_depth_int)
        k_idx = encode_key_idx(num_attributes, num_attr_vals, gt_key)
        q_tokens = decode_query_to_vocab_token(num_attributes, num_attr_vals, num_cards_per_query, q_idx, nest_depth_int)
        k_tokens = decode_key_to_vocab_token(num_attributes, num_attr_vals, k_idx)
        tokens.append((q_tokens, k_tokens))
        datapoints.append((q_idx, k_idx))
    data = {
        'num_attributes':num_attributes,
        'num_attr_vals':num_attr_vals,
        'num_cards_per_query': num_cards_per_query,
        'nest_depth_int': nest_depth_int,
        'key_support_size': num_keys,
        'query_support_size': num_queries,
        'hold_out': True,
        'train_datapoints': datapoints[:N_train],
        'val_datapoints': datapoints[N_train:N_train+N_val],
        'train_tokens': tokens[:N_train],
        'val_tokens': tokens[N_train:N_train+N_val],
        'sparsity_estimate': 1.0 / num_queries,
        'vocab_size': num_attributes * num_attr_vals + 7,
        '(': num_attributes * num_attr_vals,
        ')': num_attributes * num_attr_vals + 1,
        'NULL': num_attributes * num_attr_vals + 2,
        'SEP': num_attributes * num_attr_vals + 3,
        'SOS': num_attributes * num_attr_vals + 4,
        'EOS': num_attributes * num_attr_vals + 5,
        'PAD': num_attributes * num_attr_vals + 6,
    }

    for k in data:
        if not 'datapoints' in k and not 'tokens' in k:
            print(k, data[k])

    return data

################################################################################################
