from itertools import chain, combinations
import numpy as np


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


LOOKUP10 = {i:q for i, q in enumerate(powerset([i for i in range(10)]))}
LOOKUP10[0] = (10,)

################################################################################################
def decode_key_to_vocab_token(num_attributes, num_attr_vals, key_idx):
    return [key_idx]

def decode_query_to_vocab_token(num_attributes, num_attr_vals, num_cards_per_query, query_idx, nest_depth_int):
    # HACK for now
    assert num_attr_vals == 10
    return LOOKUP10[query_idx]

################################################################################################
def evaluate_query_idx(num_attributes, num_attr_vals, num_cards_per_query, query_idx, nest_depth_int):
    raise NotImplementedError


def construct_full_matrix(num_attributes, num_attr_vals, num_cards_per_query, nest_depth_int):
    K = num_attr_vals
    queries = list(powerset([i for i in range(K)]))
    num_queries = len(queries)
    count_table = np.zeros((K+1, num_queries)) # with null card

    for q_i, q in enumerate(queries):
        if not q:  # null card for empty set
            count_table[K, q_i] += 1 
        for k in q:
            count_table[k, q_i] += 1 
            
    return count_table

def report_countable_distribution(count_table):
    num_keys = count_table.shape[0]
    num_queries = count_table.shape[1]
    xy = count_table/np.sum(count_table)
    xy /= np.sum(xy)
    x = np.sum(xy,0)
    y = np.sum(xy,1)
    xyind = y[None].T @ x[None]
    xy_div_xyind = xy / xyind
    sparsity = np.sum(count_table) / (xy.shape[0] * xy.shape[1])
    tot_size = num_keys * num_queries
    distribution = {
        'num_keys': num_keys,
        'num_queries': num_queries,
        'tot_size': tot_size,
        'shape': xy.shape,
        'sparsity': sparsity,
        'xy_rank': num_keys, # SET is full rank
        'xy_div_xyind_rank': num_keys # SET is full rank
    } 
    return count_table, xy, xyind, xy_div_xyind, distribution

################################################################################################

def gen_full_matrices(num_attributes, num_attr_vals, num_cards_per_query, nest_depth_int):
    count_table = construct_full_matrix(num_attributes, num_attr_vals, num_cards_per_query, nest_depth_int)
    return report_countable_distribution(count_table)


def gen_full_dataset(num_attributes, num_attr_vals, num_cards_per_query, nest_depth_int):
    
    K = num_attr_vals
    queries = list(powerset([i for i in range(K)]))
    num_queries = len(queries)
    datapoints = []
    tokens = []
    lens = []
    count_table = np.zeros((K+1, num_queries)) # null card +1

    for q_i, q in enumerate(queries):
        
        if not q: # null card for empty set
            lens.append(1)
            datapoints.append((q_i, K))
            tokens.append(((K,), (K,)))
            count_table[K, q_i] += 1 
        else:
            lens.append(len(q))
            for k in q:
                datapoints.append((q_i, k))
                tokens.append((q, (k,)))
                count_table[k, q_i] += 1 
    
    base_vocab_size = K+1
    
    data = {
        'num_attributes': 1,
        'num_attr_vals': K,
        'num_cards_per_query': K,
        'nest_depth_int': 1,
        'key_support_size': K+1,
        'query_support_size': num_queries,
        'max_len_q': K,
        'len_k': 1,
        'hold_out': False,
        'train_datapoints': datapoints,
        'val_datapoints': None,
        'train_tokens': tokens,
        'val_tokens': None,
        'sparsity_estimate': sum(lens) / ((K+1) * num_queries),
        'vocab_size': base_vocab_size + 7,
        '(': base_vocab_size,
        ')': base_vocab_size + 1,
        'NULL': base_vocab_size + 2,
        'SEP': base_vocab_size + 3,
        'SOS': base_vocab_size + 4,
        'EOS': base_vocab_size + 5,
        'PAD': base_vocab_size + 6,
    }
    
    return data

