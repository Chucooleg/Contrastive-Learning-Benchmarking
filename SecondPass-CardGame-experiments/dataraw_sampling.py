'''
dataraw_sampling_NumberShatter
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

################################################################################################

def derive_shatter_bucket_probs(num_cards):
    # (81 choose 1)*1, (81 choose 2)*2, .... (81 choose 81)*81
    num_datapoints_by_bucket = np.array([ncr(num_cards, subset_size) for subset_size in range(1, num_cards+1)])
    bucket_probs = num_datapoints_by_bucket / np.sum(num_datapoints_by_bucket)
    return bucket_probs.astype(float)

def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom

def sample_keys_column(num_attributes, num_attr_vals, bucket_probs):
    sampled_bucket = sample_shattering_bucket(num_attributes, num_attr_vals, bucket_probs)
    sampled_bucket_cards = sample_subset_in_bucket(num_attributes, num_attr_vals, sampled_bucket)
    # list, int
    return sampled_bucket_cards, sampled_bucket

def sample_subset_in_bucket(num_attributes, num_attr_vals, sampled_bucket):
    sampled_bucket_cards = np.random.choice(
        a=num_attr_vals ** num_attributes, size=sampled_bucket, replace=False)
    # list
    return sampled_bucket_cards.tolist()
    
def sample_shattering_bucket(num_attributes, num_attr_vals, bucket_probs):
    sampled_bucket = np.random.choice(a=np.arange(1, (num_attr_vals ** num_attributes + 1)), p=bucket_probs)
    # int
    return int(sampled_bucket)

################################################################################################

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def sample_queries(
    num_attributes, num_attr_vals,
    N_train, N_val, N_test):
    '''
    For debugging
    '''
    num_keys = num_attr_vals ** num_attributes
    
    N = N_train + N_val + N_test

    val_tokens = []
    val_gt_idxs = []

    test_marginal_tokens = []
    test_marginal_gt_idxs = []

    buckets = {}
    
    bucket_probs = derive_shatter_bucket_probs(num_keys)
    base_vocab_size =  num_keys
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


    ##########################################
    # # Probs : Full subset

    # s = [i for i in range(num_keys)]
    # power_set = list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))
    # all_subsets = power_set[1:]
    # probs = np.array([len(subset) for subset in all_subsets]) / np.sum(np.array([len(subset) for subset in all_subsets]))

    # max_len_q = 0
    ##########################################
    # # Val : Full Ground-truth


    # for i in tqdm(range(len(all_subsets))):

    #     gt_ks_idx = list(all_subsets[i])
    #     bucket = len(gt_ks_idx)
    #     for k_idx in gt_ks_idx:
    #         q_tokens = gt_ks_idx

    #         # accumulate datapoints
    #         val_tokens.append((q_tokens, [k_idx]))
    #         val_gt_idxs.append(gt_ks_idx)
            
    #         num_or_ops = 0
    #         num_and_ops = 0
    #         depth = 0           

    #         # stats
    #         max_len_q = max(max_len_q, len(q_tokens))
    #         input_lens[len(q_tokens)] = input_lens.get(len(q_tokens), 0) + 1
    #         buckets[bucket] = buckets.get(bucket, 0) + 1
    #         or_ops[num_or_ops] = or_ops.get(num_or_ops, 0) + 1
    #         and_ops[num_and_ops] = and_ops.get(num_and_ops, 0) + 1
    #         depths[depth] = depths.get(depth, 0) + 1  
    ##########################################
    # # Train : Sample from Full Ground-truth

    # for i in tqdm(range(4608)):

    #     idx = np.random.choice(len(val_tokens))
    #     train_tokens.append(val_tokens[idx])
    #     train_gt_idxs.append(val_gt_idxs[idx])
        
    #     q_tokens = val_tokens[idx][0]
    #     num_or_ops = 0
    #     num_and_ops = 0
    #     depth = 0           

    #     # stats
    #     max_len_q = max(max_len_q, len(q_tokens))
    #     input_lens[len(q_tokens)] = input_lens.get(len(q_tokens), 0) + 1
    #     buckets[bucket] = buckets.get(bucket, 0) + 1
    #     or_ops[num_or_ops] = or_ops.get(num_or_ops, 0) + 1
    #     and_ops[num_and_ops] = and_ops.get(num_and_ops, 0) + 1
    #     depths[depth] = depths.get(depth, 0) + 1    
    ##########################################
    # Train : Sample from all subsets

    # for i in tqdm(range(4608)):
    #     gt_ks_idx = list(np.random.choice(a=all_subsets, p=probs))
    #     bucket = len(gt_ks_idx)
    #     k_idx = int(np.random.choice(gt_ks_idx))
        
    #     q_tokens = gt_ks_idx
    #     num_or_ops = 0
    #     num_and_ops = 0
    #     depth = 0
                
    #     # accumulate datapoints
    #     train_tokens.append((q_tokens, [k_idx]))
    #     train_gt_idxs.append(gt_ks_idx)

    #     # stats
    #     max_len_q = max(max_len_q, len(q_tokens))
    #     input_lens[len(q_tokens)] = input_lens.get(len(q_tokens), 0) + 1
    #     buckets[bucket] = buckets.get(bucket, 0) + 1
    #     or_ops[num_or_ops] = or_ops.get(num_or_ops, 0) + 1
    #     and_ops[num_and_ops] = and_ops.get(num_and_ops, 0) + 1
    #     depths[depth] = depths.get(depth, 0) + 1   
    
    ##########################################
    # Probs : Buckets 

    bucket_probs = derive_shatter_bucket_probs(num_keys)

    # Train : Sample from Bucket, Sample from Column
    max_len_q = num_keys
    for i in tqdm(range(N_val+N_test)):
        # list, int
        gt_ks_idx, bucket = sample_keys_column(num_attributes, num_attr_vals, bucket_probs)
        k_idx = int(np.random.choice(gt_ks_idx))

        q_tokens = gt_ks_idx

        # accumulate datapoints
        val_tokens.append((q_tokens, [k_idx]))
        val_gt_idxs.append(gt_ks_idx)

        # stats
        max_len_q = max(max_len_q, len(q_tokens))
        buckets[bucket] = buckets.get(bucket, 0) + 1


    marginal_bucket_probs = np.array([1.0/len(bucket_probs)] * len(bucket_probs))

    for i in tqdm(range(N_test)):
        # list, int
        gt_ks_idx, bucket = sample_keys_column(num_attributes, num_attr_vals, marginal_bucket_probs)
        q_tokens = gt_ks_idx

        # accumulate datapoints
        test_marginal_tokens.append((q_tokens,))
        test_marginal_gt_idxs.append(gt_ks_idx)

    ##########################################

    data = {
        'num_attributes':num_attributes,
        'num_attr_vals':num_attr_vals,
        'key_support_size': num_keys,
        'max_len_q': max_len_q,
        'len_k': 1,
        
        #################################        
        'train_gt_idxs': [],
        'val_gt_idxs': val_gt_idxs[:N_val],
        'test_gt_idxs': val_gt_idxs[N_val:],
        
        'train_tokens': [],
        'val_tokens': val_tokens[:N_val],
        'test_tokens': val_tokens[N_val:],

        'test_marginal_tokens': test_marginal_tokens,
        'test_marginal_gt_idxs': test_marginal_gt_idxs,
        
        #################################

        'vocab_size': base_vocab_size + len(symbol_vocab_token_lookup),
        'symbol_vocab_token_lookup': symbol_vocab_token_lookup,
        'vocab_by_property': False,
        
        #################################
    }

    stats = {'buckets':buckets}

    return data, stats