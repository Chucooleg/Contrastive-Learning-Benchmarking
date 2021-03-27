'''
dataraw_sampling_SETShatter
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


def resolve_prop_fn(num_attributes, num_attr_vals, query_part1, query_part2, cardpair_answer_lookup):
    if isinstance(query_part1, np.ndarray) and isinstance(query_part2, np.ndarray):
        key_properties = SET_cardpair_resolve_fn(query_part1, query_part2)
        return key_properties # nd.array
    elif isinstance(query_part1, list) and isinstance(query_part2, np.ndarray):
        assert query_part1[0] == 'OR'
        # list
        return ['OR'] + [SET_cardpair_resolve_fn(i_props, j_props) 
                         for i_props, j_props in zip(query_part1[1:], [query_part2]*(len(query_part1)-1))]
    elif isinstance(query_part1, np.ndarray) and isinstance(query_part2, list):
        assert query_part2[0] == 'OR'
        # list
        return ['OR'] + [SET_cardpair_resolve_fn(i_props, j_props) 
                         for i_props, j_props in zip([query_part1]*(len(query_part2)-1), query_part2[1:])]
    else:
        assert len(query_part1) == len(query_part2)
        # list     
        return ['OR'] + [SET_cardpair_resolve_fn(i_props, j_props) for i_props, j_props in zip(query_part1[1:], query_part2[1:])]


def resolve_idx_fn(num_attributes, num_attr_vals, query_part1, query_part2, cardpair_answer_lookup):
    if isinstance(query_part1, int) and isinstance(query_part2, int):
        # int
        return cardpair_answer_lookup[(min(query_part1, query_part2), max(query_part1, query_part2))]
    elif isinstance(query_part1, list) and isinstance(query_part2, int):
        assert query_part1[0] == 'OR'
        # list
        return ['OR'] + [cardpair_answer_lookup[(min(i, j), max(i, j))] 
                         for i, j in zip(query_part1[1:], [query_part2]*(len(query_part1)-1))]
    elif isinstance(query_part1, int) and isinstance(query_part2, list):
        assert query_part2[0] == 'OR'
        # list
        return ['OR'] + [cardpair_answer_lookup[(min(i, j), max(i, j))] 
                         for i, j in zip([query_part1]*(len(query_part2)-1), query_part2[1:])]
    else:
        assert len(query_part1) == len(query_part2)
        # list
        return ['OR'] + [cardpair_answer_lookup[(min(i, j), max(i, j))] for i, j in zip(query_part1[1:], query_part2[1:])]


def resolve_eval_expression(num_attributes, num_attr_vals, eval_expression, resolve_fn, cardpair_answer_lookup=None):
    '''
    eval_expression: a list of lists. 
    
    Use apppropriate cardpair_resolve_fn and cardpair_answer_lookup if the cards are represented by their integer indices
    e.g. ['AND', int, int, int]
    e.g. ['AND', ['OR', int, int], ['AND', int, int], int]
    
    Use apppropriate cardpair_resolve_fn if the cards are represented by their property arrays
    e.g. ['AND', array([0, 1, 2, 0]), array([0, 0, 1, 0]), array([0, 2, 1, 2])]
    e.g. ['AND', ['OR', array([0, 1, 0, 2]), array([0, 2, 1, 0])], ['AND', array([2, 2, 0, 2]), array([0, 0, 0, 0])], array([1, 0, 1, 0])]  
    
    Returns:
        int, if the cards are represented by their integer indices
        np.ndarray, if the cards are represented by their property arrays
    '''
    operator = eval_expression[0]
    assert operator in ('OR', 'AND')
    if operator == 'OR':
        ans = eval_expression
    else:
        ans = None
        for part in eval_expression[1:]:
            if isinstance(part, list):
                part = resolve_eval_expression(
                    num_attributes, num_attr_vals, part, resolve_fn, cardpair_answer_lookup)
            if ans is not None:
                res = resolve_fn(num_attributes, num_attr_vals, ans, part, cardpair_answer_lookup)
            else:
                res = part
            ans = res
    return ans

################################################################################################

def convert_eval_expression_to_properties(num_attributes, num_attr_vals, eval_expression_substituted):
    eval_expression_properties = []
    for part in eval_expression_substituted:
        if isinstance(part, str):
            eval_expression_properties.append(part)
        elif isinstance(part, int):
            eval_expression_properties.append(decode_key_idx(num_attributes, num_attr_vals, part))
        else:
            assert isinstance(part, list)
            substituted_part = convert_eval_expression_to_properties(
                num_attributes, num_attr_vals, part)
            eval_expression_properties.append(substituted_part)
    return eval_expression_properties

def weave(values_list, sym):
    return list(itertools.chain.from_iterable(zip(values_list, [sym] * len(values_list))))[:-1]

def substitude_DERIVE_into_input_expression(DERIVE, query_input_expression):
    derive_pos = query_input_expression.index('DERIVE')
    if isinstance(DERIVE, list):
        assert DERIVE[0] == 'OR'
        DERIVE_input_expression = ['('] + weave([p for p in DERIVE[1:]], '|') + [')']
        try:
            query_input_expression = (
                query_input_expression[:derive_pos] + DERIVE_input_expression + query_input_expression[derive_pos+1:]
            )
        except:
            breakpoint()
    else:
        assert isinstance(DERIVE, int)
        query_input_expression[derive_pos] = DERIVE
    return query_input_expression

################################################################################################

def derive_missing_query_part(
    num_attributes, num_attr_vals, lhs_expression,
    derive_positions_stack, rhs, expression_resolve_fn, cardpair_answer_lookup, debug=False):
    
    if lhs_expression == 'DERIVE':
        return rhs
    
    if debug:
        print('lhs_expression', lhs_expression)
    assert lhs_expression[0] in ('OR', 'AND')
    
    if lhs_expression[0] == 'OR':
        return lhs_expression
    else:
        derive_pos = derive_positions_stack.pop()
        left_of_derive = lhs_expression[:derive_pos] # include 'AND' at the front
        right_of_derive = lhs_expression[derive_pos+1:]
        right_of_derive.reverse()
        
        if right_of_derive:
            if debug: 
                print('right_of_derive:', right_of_derive[::-1])
            rhs = expression_resolve_fn(
                num_attributes=num_attributes, 
                num_attr_vals=num_attr_vals, 
                eval_expression= ['AND', rhs] + right_of_derive,
                resolve_fn=resolve_idx_fn,
                cardpair_answer_lookup=cardpair_answer_lookup
            )

            if debug: 
                print('rhs after solving right_of_derive:', rhs)
        
        if len(left_of_derive) > 1:
            if debug: 
                print('left_of_derive:', left_of_derive)
            left_evaluated = expression_resolve_fn(
                num_attributes=num_attributes, 
                num_attr_vals=num_attr_vals, 
                eval_expression=left_of_derive,
                resolve_fn=resolve_idx_fn,
                cardpair_answer_lookup=cardpair_answer_lookup
            )
            if debug: 
                print('left_evaluated:', left_evaluated)
            
            rhs = expression_resolve_fn(
                num_attributes=num_attributes, 
                num_attr_vals=num_attr_vals, 
                eval_expression= ['AND', rhs, left_evaluated],
                resolve_fn=resolve_idx_fn,
                cardpair_answer_lookup=cardpair_answer_lookup
            ) 
            if debug: 
                print('rhs after solving rhs with left_of_derive:', rhs)

        # should have one lhs expression and one rhs term now
        return derive_missing_query_part(
            num_attributes=num_attributes, 
            num_attr_vals=num_attr_vals,
            lhs_expression=lhs_expression[derive_pos],
            derive_positions_stack=derive_positions_stack,
            rhs=rhs,
            expression_resolve_fn=expression_resolve_fn,
            cardpair_answer_lookup=cardpair_answer_lookup,
            debug=debug)

################################################################################################

def nest_flat_query_parts(
    num_attributes, num_attr_vals, flat_parts, depth, max_depth):
    depth += 1
    
    # base case
    if len(flat_parts) == 1:
        depth -= 1
        
        if flat_parts[0] == 'DERIVE':
            input_expression = flat_parts[0]
            has_derive = True
            pos_derive = []
        elif isinstance(flat_parts[0], list):
            assert flat_parts[0][0] == 'OR'
            input_expression = ['('] + weave([p for p in flat_parts[0][1:]], '|') + [')']
            has_derive = False
            pos_derive = []
        else:
            input_expression = flat_parts[0]
            has_derive = False
            pos_derive = []
            
        eval_expression = flat_parts[0]
        
    # base case    
    elif len(flat_parts) == 2 or depth == max_depth:
        eval_expression = ['AND'] + flat_parts
        
        input_expression = ['(']
        for part in flat_parts:
            if isinstance(part, list):
                assert part[0] == 'OR'
                OR_part = ['('] + weave([p for p in part[1:]], '|') + [')']
                input_expression = input_expression + OR_part + ['&']
            else:
                input_expression = input_expression + [part] + ['&']
        input_expression[-1] = ')'
        
        if 'DERIVE' in flat_parts:
            has_derive = True
            pos_derive = [flat_parts.index('DERIVE')+1]
        else:
            has_derive = False
            pos_derive = []
        
    # recursive case
    else:
        mid_idx = len(flat_parts) // 2
        
        L_e, L_i, _, L_has_derive, L_pos_derive = \
            nest_flat_query_parts(
                num_attributes, num_attr_vals,
                flat_parts[:mid_idx], depth, max_depth
        )
        
        R_e, R_i, depth, R_has_derive, R_pos_derive = \
            nest_flat_query_parts(
                num_attributes, num_attr_vals,
                flat_parts[mid_idx:], depth, max_depth
        )        
        
        eval_expression = ['AND', L_e, R_e]
        input_expression = (
            ['('] + (L_i if isinstance(L_i, list) else [L_i]) + ['&'] + \
            (R_i if isinstance(R_i, list) else [R_i]) + [')']
        )
        has_derive = L_has_derive or R_has_derive
        
        if not has_derive:
            pos_derive = []
        else:
            assert not (L_has_derive and R_has_derive)
            if L_has_derive:
                if L_pos_derive == []: assert L_e == 'DERIVE'
                L_pos_derive.append(1)
                pos_derive = L_pos_derive
            else:
                if R_pos_derive == []: assert R_e == 'DERIVE'
                R_pos_derive.append(2)
                pos_derive = R_pos_derive
    
    return eval_expression, input_expression, depth, has_derive, pos_derive

################################################################################################

def sample_OR_cards(num_cards, gt_keys_idx, pos_in_query, DERIVE_pos):
    if pos_in_query == DERIVE_pos:
        return 'DERIVE'
    else:
        cards_in_OR_set = np.random.choice(a=num_cards, size=len(gt_keys_idx), replace=False).tolist()
        return ['OR'] + cards_in_OR_set


def sample_query_based_on_gt_keys(
    num_attributes, num_attr_vals, num_cards_per_query, nest_depth_int, gt_keys_idx, 
    multiple_OR_sets_bool, expression_resolve_fn, cardpair_answer_lookup, symbol_vocab_token_lookup, validate=False, debug=False):

    if debug: 
        print('\ngt_keys_idx', gt_keys_idx)
        print('\ngt key properties', [decode_key_idx(num_attributes, num_attr_vals, k_idx) for k_idx in gt_keys_idx])
    
    num_cards = num_attr_vals ** num_attributes
    
    # number of OR sets to be inserted
    if multiple_OR_sets_bool:
        num_OR_sets_in_query = int(np.random.choice(a=np.arange(1, (num_cards_per_query//len(gt_keys_idx))+1)))
        num_OR_cards = num_OR_sets_in_query * len(gt_keys_idx)
    else:
        num_OR_sets_in_query = 1
        num_OR_cards = len(gt_keys_idx)
    
    # initial slots
    num_and_slots = num_cards_per_query - num_OR_cards + num_OR_sets_in_query
    initial_card_indices = np.random.choice(
        a=num_cards, size=num_and_slots
    )
    # which positions to insert OR sets
    OR_positions = np.random.choice(a=initial_card_indices.shape[0], size=num_OR_sets_in_query, replace=False)
    DERIVE_position = np.random.choice(a=OR_positions)
    
    # fill in OR slots, if there's more than 1
    flat_card_indices = [
        sample_OR_cards(num_cards, gt_keys_idx, i, DERIVE_position) 
        if (i in OR_positions) else int(cidx)
        for i, cidx in enumerate(initial_card_indices)
    ]
    if debug: 
        print('\nflat_card_indices', flat_card_indices)
    
    # nest the query
    query_eval_expression, query_input_expression, depth, has_part_to_be_derived, derive_positions_stack = \
        nest_flat_query_parts(
            num_attributes=num_attributes, 
            num_attr_vals=num_attr_vals,
            flat_parts=flat_card_indices, 
            depth=0, 
            max_depth=nest_depth_int,
        )
    if debug: 
        print('\nquery_eval_expression', query_eval_expression)
        print('\nquery_input_expression', query_input_expression)
    
    # get 'DERIVE'
    DERIVE = derive_missing_query_part(
        num_attributes=num_attributes, 
        num_attr_vals=num_attr_vals,
        lhs_expression=query_eval_expression,
        derive_positions_stack=derive_positions_stack.copy(), # remove copy() when validated
        rhs=(['OR'] + gt_keys_idx) if (len(gt_keys_idx) > 1) else gt_keys_idx[0],
        expression_resolve_fn=expression_resolve_fn,
        cardpair_answer_lookup=cardpair_answer_lookup,
        debug=debug)
    
    if isinstance(query_input_expression, str):
        query_input_expression = [query_input_expression]
        query_eval_expression = ['AND', query_eval_expression]
        derive_positions_stack = [1]
       
    if debug: 
        print('\nquery_eval_expression', query_eval_expression)
        print('\nquery_input_expression', query_input_expression)
    
    # substitude 'DERIVE' answer into input expression
    query_input_expression = substitude_DERIVE_into_input_expression(DERIVE, query_input_expression)
    if debug: 
        print('\nDERIVE',DERIVE)
        print('\nsubstituted query_input_expression', ''.join([str(i) for i in query_input_expression]))
    
    ############################################################

    if validate:

        # NOTE: validate, can remove to speedup
        # validate complete query indeed eval back to gt keys
        eval_expression_substituted = query_eval_expression.copy()
        derive_positions_stack_c = derive_positions_stack.copy()
        if len(derive_positions_stack_c) == 1:
            query_eval_expression[derive_positions_stack_c.pop()] = DERIVE
        else:
            while len(derive_positions_stack_c) > 1:
                pop_idx = derive_positions_stack_c.pop()
                eval_expression_substituted = eval_expression_substituted[pop_idx]
            eval_expression_substituted[derive_positions_stack_c.pop()] = DERIVE
        resolved_keys_idx = resolve_eval_expression(
                num_attributes, num_attr_vals, query_eval_expression, resolve_idx_fn, cardpair_answer_lookup)
        if isinstance(resolved_keys_idx, int):
            assert gt_keys_idx == [resolved_keys_idx]
        else:
            assert gt_keys_idx == resolved_keys_idx[1:]


        # NOTE: validate, can remove to speedup
        # convert input expression from card indices to card properties
        eval_expression_properties = convert_eval_expression_to_properties(
            num_attributes, num_attr_vals,
            query_eval_expression)
        if debug: 
            print('\neval_expression_properties', eval_expression_properties)
        resolved_keys_props = resolve_eval_expression(
                num_attributes, num_attr_vals, eval_expression_properties, resolve_prop_fn, cardpair_answer_lookup)
        if isinstance(resolved_keys_props, np.ndarray):
            resolved_keys_idx = [encode_key_idx(num_attributes, num_attr_vals, resolved_keys_props)]
        else:
            resolved_keys_idx = [encode_key_idx(num_attributes, num_attr_vals, k_prop) for k_prop in resolved_keys_props[1:]]
        assert gt_keys_idx == resolved_keys_idx


    ############################################################
    
    # convert input expression from card indices to vocab tokens
    query_for_input_vocab_tokens = [sym if isinstance(sym, int) else symbol_vocab_token_lookup[sym] for sym in query_input_expression]
    if debug: 
        print('\nquery_for_input_vocab_tokens', query_for_input_vocab_tokens)
    
    num_or_ops = num_OR_cards - num_OR_sets_in_query
    num_and_ops = num_and_slots - 1
    return query_for_input_vocab_tokens, num_or_ops, num_and_ops, depth


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


def sample_one_training_datapoint(
    bucket, num_attributes, num_attr_vals, query_length_multiplier, nest_depth_int, multiple_OR_sets_bool,
    cardpair_answer_lookup, symbol_vocab_token_lookup,
    validate=False, debug=False):

    gt_ks_idx = sample_subset_in_bucket(num_attributes, num_attr_vals, bucket)
    k_idx = int(np.random.choice(gt_ks_idx))

    # determine number of cards in query
    num_cards_per_query = (
        (num_attr_vals ** num_attributes) * 
        (query_length_multiplier if query_length_multiplier else np.random.choice(a=[1,2,3,4,5]))
    )
    
    # sample a query for this column of keys
    q_tokens, _, _, _ = sample_query_based_on_gt_keys(
        num_attributes=num_attributes, 
        num_attr_vals=num_attr_vals, 
        num_cards_per_query=num_cards_per_query, 
        nest_depth_int=nest_depth_int, 
        gt_keys_idx=gt_ks_idx, 
        multiple_OR_sets_bool=multiple_OR_sets_bool,
        expression_resolve_fn=resolve_eval_expression,
        cardpair_answer_lookup=cardpair_answer_lookup,
        symbol_vocab_token_lookup=symbol_vocab_token_lookup,
        validate=validate,
        debug=debug
    )   

    return q_tokens, k_idx


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def sample_queries(
    num_attributes, query_length_multiplier, nest_depth_int, multiple_OR_sets_bool, 
    expression_resolve_fn,
    N_train, N_val, N_test,
    validate=False,
    debug=False):
    '''
    query_length_multiplier: int. 0 would sample the multiplier from 1 to 5.
    '''
    num_attr_vals = 3
    num_keys = num_attr_vals ** num_attributes
    
    N = N_train + N_val + N_test
    tokens = []
    gt_idxs = []
    
    buckets = {}
    or_ops = {}
    and_ops = {}
    depths = {}
    input_lens = {}
    
    bucket_probs = derive_shatter_bucket_probs(num_keys)
    cardpair_answer_lookup = construct_cardpair_answer_lookup(num_attributes, num_attr_vals)
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

    max_len_q = 0
    for i in tqdm(range(N)):
        # each query column hit a number of keys
        gt_ks_idx, bucket = sample_keys_column(num_attributes, num_attr_vals, bucket_probs)
        
        # draw one key from this column of keys
        k_idx = int(np.random.choice(gt_ks_idx))
        
        # determine number of cards in query
        num_cards_per_query = (
            num_keys * (query_length_multiplier if query_length_multiplier else np.random.choice(a=[1,2,3,4,5]))
        )
        
        # sample a query for this column of keys
        q_tokens, num_or_ops, num_and_ops, depth = sample_query_based_on_gt_keys(
            num_attributes=num_attributes, 
            num_attr_vals=num_attr_vals, 
            num_cards_per_query=num_cards_per_query, 
            nest_depth_int=nest_depth_int, 
            gt_keys_idx=gt_ks_idx, 
            multiple_OR_sets_bool=multiple_OR_sets_bool,
            expression_resolve_fn=expression_resolve_fn,
            cardpair_answer_lookup=cardpair_answer_lookup,
            symbol_vocab_token_lookup=symbol_vocab_token_lookup,
            validate=validate,
            debug=debug
        )
        
        # accumulate datapoints
        tokens.append((q_tokens, [k_idx]))
        gt_idxs.append(gt_ks_idx)
        
        # stats
        max_len_q = max(max_len_q, len(q_tokens))
        input_lens[len(q_tokens)] = input_lens.get(len(q_tokens), 0) + 1
        buckets[bucket] = buckets.get(bucket, 0) + 1
        or_ops[num_or_ops] = or_ops.get(num_or_ops, 0) + 1
        and_ops[num_and_ops] = and_ops.get(num_and_ops, 0) + 1
        depths[depth] = depths.get(depth, 0) + 1


    data = {
        'num_attributes':num_attributes,
        'num_attr_vals':num_attr_vals,
        'nest_depth_int': nest_depth_int,
        'key_support_size': num_keys,
        'multiple_OR_sets_bool': multiple_OR_sets_bool,

        'query_length_multiplier': query_length_multiplier,
        'max_len_q': max_len_q,
        'len_k': 1,
        
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
        'vocab_by_property': False,
        
        #################################
    }

    stats = {
            'bucket_counts': buckets,
            'or_op_counts': or_ops,
            'and_op_counts': and_ops,
            'depth_counts': depths,
            'input_lens': input_lens,
        }

    return data, stats


