import os
import numpy as np
import matplotlib.pyplot as plt
import operator as op
from functools import reduce


def get_image(show_img, save_dir, filename, img_arr):
    if show_img:
        plt.imshow(img_arr)
    else:
        plt.imsave(os.path.join(save_dir, filename), img_arr)


def plot_distribution(xy, xy_div_xyind, dist_name, figsize=(10,15), x_range_start=0, x_range_end=300, show_img=True, save_dir=None):
    figrange = (0, xy.shape[0], 0, xy.shape[1])

    plt.figure(figsize = figsize)
    plt.title(dist_name+' xy, All {} quries'.format(xy.shape[1]))
    get_image(
        show_img, save_dir, f'{dist_name}_xy.png', 
        img_arr=(xy)[figrange[0]:figrange[1], figrange[2]:figrange[3]]
    )

    plt.figure(figsize = figsize)
    plt.title(dist_name+' xy_div_xyind, All {} quries'.format(xy.shape[1]))
    get_image(
        show_img, save_dir, f'{dist_name}_xy_div_xyind.png', 
        img_arr=(xy_div_xyind)[figrange[0]:figrange[1], figrange[2]:figrange[3]]
    )

    if xy.shape[1] > (x_range_end - x_range_start):
        plt.figure(figsize = figsize)
        plt.title(dist_name+' xy, {}th-{}th queries'.format(x_range_start,x_range_end))
        get_image(
            show_img, save_dir, f'{dist_name}_xy_({x_range_start}-{x_range_end}).png', 
            img_arr=(xy)[figrange[0]:figrange[1], x_range_start:x_range_end]
        )

        plt.title(dist_name+' xy_div_xyind, {}th-{}th queries'.format(x_range_start,x_range_end))
        plt.figure(figsize = figsize)
        get_image(
            show_img, save_dir, f'{dist_name}_xy_div_xyind_({x_range_start}-{x_range_end}).png', 
            img_arr=(xy_div_xyind)[figrange[0]:figrange[1], x_range_start:x_range_end]
        )


def get_distribution(count_table, distribution_epsilon=0.0):
    xy = count_table/np.sum(count_table)
    xy += distribution_epsilon
    xy /= np.sum(xy)
    x = np.sum(xy,0)
    y = np.sum(xy,1)
    xyind = y[None].T @ x[None]
    return xy, xyind, xy/xyind


def report_countable_distribution(count_table, distribution_epsilon=0.0):
    sparsity = np.sum(count_table) * 1.0 / (count_table.shape[0] * count_table.shape[1])
    xy, xyind, xy_div_xyind = get_distribution(count_table, distribution_epsilon)
    xy_rank = np.linalg.matrix_rank(xy)
    xy_div_xyind_rank = np.linalg.matrix_rank(xy_div_xyind)
    
    distribution = {
        'shape': xy.shape,
        'size': xy.shape[0] * xy.shape[1],
        'sparsity': sparsity,
        'xy_rank': xy_rank,
        'xy_div_xyind_rank': xy_div_xyind_rank
    }
    
    return count_table, xy, xyind, xy_div_xyind, distribution


def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom


def derive_game_setup_stats(num_attributes, num_attr_vals):
    
    num_all_cards = num_attr_vals ** num_attributes
    shared_attr_to_num_queries = {}
    shared_attr_to_num_matches = {}
    shared_attr_to_total_num_matches = {}
    
    for n_shared_attr in range(num_attributes+1):
        n_queries = num_all_cards * ncr(num_attributes, n_shared_attr) *  (num_attr_vals-1)**(num_attributes - n_shared_attr)
        shared_attr_to_num_queries[n_shared_attr] = n_queries
        sum_ = 0
        if n_shared_attr == 0:
            sum_ = 1
        else:
            for i in range(n_shared_attr):
                sign = (-1)**(i)
                term = num_attr_vals**(num_attributes-(i+1)) * ncr(n_shared_attr, i+1)
                sum_ += sign * term
        shared_attr_to_num_matches[n_shared_attr] = sum_
        shared_attr_to_total_num_matches[n_shared_attr] = n_queries * sum_
    
    total_queries_with_matches = sum([shared_attr_to_num_queries[k] for k in shared_attr_to_num_queries])
    total_queries_with_real_matches = total_queries_with_matches - shared_attr_to_num_queries[0]
    total_matches = sum([
        shared_attr_to_num_queries[i]*shared_attr_to_num_matches[i] for i in range(num_attributes+1)
    ])
    
    n_queries = (num_all_cards) * (num_all_cards)
    
    stats = {
        'num_cards': num_all_cards + 1,
        'num_queries': n_queries,
        'matrix_size': (num_all_cards + 1) * n_queries,
        'total_matches': total_matches,
        'sparsity': total_matches * 1.0 / ((num_all_cards + 1) * n_queries),
        'total_queries_with_matches':total_queries_with_matches,
        'total_queries_with_real_matches':total_queries_with_real_matches,
        'percentage_queries_with_real_matches':total_queries_with_real_matches/n_queries,
        'query_shared_attr_to_num_queries':shared_attr_to_num_queries,
        'query_shared_attr_to_num_matches':shared_attr_to_num_matches,
        'query_shared_attr_to_total_num_matches':shared_attr_to_total_num_matches
    }
    
    return stats