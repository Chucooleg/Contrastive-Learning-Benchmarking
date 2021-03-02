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