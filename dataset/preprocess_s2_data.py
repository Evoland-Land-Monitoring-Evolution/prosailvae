#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon Oct 24 17:21:00 2022

@author: yoel
"""
import time
from datetime import datetime

import torch
import prosailvae
from dataset.loaders import get_mmdc_loaders
import torch
import os

from tqdm import tqdm
import time
import sys
import matplotlib.pyplot as plt

import argparse


thirdparties_path = "/home/yoel/Documents/Dev/PROSAIL-VAE/thirdparties/"
# thirdparties_path = "/work/scratch/zerahy/src/thirdparties/"

sys.path = [thirdparties_path + '/mmdc-singledate',
            thirdparties_path + '/sensorsio',
            thirdparties_path + '/torchutils/src'] + sys.path

from src.mmdc_singledate.datamodules.mmdc_datamodule import (IterableMMDCDataset,
                                                            worker_init_fn,
                                                            destructure_batch)

from src.mmdc_singledate.datamodules.components.datamodule_utils import (MMDCDataStats,
                                                        #OneSetMMDCDataclass,
                                                        average_stats,
                                                        compute_stats,
                                                        create_tensors_path)     


def get_S2_data_preprocess_parser():
    """
    Creates a new argument parser.
    """
    parser = argparse.ArgumentParser(description='Parser for data generation')
    parser.add_argument("-rsr", dest="rsr_dir",
        help="directory of rsr files",
        type=str, default='/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/')
    parser.add_argument("-d", dest="tensor_dir",
        help="directory of tensor files",
        type=str, default='/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/real_data/torchfiles/')
    parser.add_argument("-bpe", dest="batch_par_epoch",
        help="Number of 32x32 patchs per epochs",
        type=int, default=100)    
    return parser

def get_mmdc_dataset_mean_std(loader, max_batch=100):
    S1 = torch.zeros(10)
    S2 = torch.zeros(10)
    n = 0
    for idx, batch in zip(range(max_batch), loader):
        (s2_x, _, _, _, _, _, _) = destructure_batch(batch)
        s2_refl = s2_x.transpose(0,1).reshape(10,-1) / 10000
        S1 = S1 + s2_refl.sum(1)
        S2 = S2 + s2_refl.pow(2).sum(1)
        n = n + s2_refl.size(1)
    mean = S1 / n
    std = torch.sqrt(S2 / n - mean.pow(2))
    return mean, std

def normalize_patch_for_plot(s2_r_rgb, sr_min=None, sr_max=None):
    if len(s2_r_rgb.size())==3:
        assert s2_r_rgb.size(2)==3
        s2_r_rgb = s2_r_rgb.unsqueeze(0)
    else:
        assert s2_r_rgb.size(3)==3
    if sr_min is None or sr_max is None:
        sr = s2_r_rgb.reshape(-1, 3)
        sr_min = sr.min(0)[0]
        sr_max = sr.max(0)[0]
    else:
        assert sr_min.squeeze().size(0)==3
        assert sr_max.squeeze().size(0)==3
    sr_min = sr_min.squeeze().unsqueeze(0).unsqueeze(0).unsqueeze(0)
    sr_max = sr_max.squeeze().unsqueeze(0).unsqueeze(0).unsqueeze(0)
    return (s2_r_rgb - sr_min) / (sr_max - sr_min) 

# def swap_reflectances(s2_r):
#     bands_correct_ordering = torch.tensor([0,1,2,4,5,6,3,7,8,9])
#     return s2_r[:,bands_correct_ordering,:,:]


def gammacorr(s2_r,gamma=2):
    return s2_r.pow(1/gamma)

def plot_patch(s2_r, idx=1):
    # s2_r = swap_reflectances(s2_r)
    s2_r_n = normalize_patch_for_plot(s2_r[:,:3,:,:].permute(0,2,3,1), sr_min=None, sr_max=None)
    s2_r_n_rgb = s2_r_n[:,:,:,torch.tensor([2,1,0])] + 0.0
    fig, ax = plt.subplots(dpi=100)
    plt.imshow(gammacorr(s2_r_n_rgb[idx,:,:,:]))
    ax.set_xticks([])
    ax.set_yticks([])
    return fig, ax

def analyse_angles(loader, max_batch=100):

    a = torch.tensor([])
    s2_r = torch.tensor([])    
    for idx, batch in zip(range(max_batch), loader):
        (s2_x, s2_a, s2_m, s1_x, s1_a_asc, s1_a_desc, srtm_x) = destructure_batch(batch)
        angles = s2_a.transpose(0,1).reshape(6,-1).transpose(0,1)
        a = torch.concat((a, angles), axis=0)
        s2_r = torch.concat((s2_r, s2_x), axis=0)
    plot_patch(s2_r, idx=1)
    fig, ax = plt.subplots(dpi=100)
    im = ((s2_x.squeeze()[:1,:,:].permute((1,2,0))/10000).flip(2).cpu().numpy()*255).astype('int')
    plt.imshow(im, cmap='Blues')
    fig, ax = plt.subplots(2,3,dpi=100)
    ax[0,0].hist(a[:,0].numpy(),bins=50)
    ax[1,0].hist(a[:,1].numpy(),bins=50)
    ax[0,1].hist(a[:,2].numpy(),bins=50)
    ax[1,1].hist(a[:,3].numpy(),bins=50)
    ax[0,2].hist(a[:,4].numpy(),bins=50)
    ax[1,2].hist(a[:,5].numpy(),bins=50)
    fig.savefig("/home/yoel/Documents/angles_hist.png")
    return a

def main(tensors_dir='/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/real_data/torchfiles/', 
        rsr_dir="/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/", 
        batch_par_epoch=100):
    data_dir = os.path.join(os.path.join(os.path.dirname(prosailvae.__file__),
                                         os.pardir),"data/")
    
    train_loader, _, _ = get_mmdc_loaders(tensors_dir=tensors_dir,
                                          batch_par_epoch=batch_par_epoch)
    analyse_angles(train_loader, max_batch=100)                                      
    mean, std = get_mmdc_dataset_mean_std(train_loader)
    torch.save(mean, rsr_dir + "mmdc_"+ "norm_mean.pt") 
    torch.save(std, rsr_dir + "mmdc_" + "norm_std.pt") 

if __name__ == "__main__":
    parser = get_S2_data_preprocess_parser().parse_args()
    main(tensors_dir=parser.tensor_dir, rsr_dir=parser.rsr_dir, 
         batch_par_epoch=parser.batch_par_epoch)
