#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 11:18:38 2022

@author: yoel
"""
import json
import psutil
import os
import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from  matplotlib.lines import Line2D

def select_rec_loss_fn(loss_type):
    if loss_type == "diag_nll" or loss_type == "hybrid_nll":
            rec_loss_fn = gaussian_nll_loss
    elif loss_type == "full_nll":
        rec_loss_fn = full_gaussian_nll_loss
    elif loss_type =='mse':
        rec_loss_fn = mse_loss
    else:
        raise NotImplementedError("Please choose between 'diag_nll' (diagonal covariance matrix) and 'full_nll' (full covariance matrix) for nll loss option.")
    return rec_loss_fn

def get_total_RAM():
    return "{:.1f} GB".format(psutil.virtual_memory()[0]/1000000000)
def get_RAM_usage():
    return  "{:.3f} GB".format(psutil.virtual_memory()[3]/1000000000)

def get_CPU_usage():
    load1, _, _ = psutil.getloadavg()
    cpu_usage = (load1/os.cpu_count()) * 100
    return  "{:.2f} %".format(cpu_usage)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def save_dict(data_dict, dict_file_path):
    with open(dict_file_path, 'w') as fp:
        json.dump(data_dict, fp, indent=4, cls=NpEncoder)

def load_dict(dict_file_path):
    with open(dict_file_path, "r") as read_file:
        data_dict = json.load(read_file)
    return data_dict

def NaN_model_params(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            if torch.isnan(param).any() :
                return True
    return False

def full_gaussian_nll(x, mu, sigma_mat, eps=1e-6, device='cpu', regularization=1e-3):
    eps = torch.tensor(eps).to(device)
    L, L_info = torch.linalg.cholesky_ex(sigma_mat + regularization * torch.eye(sigma_mat.size(1)).unsqueeze(0).to(sigma_mat.device), 
                                 check_errors=False)
    if L_info.ne(0).any():
        raise ValueError('Baddly conditionned covariance matrix for cholesky lower triangular computation. ')
    inverse_sigma_mat = torch.cholesky_inverse(L)
    return ((x - mu).unsqueeze(1) @ inverse_sigma_mat @ (x - mu).unsqueeze(2)).squeeze() +  \
            2 * torch.log(torch.max(torch.diagonal(L,0,1,2), eps)).sum(1)

def gaussian_nll(x, mu, sigma, eps=1e-6, device='cpu'):
    eps = torch.tensor(eps).to(device)
    return (torch.square(x - mu) / torch.max(sigma, eps)).sum(1) +  \
            torch.log(torch.max(sigma, eps)).sum(1)

def gaussian_nll_loss(tgt, recs):
    # rec_err_var = torch.var(recs-tgt.unsqueeze(2), 2)
    if len(recs.size()) < 3:
        raise ValueError("NLL needs more than 1 samples of distribution.")
    elif recs.size(2)==1:
        raise ValueError("NLL needs more than 1 samples of distribution.")
    rec_err_var = torch.var(recs, 2).unsqueeze(2)
    return gaussian_nll(tgt, recs.mean(2).unsqueeze(2), rec_err_var, device=tgt.device).mean() 

def full_gaussian_nll_loss(tgt, recs):
    err = recs # - tgt.unsqueeze(2)
    errm = err - err.mean(2).unsqueeze(2)
    sigma_mat = errm @ errm.transpose(1,2) / errm.size(2)
    return full_gaussian_nll(tgt, recs.mean(2), sigma_mat, device=tgt.device).mean() 

def mse_loss(tgt, recs):
    if len(recs.size()) > 2:
        recs = recs[:,:,0]
    rec = recs.reshape(tgt.size(0), tgt.size(1))
    rec_lossfn = nn.MSELoss()
    return rec_lossfn(rec, tgt)

def cuda_cholesky(A):
    n = A.size(0)
    assert A.size(1) == n
    L = torch.zeros_like(A)
    L[0,0,:] = torch.sqrt(A[0,0,:])
    L[1:,0,:] = A[1:,0,:] / L[0,0,:]
 
    for i in range(1,n-1):
        L[i,i,:] = torch.sqrt(A[i,i,:] - L[i,:,:].pow(2).sum(0))
        L[i+1:,i,:] = (A[i+1:,i,:].squeeze() - (L[i+1:,:i,:] * L[i,:i,:]).sum(1).squeeze()).reshape(n-i-1,-1) 
        L[i+1:,i,:] = L[i+1:,i,:] / L[i,i,:]
    L[n-1,n-1,:] = torch.sqrt(A[n-1,n-1,:] - L[n-1,:,:].pow(2).sum(0))
    return L

# A = torch.rand(5,5,1).abs()
# A = A+A.transpose(0,1)+torch.eye(5).reshape(5,5,1)
# import time
# t0=time.time()
# L = cuda_cholesky(A)
# t1=time.time()
# L2 = torch.linalg.cholesky_ex(A.squeeze())
# t2=time.time()
# print(str(t1-t0))
# print(str(t2-t1))

def plot_grad_flow(model, savefile=None):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    named_parameters = model.named_parameters()
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            if p.grad is not None:
                ave_grads.append(p.grad.abs().mean().detach().cpu().numpy())
                max_grads.append(p.grad.abs().max().detach().cpu().numpy())
            else:
                print("None gradient")
                print(n)
    fig, ax = plt.subplots(dpi=150, tight_layout=True)
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    # plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xticks(range(0,len(ave_grads), 1))
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = 0) # zoom in on the lower gradient regions
    plt.yscale('symlog', linthresh=1e-8)
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    if savefile is not None:
        fig.savefig(savefile)