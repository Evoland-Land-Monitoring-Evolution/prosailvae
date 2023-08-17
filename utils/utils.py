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

from dataclasses import dataclass

@dataclass
class StandardizeCoeff:
    loc:torch.Tensor|None=None
    scale:torch.Tensor|None=None

@dataclass
class IOStandardizeCoeffs:
    bands:StandardizeCoeff
    angles:StandardizeCoeff
    idx:StandardizeCoeff

def load_standardize_coeffs(data_dir:str|None=None, prefix="", n_bands=10, n_angles=3, n_idx=4):
    coeffs_info_dict = {"bands_loc":["norm_mean", torch.zeros((n_bands))],
                        "bands_scale":["norm_std", torch.ones((n_bands))],
                        "idx_loc":["idx_loc", torch.zeros((n_idx))],
                        "idx_scale":["idx_scale", torch.ones((n_idx))],
                        "angles_loc":["angles_loc", torch.zeros((n_angles))],
                        "angles_scale": ["angles_scale", torch.ones((n_angles))]}
    coeffs_dict = {}
    for coef_name, info in coeffs_info_dict.items():
        if data_dir is not None and os.path.isfile(os.path.join(data_dir, prefix + f"{info[0]}.pt")):
            coeffs_dict[coef_name] = torch.load(os.path.join(data_dir, prefix + f"{info[0]}.pt")) 
        else:
            coeffs_dict[coef_name] = info[1]
    io_coeffs = IOStandardizeCoeffs(
        bands=StandardizeCoeff(loc=coeffs_dict['bands_loc'], scale=coeffs_dict['bands_scale']),
        idx=StandardizeCoeff(loc=coeffs_dict['idx_loc'], scale=coeffs_dict['idx_scale']),
        angles=StandardizeCoeff(loc=coeffs_dict['angles_loc'], scale=coeffs_dict['angles_scale']))
    return io_coeffs

# @dataclass
# class StandardizeCoeffs:
#     loc:torch.Tensor
#     scale:torch.Tensor

def standardize(x, loc, scale, dim=0):
    nb_dim = len(x.size())
    standardized_x = (x - torch_select_unsqueeze(loc, select_dim=dim, nb_dim=nb_dim)
                      ) / torch_select_unsqueeze(scale, select_dim=dim, nb_dim=nb_dim)
    return standardized_x

def unstandardize(x, loc, scale, dim=0):
    nb_dim = len(x.size())
    unstandardized_x = (x * torch_select_unsqueeze(scale, select_dim=dim, nb_dim=nb_dim)
                      ) + torch_select_unsqueeze(loc, select_dim=dim, nb_dim=nb_dim)
    return unstandardized_x

def select_rec_loss_fn(loss_type):
    simple_losses_1d = ["diag_nll", "hybrid_nll", "lai_nll"]
    if loss_type in simple_losses_1d:
        rec_loss_fn = NLLLoss(2,1)
    elif loss_type == "spatial_nll":
        rec_loss_fn = NLLLoss(1,2) 
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

def gaussian_nll(x, mu, sigma, eps=1e-6, device='cpu', sum_dim=1):
    eps = torch.tensor(eps).to(device)
    return ((torch.square(x - mu) / torch.max(sigma, eps)) +
            torch.log(torch.max(sigma, eps))).sum(sum_dim)

def gaussian_nll_loss(tgt, recs, sample_dim=2, feature_dim=1):
    if len(recs.size()) < 3:
        raise ValueError("recs needs a batch, a feature and a sample dimension")
    elif recs.size(sample_dim)==1:
        rec_err_var=torch.tensor(0.0001).to(tgt.device) # constant variance, enabling computation even with 1 sample
        rec_mu = recs
    else:
        rec_err_var = recs.var(sample_dim, keepdim=True)#.unsqueeze(sample_dim)
        rec_mu = recs.mean(sample_dim, keepdim=True)#.unsqueeze(sample_dim)
        # if feature_dim > sample_dim: # if feature dimension is after sample dimension, 
        #     # reducing it because sample dimension disappeared
        #     feature_dim = feature_dim - 1
    return gaussian_nll(tgt.unsqueeze(sample_dim), rec_mu, rec_err_var, sum_dim=feature_dim).mean()

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

class NLLLoss(nn.Module):
    def __init__(self, sample_dim=2, feature_dim=1) -> None:
        super().__init__()
        self.sample_dim = sample_dim
        self.feature_dim = feature_dim 
    def forward(self, targets, inputs):
        return gaussian_nll_loss(targets, inputs, sample_dim=self.sample_dim, feature_dim=self.feature_dim)
    
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


def torch_select_unsqueeze(tensor, select_dim, nb_dim):
    # assumes tensor is 1-dimensional
    
    if nb_dim < 0:
        raise ValueError 
    elif nb_dim == 1:
        return tensor
    else:
        view_dims = [1 for _ in range(nb_dim)]
        view_dims[select_dim] = -1
        return tensor.squeeze().view(view_dims)
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)