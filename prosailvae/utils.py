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

def get_total_RAM():
    return "{:.1f} GB".format(psutil.virtual_memory()[0]/1000000000)
def get_RAM_usage():
    return  "{:.3f} GB".format(psutil.virtual_memory()[3]/1000000000)

def get_CPU_usage():
    load1, _, _ = psutil.getloadavg()
    cpu_usage = (load1/os.cpu_count()) * 100
    return  "{:.2f} %".format(cpu_usage)

def save_dict(data_dict, dict_file_path):
    with open(dict_file_path, 'w') as fp:
        json.dump(data_dict, fp, indent=4)

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
    rec_err_var = torch.var(recs, 2)
    return gaussian_nll(tgt, recs.mean(2), rec_err_var, device=tgt.device).mean() 

def full_gaussian_nll_loss(tgt, recs):
    err = recs # - tgt.unsqueeze(2)
    errm = err - err.mean(2).unsqueeze(2)
    sigma_mat = errm @ errm.transpose(1,2) / errm.size(2)
    return full_gaussian_nll(tgt, recs.mean(2), sigma_mat, device=tgt.device).mean() 

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