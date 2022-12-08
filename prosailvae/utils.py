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

def gaussian_nll(x, mu, sigma, eps=1e-6, device='cpu'):
    eps = torch.tensor(eps).to(device)
    return (torch.square(x - mu) / torch.max(sigma, eps)).sum(1) +  \
            torch.log(torch.max(sigma, eps)).sum(1)

def gaussian_nll_loss(tgt, recs):
    rec_err_var = torch.var(recs-tgt.unsqueeze(2), 2)
    return gaussian_nll(tgt, recs.mean(2), rec_err_var, device=tgt.device).mean() 
