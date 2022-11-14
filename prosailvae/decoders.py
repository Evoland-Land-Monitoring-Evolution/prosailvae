#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 14:23:46 2022

@author: yoel
"""
import torch.nn as nn
import torch


def gaussian_nll(x, mu, sigma, eps=1e-6, device='cpu'):
    eps = torch.tensor(eps).to(device)
    return (torch.square(x - mu) / torch.max(sigma, eps)).mean(1).sum(1) +  \
            torch.log(torch.max(sigma.squeeze(1), eps)).sum(1)

class Decoder(nn.Module):

    def decode(self):
        raise NotImplementedError()
    
    def loss(self):
        raise NotImplementedError()

class ProsailSimulatorDecoder(Decoder):
    
    def __init__(self, prosailsimulator, ssimulator, device='cpu'):
        super().__init__()
        self.device = device
        self.prosailsimulator = prosailsimulator
        self.ssimulator = ssimulator
        
    def decode(self, z, angles):
        n_samples = z.size(2)
        batch_size = z.size(0)
        sim_input = torch.concat((z, 
            angles.unsqueeze(2).repeat(1,1,n_samples)), 
             axis=1).transpose(1,2).reshape(n_samples*batch_size, -1)
        rec = self.ssimulator(self.prosailsimulator(sim_input)).reshape(batch_size, 
                                                                        n_samples, 
                                                                        -1).transpose(1,2)
        
        return rec
    
    def loss(self, tgt, rec):        
        rec_err_var = torch.var(rec-tgt.unsqueeze(1), 1).unsqueeze(1)
        rec_loss = gaussian_nll(tgt.unsqueeze(1), rec, rec_err_var, device=self.device).mean() 
        return rec_loss

