#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 14:23:46 2022

@author: yoel
"""
import torch.nn as nn
import torch
from prosailvae.utils import select_rec_loss_fn, gaussian_nll_loss, full_gaussian_nll_loss, mse_loss




class Decoder(nn.Module):

    def decode(self):
        raise NotImplementedError()
    
    def loss(self):
        raise NotImplementedError()

class ProsailSimulatorDecoder(Decoder):
    
    def __init__(self, prosailsimulator, ssimulator, device='cpu', 
                 loss_type='diag_nll', patch_mode=False, patch_size=32):
        super().__init__()
        self.device = device
        self.prosailsimulator = prosailsimulator
        self.ssimulator = ssimulator
        self.loss_type = loss_type
        self.patch_mode = patch_mode
        self.patch_size = patch_size
        self.nbands = len(ssimulator.bands)
    
    def change_device(self, device):
        self.device=device
        self.ssimulator.change_device(device)
        self.prosailsimulator.change_device(device)
        pass

    def decode(self, z, angles):
        n_samples = z.size(2)
        batch_size = z.size(0)
        
        sim_input = torch.concat((z, 
            angles.unsqueeze(2).repeat(1,1,n_samples)), 
             axis=1).transpose(1,2).reshape(n_samples*batch_size, -1)
        rec = self.ssimulator(self.prosailsimulator(sim_input)).reshape(batch_size, 
                                                                        n_samples, 
                                                                        -1).transpose(1,2)
        if self.patch_mode:
            rec = rec.reshape(-1, self.patch_size, self.patch_size, self.nbands, n_samples)
            rec = rec.permute(0,3,1,2,4)
        return rec
    
    # def loss(self, tgt, rec):        
    #     rec_err_var = torch.var(rec-tgt.unsqueeze(1), 1).unsqueeze(1)
    #     rec_loss = gaussian_nll(tgt.unsqueeze(1), rec, rec_err_var, device=self.device).mean() 
    #     return rec_loss
    
    def loss(self, tgt, rec):        
        if self.ssimulator.apply_norm:
            tgt = self.ssimulator.normalize(tgt)
        rec_loss_fn = select_rec_loss_fn(self.loss_type)
        rec_loss = rec_loss_fn(tgt, rec)
        return rec_loss

