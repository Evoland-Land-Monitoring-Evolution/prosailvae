#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 08:25:49 2022

@author: yoel
"""
import torch.nn as nn
import torch
import time

class SimVAE(nn.Module):
    """
    A class used to represent an encoder with simulator-decoder.

    ...

    Attributes
    ----------
    encoder : Encoder
        A torch NN that encodes a time series into a low dimension vector to be interpreted as distribution parameters.
    lat_space : LatentSpace
        A torch object representing the latent distributions produced by the encoder.
    sim_space : SimSpace
        A torch object representing the distribution of the decoder parameters, to be derived from the latent distribution.
    decoder : Decoder
        A torch object that decodes samples of parameters distributions from sim_space.
    supervised : bool
        Indicate whether the Encoder is to be trained from a labelled dataset or not.
    dt_nll_coef : float 
        coefficient of the NLL of the derivative of the reconstruction in the total loss.
    dt_order : int
        order of the numerical approximation of time series derivative to be computed in the loss.
    cumsum_coef : float 
        coefficient of the NLL of the cumulative sum of time series to be computed in the loss.
    Methods
    -------
    encode(x)
        Encode time series in x using attribute encoder.
    encode2lat_params(x): 
        Encode time series using attribute encoder and converts it into latent distribution parameters.
    sample_latent_from_params(dist_params, n_samples=1)
        Outputs n_samples samples from latent distributions parametrized by dist_params.
    transfer_latent(z)
        Transforms latent distribution samples into samples of the distribution of parameters of the decoder.
    decode(sim)
        Decode parameters using decoder and reconstruct time series.
    forward(x, n_samples=1)
        Output n_samples samples of distribution of reconstructions from encoding time series x. 
    point_estimate_rec(x, mode='random')
        Outputs the latent distribution parameters, a sample from the latent distribution, a sample from the decoder parameters distribution and a reconstruction. 
        Samples can be random, the mode, the expectation, the median from distributions. This is selected by mode.
    """
    def __init__(self, encoder, decoder, lat_space, sim_space, 
                 supervised=False,  device='cpu', 
                 beta_kl=0):
        
        super(SimVAE, self).__init__()
        # encoder
        self.encoder = encoder
        self.lat_space = lat_space
        self.sim_space = sim_space
        self.decoder = decoder
        self.encoder.eval()
        self.lat_space.eval()
        self.supervised = supervised
        self.device=device
        self.beta_kl = beta_kl
        self.eval()
        
    def encode(self, x, angles):
        y = self.encoder.encode(x, angles)
        return y
    
    def encode2lat_params(self, x, angles):
        y = self.encode(x, angles)
        dist_params = self.lat_space.get_params_from_encoder(y)
        return dist_params
    
    def sample_latent_from_params(self, dist_params, n_samples=1):
        z = self.lat_space.sample_latent_from_params(dist_params, n_samples=n_samples)
        return z
    
    def transfer_latent(self, z):
        sim = self.sim_space.z2sim(z)
        return sim
    
    def decode(self, sim, dec_args):
        rec = self.decoder.decode(sim, dec_args)
        return rec
        
    def forward(self, x, angles=None, n_samples=1):
        # encoding
        y = self.encode(x, angles)
        dist_params = self.lat_space.get_params_from_encoder(y)
        
        # latent sampling
        z = self.sample_latent_from_params(dist_params, n_samples=n_samples)
        
        # transfer to simulator variable
        sim = self.transfer_latent(z)
        
        # decoding
        rec = self.decode(sim, angles)
        
        return dist_params, z, sim, rec
    
    def point_estimate_rec(self, x, angles, mode='random'):
        if mode == 'random':
            dist_params, z, sim, rec = self.forward(x, angles, n_samples=1)
            
        elif mode == 'lat_mode':
            y = self.encode(x, angles)
            dist_params = self.lat_space.get_params_from_encoder(y)
            # latent mode
            z = self.lat_space.latent_mode(x)
            # transfer to simulator variable
            sim = self.transfer_latent(z)
            # decoding
            rec = self.decode(sim, angles)
            
        elif mode == "sim_mode":
            y = self.encode(x, angles)
            dist_params = self.lat_space.get_params_from_encoder(y)
            lat_pdfs, lat_supports = self.lat_space.latent_pdf(dist_params)
            sim = self.sim_space.sim_mode(lat_pdfs, lat_supports, n_pdf_sample_points=5001)
            z = self.sim_space.sim2z(sim)
            rec = self.decode(sim, angles)
            
        elif mode == "sim_median":
            y = self.encode(x, angles)
            dist_params = self.lat_space.get_params_from_encoder(y)
            lat_pdfs, lat_supports = self.lat_space.latent_pdf(dist_params)
            sim = self.sim_space.sim_median(lat_pdfs, lat_supports, n_samples=5001)
            z = self.sim_space.sim2z(sim)
            rec = self.decode(sim, angles)
            
        elif mode == "sim_expectation":
            y = self.encode(x, angles)
            dist_params = self.lat_space.get_params_from_encoder(y)
            lat_pdfs, lat_supports = self.lat_space.latent_pdf(dist_params)
            sim = self.sim_space.sim_expectation(lat_pdfs, lat_supports, n_samples=5001)
            z = self.sim_space.sim2z(sim)
            rec = self.decode(sim, angles)           
            
        else:
            raise NotImplementedError()
            
        return dist_params, z, sim, rec
    
    def compute_unsupervised_loss_over_batch(self, data, angles, normalized_loss_dict, 
                                             len_loader=1, n_samples=1, eps=1e-9):
        # assert n_samples>1
        batch_size = data.size(0)
        data = data.view(batch_size, -1).float()
        params, _, _, rec = self.forward(data, n_samples=n_samples, angles=angles)       
        rec_loss = self.decoder.loss(data, rec)

        loss_dict = {'rec_loss': rec_loss.item()}
        loss_sum=rec_loss
        
        if self.beta_kl > 0:
            kl_loss = self.beta_kl * self.lat_space.kl(params).sum(1).mean()
            loss_sum+=kl_loss
            loss_dict['kl_loss'] = kl_loss.item()
            
        loss_dict['loss_sum'] = loss_sum.item()

        for key in loss_dict:
            if key not in normalized_loss_dict:
                normalized_loss_dict[key] = 0.0
            normalized_loss_dict[key] += loss_dict[key]/len_loader
        
        return loss_sum, normalized_loss_dict
    
    def compute_supervised_loss_over_batch(self, data, angles, tgt, normalized_loss_dict, 
                                           len_loader=1, n_samples=1):
        
        batch_size = data.size(0)
        data = data.view(batch_size, -1).float()
        params = self.encode2lat_params(data, angles)
        loss_sum = self.lat_space.loss(tgt, params)
        all_losses = {'lat_loss': loss_sum.item()}
       
        all_losses['loss_sum'] = loss_sum.item()
        for key in all_losses:
            if key not in normalized_loss_dict:
                normalized_loss_dict[key] = 0.0
            normalized_loss_dict[key] += all_losses[key]/len_loader
        
        return loss_sum, normalized_loss_dict
    
    def save_ae(self, epoch, optimizer, loss, path):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, path)  
    
    def load_ae(self, path, optimizer=None):
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        return epoch, loss
    
    def fit(self, dataloader, optimizer, supervised=False, n_samples=1):
        self.train()
        train_loss_dict = {}
        len_loader = len(dataloader.dataset)
        for i, batch in enumerate(dataloader):
            optimizer.zero_grad()
            s2_refl = batch[0].to(self.device) 
            angles = batch[1].to(self.device) 
            s2_refl.requires_grad = True
            angles.requires_grad = True
            
            if not supervised:
                loss_sum, _ = self.compute_unsupervised_loss_over_batch(s2_refl, 
                                                                        angles, 
                    train_loss_dict, n_samples=n_samples, len_loader=len_loader)
            else:
                tgt = batch[2]
                tgt.requires_grad = True
                loss_sum, _ = self.compute_supervised_loss_over_batch(s2_refl, tgt, 
                    train_loss_dict, n_samples=n_samples, len_loader=len_loader)
            loss_sum.backward()
            optimizer.step()
        self.eval()
        return train_loss_dict

    def validate(self, dataloader, supervised=False, n_samples=1):
        self.eval()
        valid_loss_dict = {}
        len_loader = len(dataloader.dataset)
        with torch.no_grad():
            for _, batch in enumerate(dataloader):
                s2_refl = batch[0] 
                angles = batch[1]
                if not supervised:
                    loss_sum, _ = self.compute_unsupervised_loss_over_batch(s2_refl, angles, 
                        valid_loss_dict, n_samples=n_samples, len_loader=len_loader)
                else:
                    tgt = batch[2]
                    loss_sum, _ = self.compute_supervised_loss_over_batch(s2_refl, angles, tgt, 
                        valid_loss_dict, n_samples=n_samples, len_loader=len_loader)
    
        return valid_loss_dict

