#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 08:25:49 2022

@author: yoel
"""
import torch.nn as nn
import torch
import logging
from prosailvae.utils import NaN_model_params, select_rec_loss_fn
from mmdc_singledate.datamodules.mmdc_datamodule import destructure_batch
import socket


def unbatchify(tensor):
    n_tensor_dim = len(tensor.size())
    if n_tensor_dim == 3:
        patch_size = torch.sqrt(torch.tensor(tensor.size(0))).int().item()
        n_feat = tensor.size(1)
        n_samples = tensor.size(2)
        return tensor.reshape(patch_size, patch_size, n_feat, n_samples).permute(3,2,0,1)
    else:
        raise NotImplementedError

def crop_s2_input(s2_input, hw_crop=0):
    return s2_input[:, :, hw_crop:-hw_crop, hw_crop:-hw_crop]
 

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
                 beta_kl=0, beta_index=0, logger_name='PROSAIL-VAE logger',
                 inference_mode=False, supervised_model=None,
                 lat_nll="",
                 spatial_mode=False):
        
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
        self.logger = logging.getLogger(logger_name)
        self.beta_index = beta_index
        self.inference_mode = inference_mode
        self.supervised_model = supervised_model
        self.lat_nll = lat_nll
        self.spatial_mode = spatial_mode

    def change_device(self, device):
        self.device=device
        self.encoder.change_device(device)
        self.lat_space.change_device(device)
        self.sim_space.change_device(device)
        self.decoder.change_device(device)
        pass

    def encode(self, x, angles):
        y, angles = self.encoder.encode(x, angles)
        return y, angles
    
    def encode2lat_params(self, x, angles):
        y, angles = self.encode(x, angles)
        dist_params = self.lat_space.get_params_from_encoder(y)
        return dist_params
    
    def sample_latent_from_params(self, dist_params, n_samples=1):
        z = self.lat_space.sample_latent_from_params(dist_params, n_samples=n_samples)
        return z
    
    def transfer_latent(self, z):
        sim = self.sim_space.z2sim(z)
        return sim
    
    def decode(self, sim, angles, apply_norm=None):
        rec = self.decoder.decode(sim, angles, apply_norm=apply_norm)
        return rec
        
    def forward(self, x, angles=None, n_samples=1, apply_norm=None):
        
        # encoding
        if angles is None:
            angles = x[:,-3:]
            x = x[:,:-3]
        
        y, angles = self.encode(x, angles)
        dist_params = self.lat_space.get_params_from_encoder(y)
        if self.inference_mode:
            return dist_params, None, None, None
        # latent sampling
        z = self.sample_latent_from_params(dist_params, n_samples=n_samples)
        
        # transfer to simulator variable
        sim = self.transfer_latent(z)
        
        # decoding
        rec = self.decode(sim, angles, apply_norm=apply_norm)
        if self.spatial_mode:
            return dist_params, z, unbatchify(sim), unbatchify(rec)
        else:
            return dist_params, z, sim, rec
    
    def point_estimate_rec(self, x, angles, mode='random', apply_norm=False):
        if mode == 'random':
            dist_params, z, sim, rec = self.forward(x, angles, n_samples=1, apply_norm=apply_norm)
            
        elif mode == 'lat_mode':
            y, angles = self.encode(x, angles)
            dist_params = self.lat_space.get_params_from_encoder(y)
            # latent mode
            z = self.lat_space.latent_mode(x)
            # transfer to simulator variable
            sim = self.transfer_latent(z)
            # decoding
            rec = self.decode(sim, angles, apply_norm=apply_norm)
            
        elif mode == "sim_mode":
            y, angles = self.encode(x, angles)
            dist_params = self.lat_space.get_params_from_encoder(y)
            lat_pdfs, lat_supports = self.lat_space.latent_pdf(dist_params)
            sim = self.sim_space.sim_mode(lat_pdfs, lat_supports, n_pdf_sample_points=5001)
            z = self.sim_space.sim2z(sim)
            # Quickfix for angle dimension:
            if len(angles.size())==4:
                angles = angles.permute(0,2,3,1)
                angles = angles.reshape(-1, 3)
            rec = self.decode(sim, angles, apply_norm=apply_norm)
            
        elif mode == "sim_median":
            y = self.encode(x, angles)
            dist_params = self.lat_space.get_params_from_encoder(y)
            lat_pdfs, lat_supports = self.lat_space.latent_pdf(dist_params)
            sim = self.sim_space.sim_median(lat_pdfs, lat_supports, n_samples=5001)
            z = self.sim_space.sim2z(sim)
            rec = self.decode(sim, angles, apply_norm=apply_norm)
            
        elif mode == "sim_expectation":
            y, angles = self.encode(x, angles)
            dist_params = self.lat_space.get_params_from_encoder(y)
            lat_pdfs, lat_supports = self.lat_space.latent_pdf(dist_params)
            sim = self.sim_space.sim_expectation(lat_pdfs, lat_supports, n_samples=5001)
            z = self.sim_space.sim2z(sim)
            rec = self.decode(sim, angles, apply_norm=apply_norm)           
            
        else:
            raise NotImplementedError()
            
        if self.spatial_mode and mode != 'random':
            return dist_params, z, unbatchify(sim), unbatchify(rec)
        else:
            return dist_params, z, sim, rec
    
    def compute_unsupervised_loss_over_batch(self, batch, normalized_loss_dict, 
                                             len_loader=1, n_samples=1, mmdc_dataset=True):

        if mmdc_dataset==True:
            (s2_r, s2_a) = batch
            # (s2_r, s2_a, _, _, _, _, _) = destructure_batch(batch)
            s2_r = s2_r.to(self.device)
            s2_a = s2_a.to(self.device)
        else:
            s2_r = batch[0].to(self.device) 
            s2_a = batch[1].to(self.device)  
        params, z, sim, rec = self.forward(s2_r, n_samples=n_samples, angles=s2_a)     
        if self.decoder.loss_type=='spatial_nll':
            s2_r = crop_s2_input(s2_r, self.encoder.nb_enc_cropped_hw)
            rec_loss = self.decoder.loss(s2_r, rec)
        else:
            rec_loss = self.decoder.loss(s2_r, rec)

        loss_dict = {'rec_loss': rec_loss.item()}
        loss_sum = rec_loss
        
        if self.beta_kl > 0:
            if self.supervised_model is None:
                kl_loss = self.beta_kl * self.lat_space.kl(params).sum(1).mean()

            else:
                params2 = self.supervised_model.encode2lat_params(s2_r, s2_a)
                kl_loss = self.beta_kl * self.lat_space.kl(params, params2).sum(1).mean()

            loss_sum += kl_loss
            loss_dict['kl_loss'] = kl_loss.item()

        if self.beta_index > 0:
            index_loss = self.beta_index * self.decoder.rec_loss_fn(s2_r, rec)
            loss_sum += index_loss
            loss_dict['index_loss'] = index_loss.item()

        loss_dict['loss_sum'] = loss_sum.item()

        for key in loss_dict:
            if key not in normalized_loss_dict:
                normalized_loss_dict[key] = 0.0
            normalized_loss_dict[key] += loss_dict[key]/len_loader
        
        return loss_sum, normalized_loss_dict
    
    def compute_supervised_loss_over_batch(self, batch, normalized_loss_dict, len_loader=1):
        s2_r = batch[0].to(self.device) 
        s2_a = batch[1].to(self.device)  
        ref_sim = batch[2].to(self.device)  
        ref_lat = self.sim_space.sim2z(ref_sim)
        y, angles = self.encode(s2_r, s2_a)
        if y.isnan().any() or y.isinf().any():
            nan_in_params = NaN_model_params(self)
            err_str = "NaN encountered during encoding, but there is no NaN in network parameters!"
            if nan_in_params:
                err_str = "NaN encountered during encoding, there are NaN in network parameters!"
            raise ValueError(err_str)
        params = self.lat_space.get_params_from_encoder(y=y)
        reduction_nll = "sum"
        if self.lat_nll == "lai_nll":
            reduction_nll = "lai"
        loss_sum = self.lat_space.loss(ref_lat, params, reduction_nll=reduction_nll)
        if loss_sum.isnan().any() or loss_sum.isinf().any():
            raise ValueError
        all_losses = {'lat_loss': loss_sum.item()}
        all_losses['loss_sum'] = loss_sum.item()
        for key in all_losses:
            if key not in normalized_loss_dict:
                normalized_loss_dict[key] = 0.0
            normalized_loss_dict[key] += all_losses[key]/len_loader
        
        return loss_sum, normalized_loss_dict

    def compute_lat_nlls_batch(self, batch):
        s2_r = batch[0].to(self.device) 
        s2_a = batch[1].to(self.device)  
        ref_sim = batch[2].to(self.device)  
        ref_lat = self.sim_space.sim2z(ref_sim)
        y, angles = self.encode(s2_r, s2_a)
        params = self.lat_space.get_params_from_encoder(y=y)
        nll = self.lat_space.loss(ref_lat, params, reduction=None, reduction_nll=None)
        if nll.isnan().any() or nll.isinf().any():
            raise ValueError
        return nll

    def compute_lat_nlls(self, dataloader, batch_per_epoch=None):
        self.eval()
        all_nlls = []
        with torch.no_grad():
            if batch_per_epoch is None:
                batch_per_epoch = len(dataloader)
            for _, batch in zip(range(min(len(dataloader),batch_per_epoch)),dataloader):
                nll_batch = self.compute_lat_nlls_batch(batch)
                all_nlls.append(nll_batch)
                if torch.isnan(nll_batch).any():
                    self.logger.error("NaN Loss encountered during validation !")
        all_nlls = torch.vstack(all_nlls)
        return all_nlls
    def save_ae(self, epoch, optimizer, loss, path):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, path)  
    
    def load_ae(self, path, optimizer=None, weights_only=False):
        # map_location = 'cuda:0' if self.device != torch.device('cpu') else 'cpu'
        checkpoint = torch.load(path, map_location=self.device, weights_only=weights_only)
        self.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        return epoch, loss
    
    def fit(self, dataloader, optimizer, n_samples=1, batch_per_epoch=None, mmdc_dataset=False):
        self.train()
        train_loss_dict = {}
        len_loader = len(dataloader.dataset)
        if batch_per_epoch is None:
            batch_per_epoch = len(dataloader)
        for i, batch in zip(range(min(len(dataloader), batch_per_epoch)), dataloader):
            optimizer.zero_grad()
            
            if not self.supervised:
                loss_sum, _ = self.compute_unsupervised_loss_over_batch(batch, 
                    train_loss_dict, n_samples=n_samples, len_loader=len_loader, mmdc_dataset=mmdc_dataset)
            else:
                loss_sum, _ = self.compute_supervised_loss_over_batch(batch, train_loss_dict, 
                                                        len_loader=len_loader)
                
            if torch.isnan(loss_sum).any():
                self.logger.error(f"NaN Loss encountered during training at batch {i}!")
                
            loss_sum.backward()
            optimizer.step()
            if NaN_model_params(self):
                self.logger.debug(f"NaN model parameters after batch {i}!")
        self.eval()
        return train_loss_dict

    def validate(self, dataloader, n_samples=1, batch_per_epoch=None, mmdc_dataset=False):
        self.eval()
        valid_loss_dict = {}
        len_loader = len(dataloader.dataset)
        with torch.no_grad():
            if batch_per_epoch is None:
                batch_per_epoch = len(dataloader)
            for _, batch in zip(range(min(len(dataloader),batch_per_epoch)),dataloader):
                if not self.supervised:
                    loss_sum, _ = self.compute_unsupervised_loss_over_batch(batch, 
                        valid_loss_dict, n_samples=n_samples, len_loader=len_loader, mmdc_dataset=mmdc_dataset)
                else:
                    loss_sum, _ = self.compute_supervised_loss_over_batch(batch, valid_loss_dict, 
                                                        len_loader=len_loader)
            if torch.isnan(loss_sum).any():
                self.logger.error("NaN Loss encountered during validation !")
        return valid_loss_dict



from prosailvae.dist_utils import kl_tn_uniform, truncated_gaussian_nll

class lr_finder_elbo(nn.Module):
    def __init__(self, index_loss, beta_kl=1, beta_index=0, loss_type='diag_nll', ssimulator=None) -> None:
        super(lr_finder_elbo,self).__init__()
        self.beta_kl = beta_kl
        self.beta_index = beta_index
        self.index_loss = index_loss
        self.loss_type = loss_type
        self.ssimulator = ssimulator 
        self.rec_loss_fn = select_rec_loss_fn(self.loss_type)
        pass

    def lr_finder_elbo_inner(self, model_outputs, label):
        dist_params, _, _, rec = model_outputs

        if self.ssimulator.apply_norm:
            label = self.ssimulator.normalize(label)
        
        if len(label.size()) == 2:
            label = label.unsqueeze(2)
        if self.loss_type=="spatial_nll":
            hw = (label.size(3) - rec.size(3)) // 2
            if hw > 0:
                label = label[:,:,hw:-hw,hw:-hw]
        rec_loss = self.rec_loss_fn(label, rec)
        loss_sum = rec_loss.mean()
        sigma = dist_params[:, :, 1].squeeze()
        mu = dist_params[:, :, 0].squeeze()
        if self.beta_kl > 0:
            kl_loss = self.beta_kl * kl_tn_uniform(mu, sigma).sum(1).mean()
            loss_sum += kl_loss
        if self.beta_index > 0:
            index_loss = self.beta_index * self.index_loss(label, rec, lossfn=self.rec_loss_fn)
            loss_sum += index_loss
        return loss_sum

    def forward(self, model_outputs, label):
        return self.lr_finder_elbo_inner(model_outputs, label)

def lr_finder_sup_nll(model_outputs, label):
    dist_params, _, _, _ = model_outputs
    sigma = dist_params[:, :, 1].squeeze()
    mu = dist_params[:, :, 0].squeeze()
    loss = truncated_gaussian_nll(label, mu, sigma).mean() 
    return loss