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
from dataset.loaders import get_flattened_patch
from mmdc_singledate.datamodules.mmdc_datamodule import destructure_batch
import socket
    

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
                 flat_patch_mode=False, inference_mode=False, supervised_model=None,
                 lat_nll=""):
        
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
        self.flat_patch_mode = flat_patch_mode
        self.beta_index = beta_index
        self.inference_mode = inference_mode
        self.supervised_model = supervised_model
        multi_output_encoder=False
        try:
            a = self.encoder.dnet
            multi_output_encoder=True
        except:
            pass
        self.multi_output_encoder = multi_output_encoder
        self.lat_nll = lat_nll

    def change_device(self, device):
        self.device=device
        self.encoder.change_device(device)
        self.lat_space.change_device(device)
        self.sim_space.change_device(device)
        self.decoder.change_device(device)
        pass

    def encode(self, x, angles):
        if self.multi_output_encoder:
            _,y = self.encoder.encode(x, angles)
        else:
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
        if angles is None:
            angles = x[:,-3:]
            x = x[:,:-3]
        
        y = self.encode(x, angles)
        if self.multi_output_encoder:
            y=y[-1,:,:]
        dist_params = self.lat_space.get_params_from_encoder(y)
        if self.inference_mode:
            return dist_params, None, None, None
        # latent sampling
        z = self.sample_latent_from_params(dist_params, n_samples=n_samples)
        
        # transfer to simulator variable
        sim = self.transfer_latent(z)
        
        # decoding
        # Quickfix for angle dimension:
        if len(angles.size())==4:
            angles = angles.permute(0,2,3,1)
            angles = angles.reshape(-1, 3)
        rec = self.decode(sim, angles)
        return dist_params, z, sim, rec
    
    def point_estimate_rec(self, x, angles, mode='random'):
        if mode == 'random':
            dist_params, z, sim, rec = self.forward(x, angles, n_samples=1)
            
        elif mode == 'lat_mode':
            y = self.encode(x, angles)
            dist_params = self.lat_space.get_params_from_encoder(y)
            if self.multi_output_encoder:
                dist_params = dist_params[-1,:,:,:].unsqueeze(3)
            # latent mode
            z = self.lat_space.latent_mode(x)
            # transfer to simulator variable
            sim = self.transfer_latent(z)
            # decoding
            rec = self.decode(sim, angles)
            
        elif mode == "sim_mode":
            y = self.encode(x, angles)
            dist_params = self.lat_space.get_params_from_encoder(y)
            if self.multi_output_encoder:
                dist_params = dist_params[-1,:,:,:].unsqueeze(3)
            lat_pdfs, lat_supports = self.lat_space.latent_pdf(dist_params)
            sim = self.sim_space.sim_mode(lat_pdfs, lat_supports, n_pdf_sample_points=5001)
            z = self.sim_space.sim2z(sim)
            # Quickfix for angle dimension:
            if len(angles.size())==4:
                angles = angles.permute(0,2,3,1)
                angles = angles.reshape(-1, 3)
            rec = self.decode(sim, angles)
            
        elif mode == "sim_median":
            y = self.encode(x, angles)
            dist_params = self.lat_space.get_params_from_encoder(y)
            if self.multi_output_encoder:
                dist_params = dist_params[-1,:,:,:].unsqueeze(3)
            lat_pdfs, lat_supports = self.lat_space.latent_pdf(dist_params)
            sim = self.sim_space.sim_median(lat_pdfs, lat_supports, n_samples=5001)
            z = self.sim_space.sim2z(sim)
            rec = self.decode(sim, angles)
            
        elif mode == "sim_expectation":
            y = self.encode(x, angles)
            dist_params = self.lat_space.get_params_from_encoder(y)
            if self.multi_output_encoder:
                dist_params = dist_params[-1,:,:,:].unsqueeze(3)
            lat_pdfs, lat_supports = self.lat_space.latent_pdf(dist_params)
            sim = self.sim_space.sim_expectation(lat_pdfs, lat_supports, n_samples=5001)
            z = self.sim_space.sim2z(sim)
            rec = self.decode(sim, angles)           
            
        else:
            raise NotImplementedError()
            
        return dist_params, z, sim, rec
    
    def compute_unsupervised_loss_over_batch(self, batch, normalized_loss_dict, 
                                             len_loader=1, n_samples=1, mmdc_dataset=True):
        # assert n_samples>1
        if mmdc_dataset==True:
            if self.flat_patch_mode:      
                s2_r, s2_a = get_flattened_patch(batch, device=self.device)
                # s2_r = s2_r / 10000
            else:
                (s2_r, s2_a, _, _, _, _, _) = destructure_batch(batch)
                s2_r = s2_r.to(self.device)
                s2_a = s2_a.to(self.device)
                if socket.gethostname()=='CELL200973': #DEV mode with smaller patch
                    s2_r = s2_r[:,:,:16,:16]
                    s2_a = s2_a[:,:,:16,:16]
        else:
            s2_r = batch[0].to(self.device) 
            s2_a = batch[1].to(self.device)  
        params, z, sim, rec = self.forward(s2_r, n_samples=n_samples, angles=s2_a)     
        if params.isnan().any() or params.isinf().any():
            nan_in_params = NaN_model_params(self)
            err_str = "NaN encountered during encoding, but there is no NaN in network parameters!"
            if nan_in_params:
                err_str = "NaN encountered during encoding, there are NaN in network parameters!"
            raise ValueError(err_str)
        nan_rec = torch.isnan(rec[:,0,:]).detach()
        if nan_rec.any():
            n_samples = z.size(2)
            batch_size = z.size(0)
            # sim_input = torch.concat((z, 
            #     angles.unsqueeze(2).repeat(1,1,n_samples)), 
            #      axis=1).transpose(1,2).reshape(n_samples*batch_size, -1)
            self.logger.error("NaN in reconstruction parameters !")
            nan_batch_idx = torch.where(nan_rec)[0]
            nan_sample_idx = torch.where(nan_rec)[1]
            self.logger.debug(f"{len(nan_batch_idx)} reconstructions have NaNs.")
            self.logger.debug(f"{len(nan_batch_idx)} reconstructions have NaNs.")
            self.logger.debug("The First NaN reconstruction has:")
            self.logger.debug("z = ")
            self.logger.debug(f"{z[nan_batch_idx[0], :, nan_sample_idx[0]].squeeze()}")
            self.logger.debug("sim = ")
            self.logger.debug(f"{sim[nan_batch_idx[0], :, nan_sample_idx[0]].squeeze()}")
            self.logger.debug("angles = ")
            self.logger.debug(f"{s2_a[nan_batch_idx[0], :].squeeze()}")
            self.logger.debug("mu = ")
            self.logger.debug(f"{params[nan_batch_idx[0],:,0].squeeze()}")
            self.logger.debug("sigma = ")
            self.logger.debug(f"{params[nan_batch_idx[0],:,1].squeeze()}")
        
        # TODO : remove Quickfix to reshape s2_r like rec:
        if len(s2_r.size())==4:
            s2_r = s2_r.permute(0,2,3,1).reshape(rec.size(0), rec.size(1),1)
        rec_loss = self.decoder.loss(s2_r, rec)

        loss_dict = {'rec_loss': rec_loss.item()}
        loss_sum = rec_loss
        
        if self.beta_kl > 0:
            if self.supervised_model is None:
                kl_loss = self.beta_kl * self.lat_space.kl(params).sum(1).mean()

            else:
                # self.logger.info(f's2_r device : {s2_r.device}')
                # self.logger.info(f'self.supervised_model.encoder : {self.supervised_model.encoder.device}')
                # self.logger.info(f'self.supervised_model.encoder.net : {self.supervised_model.encoder.net.device}')
                # self.logger.info(f'self.supervised_model.encoder.norm_mean : {self.supervised_model.encoder.norm_mean.device}')
                # self.logger.info(f'self.supervised_model.encoder.norm_std : {self.supervised_model.encoder.norm_std.device}')
                params2 = self.supervised_model.encode2lat_params(s2_r, s2_a)
                kl_loss = self.beta_kl * self.lat_space.kl(params, params2).sum(1).mean()

            loss_sum += kl_loss
            loss_dict['kl_loss'] = kl_loss.item()

        if self.beta_index > 0:
            rec_loss_fn = select_rec_loss_fn(self.decoder.loss_type)
            index_loss = self.beta_index * self.decoder.ssimulator.index_loss(s2_r, rec, lossfn=rec_loss_fn)
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
        y = self.encode(s2_r, s2_a)
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
        if not self.multi_output_encoder:
            loss_sum = self.lat_space.loss(ref_lat, params, reduction_nll=reduction_nll)
            if loss_sum.isnan().any() or loss_sum.isinf().any():
                raise ValueError
            all_losses = {'lat_loss': loss_sum.item()}
        else:
            main_loss = self.lat_space.loss(ref_lat, params[-1,:,:,:])
            all_losses = {'main_lat_loss': main_loss.item()}
            loss_sum = torch.zeros_like(main_loss)
            loss_sum += main_loss
            for i in range(params.size(0)):
                loss_i = self.lat_space.loss(ref_lat, params[i,:,:,:])
                loss_sum += loss_i / params.size(0)
                all_losses[f"{i}_lat_loss"] = loss_i.item()

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
        y = self.encode(s2_r, s2_a)
        params = self.lat_space.get_params_from_encoder(y=y)
        if not self.multi_output_encoder:
            nll = self.lat_space.loss(ref_lat, params, reduction=None, reduction_nll=None)
            if nll.isnan().any() or nll.isinf().any():
                raise ValueError
        else:
            raise NotImplementedError
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
        # for i, batch in zip(range(batch_per_epoch), test_dataloader):
        if batch_per_epoch is None:
            batch_per_epoch = len(dataloader)
        for i, batch in zip(range(min(len(dataloader),batch_per_epoch)),dataloader):
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


# def gaussian_nll(x, mu, sigma, eps=1e-6, device='cpu'):
#     eps = torch.tensor(eps).to(device)
#     return (torch.square(x - mu) / torch.max(sigma, eps)).sum(1) +  \
#             torch.log(torch.max(sigma, eps)).sum(1)
from prosailvae.dist_utils import kl_tn_uniform, truncated_gaussian_nll
from prosailvae.utils import gaussian_nll, gaussian_nll_loss, full_gaussian_nll_loss, mse_loss

class lr_finder_elbo(nn.Module):
    def __init__(self, index_loss, beta_kl=1, beta_index=0, loss_type='diag_nll', ssimulator=None) -> None:
        super(lr_finder_elbo,self).__init__()
        self.beta_kl = beta_kl
        self.beta_index = beta_index
        self.index_loss = index_loss
        self.loss_type = loss_type
        self.ssimulator = ssimulator 
        pass

    def lr_finder_elbo_inner(self, model_outputs, label):
        dist_params, _, _, rec = model_outputs

        if self.ssimulator.apply_norm:
            label = self.ssimulator.normalize(label)
        rec_loss_fn = select_rec_loss_fn(self.loss_type)
        rec_loss = rec_loss_fn(label, rec)
        loss_sum = rec_loss.mean()
        sigma = dist_params[:, :, 1].squeeze()
        mu = dist_params[:, :, 0].squeeze()
        if self.beta_kl > 0:
            kl_loss = self.beta_kl * kl_tn_uniform(mu, sigma).sum(1).mean()
            loss_sum += kl_loss
        if self.beta_index>0:
            index_loss = self.beta_index * self.index_loss(label, rec, lossfn=rec_loss_fn)
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