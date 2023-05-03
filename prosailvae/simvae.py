#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 08:25:49 2022

@author: yoel
"""

import logging
import torch.nn as nn
import torch
from utils.utils import NaN_model_params
from utils.image_utils import unbatchify, crop_s2_input, batchify_batch_latent

class SimVAE(nn.Module):
    """
    A class used to represent an encoder with simulator-decoder.

    ...

    Attributes
    ----------
    encoder : Encoder
        A torch NN that encodes a time series into a low dimension vector to be interpreted 
        as distribution parameters.
    lat_space : LatentSpace
        A torch object representing the latent distributions produced by the encoder.
    sim_space : SimSpace
        A torch object representing the distribution of the decoder parameters, to be derived 
        from the latent distribution.
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
        Encode time series using attribute encoder and converts it into latent distribution 
        parameters.
    sample_latent_from_params(dist_params, n_samples=1)
        Outputs n_samples samples from latent distributions parametrized by dist_params.
    transfer_latent(z)
        Transforms latent distribution samples into samples of the distribution of 
        parameters of the decoder.
    decode(sim)
        Decode parameters using decoder and reconstruct time series.
    forward(x, n_samples=1)
        Output n_samples samples of distribution of reconstructions from encoding time series x. 
    point_estimate_rec(x, mode='random')
        Outputs the latent distribution parameters, a sample from the latent distribution, 
        a sample from the decoder parameters distribution and a reconstruction. 
        Samples can be random, the mode, the expectation, the median from distributions. 
        This is selected by mode.
    """
    def __init__(self, encoder, decoder, lat_space, sim_space, config,
                 supervised:bool=False,  device:str='cpu',
                 beta_kl:float=0, beta_index:float=0, logger_name:str='PROSAIL-VAE logger',
                 inference_mode:bool=False,
                 lat_nll:str=""):
        super(SimVAE, self).__init__()
        # encoder
        self.config = config
        self.encoder = encoder
        self.lat_space = lat_space
        self.sim_space = sim_space
        self.decoder = decoder
        # self.loss = loss
        self.encoder.eval()
        self.lat_space.eval()
        self.supervised = supervised
        self.device = device
        self.beta_kl = beta_kl
        self.eval()
        self.logger = logging.getLogger(logger_name)
        self.beta_index = beta_index
        self.inference_mode = inference_mode
        self.hyper_prior = None
        self.lat_nll = lat_nll
        self.spatial_mode = self.encoder.get_spatial_encoding()

    def set_hyper_prior(self, hyper_prior:nn.Module|None=None):
        self.hyper_prior = hyper_prior

    def change_device(self, device:str):
        """
        Changes all attributes to desired device
        """
        self.device=device
        self.encoder.change_device(device)
        self.lat_space.change_device(device)
        self.sim_space.change_device(device)
        self.decoder.change_device(device)
        if self.hyper_prior is not None:
            self.hyper_prior.change_device(device)

    def encode(self, s2_r, s2_a):
        """
        Uses encoder to encode data
        """
        y, angles = self.encoder.encode(s2_r, s2_a)
        return y, angles

    def encode2lat_params(self, s2_r, s2_a):
        """
        Uses encoder to encode data into latent distribution parameters
        """
        y, _ = self.encode(s2_r, s2_a)
        dist_params = self.lat_space.get_params_from_encoder(y)
        return dist_params

    def sample_latent_from_params(self, dist_params, n_samples=1):
        """
        Sample latent distribution
        """
        z = self.lat_space.sample_latent_from_params(dist_params, n_samples=n_samples)
        return z

    def transfer_latent(self, z):
        """
        Transform latent samples into physical variables
        """
        sim = self.sim_space.z2sim(z)
        return sim

    def decode(self, sim, angles, apply_norm=None):
        """
        Uses decoder to reconstruct data
        """
        rec = self.decoder.decode(sim, angles, apply_norm=apply_norm)
        return rec

    def forward(self, x, angles=None, n_samples=1, apply_norm=None):
        """
        Forward pass through the VAE
        """
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
        """
        Forward pass with point estimate of latent distribution
        """
        if mode == 'random':
            dist_params, z, sim, rec = self.forward(x, angles, n_samples=1, apply_norm=apply_norm)

        elif mode == 'lat_mode':
            y, angles = self.encode(x, angles)
            dist_params = self.lat_space.get_params_from_encoder(y)
            # latent mode
            z = self.lat_space.mode(dist_params)
            # transfer to simulator variable
            sim = self.transfer_latent(z.unsqueeze(2))
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

        if self.spatial_mode:# and mode != 'random':
            if mode == 'random':
                return unbatchify(dist_params), z, sim, rec
            return unbatchify(dist_params), z, unbatchify(sim), unbatchify(rec)
        return dist_params, z, sim, rec

    def unsupervised_batch_loss(self, batch, normalized_loss_dict, len_loader=1,
                                n_samples=1):
        """
        Computes the unsupervised loss on batch (ELBO)
        """
        s2_r = batch[0].to(self.device)
        s2_a = batch[1].to(self.device)
        # Forward Pass
        params, _, _, rec = self.forward(s2_r, n_samples=n_samples, angles=s2_a)
        # Reconstruction term
        if self.decoder.loss_type=='spatial_nll':
            s2_r = crop_s2_input(s2_r, self.encoder.nb_enc_cropped_hw)
            s2_a = crop_s2_input(s2_a, self.encoder.nb_enc_cropped_hw)
            rec_loss = self.decoder.loss(s2_r, rec)
        else:
            rec_loss = self.decoder.loss(s2_r, rec)

        loss_dict = {'rec_loss': rec_loss.item()}
        loss_sum = rec_loss
        # Kl term
        if self.beta_kl > 0:
            if self.hyper_prior is None: # KL Truncated Normal latent || Uniform prior
                kl_loss = self.beta_kl * self.lat_space.kl(params).sum(1).mean()
            else: # KL Truncated Normal latent || Truncated Normal hyperprior
                s2_r_sup = s2_r
                s2_a_sup = s2_a
                if self.spatial_mode: # if encoder 1 encodes patches
                    # if self.encoder.nb_enc_cropped_hw > 0: # Padding management
                    #     s2_r_sup = crop_s2_input(s2_r_sup, self.encoder.nb_enc_cropped_hw)
                    #     s2_a_sup = crop_s2_input(s2_a_sup, self.encoder.nb_enc_cropped_hw)
                    #     print(s2_r_sup.size())
                    if self.hyper_prior.encoder.get_spatial_encoding():
                        # Case of a spatial hyperprior
                        raise NotImplementedError
                    s2_r_sup = batchify_batch_latent(s2_r_sup)
                    s2_a_sup = batchify_batch_latent(s2_a_sup)
                params_hyper = self.hyper_prior.encode2lat_params(s2_r_sup, s2_a_sup)
                kl_loss = self.beta_kl * self.lat_space.kl(params, params_hyper).sum(1).mean()

            loss_sum += kl_loss
            loss_dict['kl_loss'] = kl_loss.item()

        if self.beta_index > 0:
            index_loss = self.beta_index * self.decoder.rec_loss_fn(s2_r, rec)
            loss_sum += index_loss
            loss_dict['index_loss'] = index_loss.item()

        loss_dict['loss_sum'] = loss_sum.item()

        for loss_type, loss in loss_dict.items():
            if loss_type not in normalized_loss_dict.keys():
                normalized_loss_dict[loss_type] = 0.0
            normalized_loss_dict[loss_type] += loss / len_loader
        return loss_sum, normalized_loss_dict

    def supervised_batch_loss(self, batch, normalized_loss_dict, len_loader=1):
        """
        Computes supervised loss on batch (gaussian NLL)
        """
        s2_r = batch[0].to(self.device)
        s2_a = batch[1].to(self.device) 
        ref_sim = batch[2].to(self.device) 
        ref_lat = self.sim_space.sim2z(ref_sim)
        encoder_outputs, _ = self.encode(s2_r, s2_a)
        if encoder_outputs.isnan().any() or encoder_outputs.isinf().any():
            nan_in_params = NaN_model_params(self)
            err_str = "NaN encountered during encoding, but there is no NaN in network parameters!"
            if nan_in_params:
                err_str = "NaN encountered during encoding, there are NaN in network parameters!"
            raise ValueError(err_str)
        params = self.lat_space.get_params_from_encoder(encoder_outputs=encoder_outputs)
        reduction_nll = "sum"
        if self.lat_nll == "lai_nll":
            reduction_nll = "lai"
        loss_sum = self.lat_space.supervised_loss(ref_lat, params, reduction_nll=reduction_nll)
        if loss_sum.isnan().any() or loss_sum.isinf().any():
            raise ValueError
        all_losses = {'lat_loss': loss_sum.item()}
        all_losses['loss_sum'] = loss_sum.item()
        for loss_type, loss in all_losses.items():
            if loss_type not in normalized_loss_dict.keys():
                normalized_loss_dict[loss_type] = 0.0
            normalized_loss_dict[loss_type] += loss / len_loader

        return loss_sum, normalized_loss_dict

    def compute_lat_nlls_batch(self, batch):
        """
        Computes NLL loss on batch 
        """
        s2_r = batch[0].to(self.device)
        s2_a = batch[1].to(self.device)
        ref_sim = batch[2].to(self.device)
        ref_lat = self.sim_space.sim2z(ref_sim)
        encoder_outputs, _ = self.encode(s2_r, s2_a)
        params = self.lat_space.get_params_from_encoder(encoder_outputs=encoder_outputs)
        nll = self.lat_space.supervised_loss(ref_lat, params, reduction=None, reduction_nll=None)
        if nll.isnan().any() or nll.isinf().any():
            raise ValueError
        return nll

    def compute_lat_nlls(self, dataloader, batch_per_epoch=None):
        """
        Computes NLL loss for all samples in dataloader
        """
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
    
    def save_ae(self, epoch: int, optimizer, loss, path: str):
        """
        Saves the neural network weights and optimizer state into file
        """
        hyper_prior = None
        if self.hyper_prior is not None: # Removing hyperprior before saving
            hyper_prior = self.hyper_prior.config # Not a deep copy, but it seems to work...
            self.set_hyper_prior(None) 
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
            }, path)
        if hyper_prior is not None:
            self.set_hyper_prior(hyper_prior)

    def load_ae(self, path:str, optimizer=None, weights_only:bool=False):
        """
        Loads neural network weights from file.
        """
        # map_location = 'cuda:0' if self.device != torch.device('cpu') else 'cpu'
        hyper_prior = None
        if self.hyper_prior is not None: # Removing hyperprior before saving
            hyper_prior = self.hyper_prior.config # Not a deep copy, but it seems to work...
            self.set_hyper_prior(None) 
        checkpoint = torch.load(path, map_location=self.device, weights_only=weights_only)
        try:
            self.load_state_dict(checkpoint['model_state_dict'])
        except Exception as exc:
            print("checkpoint state dict")
            print(checkpoint['model_state_dict'].keys())
            print("self state dict")
            print(self.state_dict().keys())
            print(exc)
            raise ValueError
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if hyper_prior is not None:
            self.set_hyper_prior(hyper_prior)
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        return epoch, loss

    def fit(self, dataloader, optimizer, n_samples=1,
            batch_per_epoch=None, max_samples=None):
        """
        Computes loss and steps optimizer for a whole epoch
        """
        self.train()
        train_loss_dict = {}
        len_loader = len(dataloader.dataset)
        if batch_per_epoch is None:
            batch_per_epoch = len(dataloader)
        for i, batch in zip(range(min(len(dataloader), batch_per_epoch)), dataloader):
            if NaN_model_params(self):
                self.logger.debug("NaN model parameters at batch %d!", i)
            if max_samples is not None:
                if i == max_samples:
                    break
            optimizer.zero_grad()
            try:
                if not self.supervised:
                    loss_sum, _ = self.unsupervised_batch_loss(batch, train_loss_dict,
                                                               n_samples=n_samples,
                                                               len_loader=len_loader,)
                else:
                    loss_sum, _ = self.supervised_batch_loss(batch, train_loss_dict,
                                                            len_loader=len_loader)
            except Exception as exc:
                self.logger.error("Couldn't compute loss at batch %d!", i)
                self.logger.error("s2_r : %d NaN", torch.isnan(batch[0]).sum().item())
                self.logger.error("s2_a : %d NaN", torch.isnan(batch[1]).sum().item())
                self.logger.error(exc)
                raise ValueError(f"Couldn't compute loss at batch {i}!") from exc

            if torch.isnan(loss_sum).any():
                self.logger.error("NaN Loss encountered during training at batch %d!", i)

            loss_sum.backward()
            optimizer.step()
            if NaN_model_params(self):
                self.logger.debug("NaN model parameters after batch %d!", i)
        self.eval()
        return train_loss_dict

    def validate(self, dataloader, n_samples=1, batch_per_epoch=None, max_samples=None):
        """
        Computes loss for a whole epoch
        """
        self.eval()
        valid_loss_dict = {}
        len_loader = len(dataloader.dataset)
        with torch.no_grad():
            if batch_per_epoch is None:
                batch_per_epoch = len(dataloader)
            for i, batch in zip(range(min(len(dataloader),batch_per_epoch)), dataloader):
                if max_samples is not None:
                    if i == max_samples:
                        break
                if not self.supervised:
                    loss_sum, _ = self.unsupervised_batch_loss(batch, valid_loss_dict,
                                                               n_samples=n_samples, 
                                                               len_loader=len_loader)
                else:
                    loss_sum, _ = self.supervised_batch_loss(batch, valid_loss_dict,
                                                        len_loader=len_loader)
            if torch.isnan(loss_sum).any():
                self.logger.error("NaN Loss encountered during validation !")
        return valid_loss_dict
