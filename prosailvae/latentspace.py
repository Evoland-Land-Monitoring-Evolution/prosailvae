#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 14:54:24 2022

@author: yoel
"""
import torch.nn as nn
import torch
from .dist_utils import kl_tn_uniform, truncated_gaussian_cdf, ordered_truncated_gaussian_nll, get_latent_ordered_truncated_pdfs
eps=5e-4
MAX_MATRIX = torch.eye(10)

class LatentSpace(nn.Module):
    """ 
    A class used to represent the latent space of an auto-encoder. All latent spaces are to inherit from this class
    ...

    Methods
    -------
    reparametrize(y, n_samples)
        Uses encoding y of time series and outputs n_samples samples from latent distribution.
    latent_pdf(params, support_sampling)
        Uses latent distributions parameters derived from encoding of time series to output a discretized pdf of the latent distribution sampled with step support_sampling.
    loss(z, params)
        computes a loss (likely a NLL) of latent parameters params from sample z. 
    """
    def reparametrize(self, y, n_samples):
        raise NotImplementedError()
    
    def latent_pdf(self, params, support_sampling):
        raise NotImplementedError()
        
    def loss(self, z, params):
        raise NotImplementedError()

class OrderedTruncatedGaussianLatent(LatentSpace):
    def __init__(self, latent_dim=10, min_sigma=5e-4, max_sigma=1.4,
                 max_mu=1, device='cpu', max_matrix=MAX_MATRIX):
        super().__init__()
        self.device=device
        self.latent_dim = latent_dim
        self.max_sigma = torch.tensor(max_sigma).to(device)
        self.log_max_sigma = torch.log(self.max_sigma).to(self.device)
        self.min_sigma = torch.tensor(min_sigma).to(device)
        self.log_min_sigma = torch.log(self.min_sigma).to(device)
        # self.max_mu = max_mu.to(device)
        # self.eps=torch.tensor(5e-4).float().to(self.device)
        if max_matrix is None:
            self.max_matrix = torch.eye(latent_dim).to(device)
        else:
            assert max_matrix.size(0)==latent_dim
            self.max_matrix = max_matrix.to(self.device)
        pass
    
    def get_params_from_encoder(self, y):
        y = y.view(-1, 2, self.latent_dim)
        mu = torch.sigmoid(y[:, 0, :].view(-1, self.latent_dim, 1))
        ordered_mu = rectify(mu, self.max_matrix)
        logstd = torch.sigmoid(y[:, 1, :].view(-1, self.latent_dim, 1)) 
        logstd = logstd * (self.log_max_sigma - self.log_min_sigma) + self.log_min_sigma
        sigma = torch.exp(logstd) 
        params = torch.stack([ordered_mu, sigma], dim=2)
        return params
    
    def get_order_loss(self, y, params):
        y = y.view(-1, 2, self.latent_dim)
        mu = torch.sigmoid(y[:, 0, :].view(-1, self.latent_dim, 1)).squeeze(2)
        ordered_mu = params[:, :, 0].squeeze(2)
        order_loss = torch.relu(ordered_mu - mu).mean()
        return order_loss
    
    def reparametrize(self, y, n_samples=1):
        params, _ = self.get_params_from_encoder(y)
        z = self.sample_latent_from_params(params, n_samples=n_samples)
        return z
    
    def get_u_bounds(self, mu, sigma, n_sigma=4.5):
        u_ubound = truncated_gaussian_cdf(mu + n_sigma*sigma, mu, sigma)
        u_lbound = truncated_gaussian_cdf(mu - n_sigma*sigma, mu, sigma)
        return u_ubound, u_lbound
    
    def sample_latent_from_params(self, params, n_samples=1, n_sigma=4.5):
        mu = params[:, :, 0].squeeze(2)
        sigma = params[:, :, 1].squeeze(2)
        u_ubound, u_lbound = self.get_u_bounds(mu, sigma, n_sigma=n_sigma)
        try:
            u_dist = torch.distributions.uniform.Uniform(u_lbound, u_ubound)
        except:
            print("Inverted uniform bounds for Inverse transform method !")
            print(f"u : {u_ubound[u_ubound <= u_lbound]}")
            print(f"l : {u_lbound[u_ubound <= u_lbound]}")
            print(f"mu : {mu[u_ubound <= u_lbound]}")
            print(f"sigma : {sigma[u_ubound <= u_lbound]}")
            raise ValueError
        mu = params[:, :, 0].repeat(1, 1, n_samples) 
        sigma = params[:, :, 1].repeat(1, 1, n_samples) 
        n_dist = torch.distributions.normal.Normal(torch.zeros_like(mu), 
                                                   torch.ones_like(sigma))
        u = u_dist.rsample(torch.tensor([n_samples])).permute(1,2,0)
        z = mu + sigma * n_dist.icdf(n_dist.cdf((torch.zeros_like(mu)-mu)/sigma) + 
                                   u * (n_dist.cdf((torch.ones_like(mu)-mu)/sigma) 
                                        - n_dist.cdf((torch.zeros_like(mu)-mu)/sigma)))
        ordered_z = rectify(z, self.max_matrix)
        return ordered_z 
    
    def latent_pdf(self, params, support_sampling=0.001, min_interval_bound=1, 
                   n_sigma_interval=5):
        mu = params[:, :, 0]
        sigma = params[:, :, 1]
        pdfs, supports = get_latent_ordered_truncated_pdfs(mu, sigma, 
                                                           n_sigma_interval, 
                                                           support_sampling, 
                                                           self.max_matrix, 
                                                           latent_dim=self.latent_dim)
        return pdfs, supports
    
    def loss(self, z, params):
        mu = params[:, :, 0].squeeze()
        sigma = params[:, :, 1].squeeze()
        nll = ordered_truncated_gaussian_nll(z, mu, sigma, self.max_matrix, 
                                             device=self.device).mean()
        return nll
    
    def kl(self, params):
        sigma = params[:, :, 1].squeeze()
        mu = params[:, :, 0].squeeze()
        return kl_tn_uniform(mu, sigma) 


def rectify(z, max_matrix):
    max_matrix = max_matrix.unsqueeze(0).unsqueeze(3)
    z = z.unsqueeze(2)
    rectified_z = (max_matrix * z).max(axis=1)[0]
    return rectified_z
