#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 08:24:27 2022

@author: yoel
"""
import torch
import torch.nn as nn
import numpy as np
from .dist_utils import convolve_pdfs, pdfs2cdfs, cdfs2quantiles
from utils.TruncatedNormal import TruncatedNormal

class SimVarSpace(nn.Module):
    def lat2sim(self):
        raise NotImplementedError

class LinearVarSpace(SimVarSpace):
    def __init__(self, latent_dim=6, z2sim_mat=None, z2sim_offset=None, 
                 sim_pdf_support_span=None,  device='cpu'):
        super().__init__()
        self.device=device
        self.eps=1e-3
        self.latent_dim=latent_dim
        self.z2sim_mat = z2sim_mat.to(device)
        self.z2sim_offset = z2sim_offset.to(device)
        self.sim_pdf_support_span = sim_pdf_support_span
        if z2sim_mat is None:
            self.z2sim_mat = torch.eye(latent_dim).to(device)
        if z2sim_offset is None:
            self.z2sim_offset = torch.zeros((1,latent_dim)).to(device)
        if sim_pdf_support_span is None:
            self.sim_pdf_support_span = 2 * torch.ones((1,latent_dim)).view(1,-1, 1).to(device)
        self.inv_z2sim_mat = torch.from_numpy(np.linalg.inv(self.z2sim_mat.detach().cpu())).to(self.device)


    def change_device(self, device):
        self.device=device
        self.z2sim_mat = self.z2sim_mat.to(device)
        self.z2sim_offset = self.z2sim_offset.to(device)
        self.sim_pdf_support_span = self.sim_pdf_support_span .to(device)
        self.inv_z2sim_mat = self.inv_z2sim_mat.to(device)

    def get_distribution_from_lat_params(self,lat_params, distribution_type="tn"):
        if distribution_type == "tn":
            lat_mu = lat_params[..., 0]
            sim_mu = self.z2sim(lat_mu)
            lat_sigma = lat_params[..., 1]
            lat_sigma2 = lat_sigma.pow(2)
            sim_sigma2 = None
            sim_sigma = sim_sigma2.sqrt()
            distribution = TruncatedNormal(loc=sim_mu, scale=sim_sigma,
                                           low=torch.zeros_like(sim_mu),
                                           high=torch.ones_like(sim_mu))
            pass
        else:
            raise NotImplementedError
        return distribution
    
    def z2sim(self, z):
        sim = torch.matmul(self.z2sim_mat, z) + self.z2sim_offset
        return sim

    def sim2z(self, sim):
        if len(sim.size())==2:
            sim=sim.unsqueeze(2)
        z = torch.matmul(self.inv_z2sim_mat, sim
                         - self.z2sim_offset)
        return z

    def sim_pdf(self, pdfs, supports, n_pdf_sample_points=3001):
        sim_pdfs = torch.zeros((pdfs.size(0), self.z2sim_mat.size(0),
                                n_pdf_sample_points)).to(self.device)
        sim_supports = torch.zeros((pdfs.size(0), self.z2sim_mat.size(0),
                                    n_pdf_sample_points)).to(self.device)
        for i in range(self.latent_dim):
            transfer_mat_line = self.z2sim_mat[i]
            sim_pdf, sim_support = convolve_pdfs(pdfs, supports,
                                                 transfer_mat_line,
                                                 n_pdf_sample_points=n_pdf_sample_points,
                                                 support_max=self.sim_pdf_support_span[i])
            sim_support = sim_support + self.z2sim_offset[i].item()
            sim_pdfs[:,i,:] = sim_pdf
            sim_supports[:,i,:] = sim_support
        return sim_pdfs, sim_supports

    def sim_mode(self, pdfs, supports, n_pdf_sample_points=3001):
        batch_size=pdfs.size(0)
        latent_size=pdfs.size(1)
        sim_pdfs, sim_supports = self.sim_pdf(pdfs, supports, 
                                              n_pdf_sample_points=n_pdf_sample_points)
        max_index = sim_pdfs.view(batch_size * latent_size,-1).argmax(dim=1).view(-1,1)
        sim_mode = torch.gather(sim_supports.view(batch_size * latent_size,-1),
                                dim=1, index=max_index).view(batch_size, latent_size,-1)
        return sim_mode

    def sim_quantiles(self, pdfs, supports, alpha=[0.5], n_pdf_sample_points=3001):
        sim_pdfs, sim_supports = self.sim_pdf(pdfs, supports, 
                                              n_pdf_sample_points=n_pdf_sample_points)
        sim_cdfs = pdfs2cdfs(sim_pdfs)
        quantiles = torch.zeros((pdfs.size(0), pdfs.size(1), len(alpha))).to(pdfs.device)
        # for batch_sizes greater than 1 :
        for i in range(pdfs.size(0)):
            quantiles[i,:,:] = cdfs2quantiles(sim_cdfs[i,:,:].cpu(),
                                              sim_supports[i,:,:].cpu(),
                                              alpha=alpha).to(pdfs.device)
        return quantiles

    def sim_median(self, pdfs, supports, n_pdf_sample_points=3001):
        sim_median = self.sim_quantiless(pdfs, supports, n_pdf_sample_points=n_pdf_sample_points,
                                          alpha=[0.5]).view(1,-1, 1)
        return sim_median

    def sim_expectation(self, pdfs, supports, n_pdf_sample_points=3001):
        sim_pdfs, sim_supports = self.sim_pdf(pdfs, supports,
                                              n_pdf_sample_points=n_pdf_sample_points)
        sampling = sim_supports[:, 1] - sim_supports[:, 0]
        sim_expected = (sim_pdfs * sim_supports) * sampling.view(-1,1)
        return sim_expected.sum(1).view(1,-1, 1)

    def sim_all_point_estimates(self, pdfs, supports, n_pdf_sample_points=3001):
        sim_pdfs, sim_supports = self.sim_pdf(pdfs, supports,
                                              n_pdf_sample_points=n_pdf_sample_points)
        sampling = sim_supports[:, 1] - sim_supports[:, 0]
        sim_expected = (sim_pdfs * sim_supports) * sampling.view(-1,1)
        sim_expected = sim_expected.sum(1).view(-1,1)
        sim_modes = torch.gather(sim_supports,dim=1, 
                                 index=sim_pdfs.argmax(dim=1).view(-1,1))
        sim_cdfs = pdfs2cdfs(sim_pdfs)
        sim_median = cdfs2quantiles(sim_cdfs, sim_supports, alpha=[0.5])[0]

        return sim_modes, sim_median, sim_expected
