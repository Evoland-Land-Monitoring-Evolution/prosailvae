#!/usr/bin/env python3
"""
Created on Wed Nov  9 13:39:15 2022

@author: yoel
"""
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch

from prosailvae.spectral_indices import get_spectral_idx
from utils.utils import gaussian_nll_loss, standardize, unstandardize

band_rsr_index = {
    "B2": 1,
    "B3": 2,
    "B4": 3,
    "B5": 4,
    "B6": 5,
    "B7": 6,
    "B8": 7,
    "B8A": 8,
    "B11": 11,
    "B12": 12,
}


class SensorSimulator:
    """Simulates the reflectances of a sensor from a full spectrum and the
    RSR of the sensor.

        The rsr_file is supposed to follow the 6S format: lambdas are given
        in micrometers and regularly sampled with a 0.001 step (1 nm). The
        input full spectrum can be a Prosail simulation and will be sampled
        with the same step. The wavelength ranges of the RSR are typically
        [0.350,2400] and Prosail gives [400, 2500]. The output will be given
        in [400, 2500] as the input.

        The RSR file columns are: lambda, solar spectrum and bands.
    """

    def __init__(
        self,
        prospect_range: tuple[int, int] = (400, 2500),
        bands=[1, 2, 3, 4, 5, 6, 7, 8, 11, 12],
        device="cpu",
        bands_loc=None,
        bands_scale=None,
        idx_loc=None,
        idx_scale=None,
        apply_norm=True,
        spectral_resolution=1,
    ):
        super().__init__()
        rsr_path = ""
        self.spectral_resolution = spectral_resolution
        self.bands_rsr_idx = [band_rsr_index[b] for b in bands]

        self.device = device
        self.prospect_range = prospect_range
        self.rsr = torch.from_numpy(np.loadtxt(rsr_path, unpack=True)).to(device)
        self.nb_bands = self.rsr.shape[0] - 2
        self.rsr_range = (
            int(self.rsr[0, 0].item() * 1000),
            int(self.rsr[0, -1].item() * 1000),
        )
        self.nb_lambdas = prospect_range[1] - prospect_range[0] + 1
        self.rsr_prospect = torch.zeros([self.rsr.shape[0], self.nb_lambdas]).to(device)
        self.rsr_prospect[0, :] = torch.linspace(
            prospect_range[0], prospect_range[1], self.nb_lambdas
        ).to(device)
        self.rsr_prospect[
            1:, : -(self.prospect_range[1] - self.rsr_range[1])
        ] = self.rsr[1:, (self.prospect_range[0] - self.rsr_range[0]) :]

        self.solar = self.rsr_prospect[1, :].unsqueeze(0)
        self.rsr = self.rsr_prospect[2:, :].unsqueeze(0)
        self.rsr = self.rsr[:, self.bands_rsr_idx, :]

        bands_loc = (
            bands_loc if bands_loc is not None else torch.zeros(len(self.bands_rsr_idx))
        )
        bands_scale = (
            bands_scale
            if bands_scale is not None
            else torch.ones(len(self.bands_rsr_idx))
        )
        idx_loc = idx_loc if idx_loc is not None else torch.zeros(5)
        idx_scale = idx_scale if idx_scale is not None else torch.ones(5)

        self.bands_loc = bands_loc.float().to(device)
        self.bands_scale = bands_scale.float().to(device)
        self.idx_loc = idx_loc.float().to(device)
        self.idx_scale = idx_scale.float().to(device)
        self.apply_norm = apply_norm

        self.s2norm_factor_d = (self.rsr * self.solar).sum(axis=2)
        self.s2norm_factor_n = self.rsr * self.solar
        if self.spectral_resolution > 1:
            self.s2norm_factor_n = (
                self.s2norm_factor_n[:, :, :-1]
                .reshape(1, len(self.bands_rsr_idx), -1, self.spectral_resolution)
                .mean(3)
                * self.spectral_resolution
            )

    def change_device(self, device):
        self.device = device
        self.rsr = self.rsr.to(device)
        self.bands_loc = self.bands_loc.to(device)
        self.bands_scale = self.bands_scale.to(device)
        self.idx_loc = self.idx_loc.to(device)
        self.idx_scale = self.idx_scale.to(device)
        self.s2norm_factor_d = self.s2norm_factor_d.to(device)
        self.s2norm_factor_n = self.s2norm_factor_n.to(device)
        self.solar = self.solar.to(device)

    def __call__(self, prosail_output: torch.Tensor, apply_norm=None):
        return self.forward(prosail_output, apply_norm=apply_norm)

    def apply_s2_sensor(self, prosail_output: torch.Tensor) -> torch.Tensor:
        # The input should have shape = batch, wavelengths, otherwise,
        # we add the first dimension
        x = prosail_output
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        elif len(x.shape) == 2:
            x = x.unsqueeze(1)
        simu = (self.s2norm_factor_n * x).sum(axis=2) / self.s2norm_factor_d
        return simu

    def normalize(self, s2_r, bands_dim: int = 1):
        return standardize(s2_r, self.bands_loc, self.bands_scale, dim=bands_dim)

    def unnormalize(self, s2_r, bands_dim=1):
        return unstandardize(s2_r, self.bands_loc, self.bands_scale, dim=bands_dim)

    def forward(self, prosail_output: torch.Tensor, apply_norm=None) -> torch.Tensor:
        simu = self.apply_s2_sensor(prosail_output)
        if apply_norm is None:  # using SensorSimulator apply_norm as default
            apply_norm = self.apply_norm
        if apply_norm:
            simu = self.normalize(simu)
        return simu  # type: ignore

    def index_loss(
        self,
        s2_r,
        s2_rec,
        lossfn=gaussian_nll_loss,
        normalize_idx=True,
        s2_r_bands_dim=1,
        rec_bands_dim=2,
    ):
        # u_s2_r = self.unnormalize(s2_r)
        if self.apply_norm:
            u_s2_rec = self.unnormalize(s2_rec, rec_bands_dim)
        else:
            u_s2_rec = s2_rec

        spectral_idx_tgt = get_spectral_idx(s2_r=s2_r, bands_dim=s2_r_bands_dim)
        spectral_idx_rec = get_spectral_idx(s2_r=u_s2_rec, bands_dim=rec_bands_dim)
        # spectral_idx_rec = spectral_idx_rec.mean(s2_r_bands_dim, keepdim=True)

        if normalize_idx:
            spectral_idx_tgt = standardize(
                spectral_idx_tgt, self.idx_loc, self.idx_scale, s2_r_bands_dim
            )
            spectral_idx_rec = standardize(
                spectral_idx_rec, self.idx_loc, self.idx_scale, rec_bands_dim
            )
        loss = lossfn(spectral_idx_tgt, spectral_idx_rec)

        return loss
