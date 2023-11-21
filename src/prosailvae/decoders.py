#!/usr/bin/env python3
"""
Created on Wed Aug 31 14:23:46 2022

@author: yoel
"""
import torch
import torch.nn as nn

from utils.utils import (
    full_gaussian_nll_loss,
    gaussian_nll_loss,
    mse_loss,
    select_rec_loss_fn,
)


class Decoder(nn.Module):
    def decode(self):
        raise NotImplementedError()

    def loss(self):
        raise NotImplementedError()


class ProsailSimulatorDecoder(Decoder):
    def __init__(
        self, prosailsimulator, ssimulator, device="cpu", loss_type="diag_nll"
    ):
        super().__init__()
        self.device = device
        self.prosailsimulator = prosailsimulator
        self.ssimulator = ssimulator
        self.loss_type = loss_type
        self.nbands = len(ssimulator.bands)
        self.rec_loss_fn = select_rec_loss_fn(self.loss_type)

    def change_device(self, device):
        self.device = device
        self.ssimulator.change_device(device)
        self.prosailsimulator.change_device(device)
        pass

    def decode(self, z, angles, apply_norm=None):
        n_samples = z.size(2)
        batch_size = z.size(0)

        sim_input = (
            torch.concat((z, angles.unsqueeze(2).repeat(1, 1, n_samples)), axis=1)
            .transpose(1, 2)
            .reshape(n_samples * batch_size, -1)
        )
        prosail_output = self.prosailsimulator(sim_input)
        rec = (
            self.ssimulator(prosail_output, apply_norm=apply_norm)
            .reshape(batch_size, n_samples, -1)
            .transpose(1, 2)
        )
        return rec

    def loss(self, tgt, rec):
        if self.ssimulator.apply_norm:
            tgt = self.ssimulator.normalize(tgt)
        rec_loss = self.rec_loss_fn(tgt, rec)
        return rec_loss
