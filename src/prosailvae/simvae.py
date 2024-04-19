#!/usr/bin/env python3
"""
Created on Thu Sep  1 08:25:49 2022

@author: yoel
"""

import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

# from sensorsio.utils import rgb_render
# import matplotlib.pyplot as plt
from .bvnet_regression.bvnet_utils import (
    get_bvnet_biophyiscal_from_batch,
    get_bvnet_biophyiscal_from_pixellic_batch,
)
from .latentspace import LatentSpace
from .simspaces import SimVarSpace
from .utils.image_utils import (
    batchify_batch_latent,
    check_is_patch,
    crop_s2_input,
    unbatchify,
)
from .utils.utils import NaN_model_params, count_parameters, unstandardize


@dataclass
class SimVAEConfig:
    """

    Attributes
    ----------
    encoder : Encoder
        A torch NN that encodes a time series into a low dimension vector to be
        interpreted as distribution parameters.
    decoder : Decoder
        A torch object that decodes samples of parameters distributions from sim_space.
    lat_space : LatentSpace
        A torch object representing the latent distributions produced by the encoder.
    sim_space : SimSpace
        A torch object representing the distribution of the decoder parameters, to be
        derived from the latent distribution.

    supervised : bool
        Indicate whether the Encoder is to be trained from a labelled dataset or not.
    lat_idx: torch.Tensor | list[int] | None = None
        Indices of the latent vars on which the KL will be applied. None means all.

    """

    encoder: nn.Module
    decoder: nn.Module
    lat_space: LatentSpace
    sim_space: SimVarSpace
    reconstruction_loss: nn.Module
    deterministic: bool = False
    index_loss: Any | None = None
    supervised: bool = False
    device: str = "cpu"
    beta_kl: float = 0
    beta_index: float = 0
    logger_name: str = "PROSAIL-VAE logger"
    beta_cyclical: float = 0.0
    snap_cyclical: bool = False
    inference_mode: bool = False
    lat_idx: torch.Tensor | list[int] | None = None
    disabled_latent = None
    disabled_latent_values = None
    lat_nll: str = "diag_nll"


class SimVAE(nn.Module):
    """
    A class used to represent an encoder with simulator-decoder.

    Methods
    -------
    encode(x)
        Encode time series in x using attribute encoder.
    encode2lat_params(x):
        Encode time series using attribute encoder and converts it into latent
    distribution
        parameters.
    sample_latent_from_params(dist_params, n_samples=1)
        Outputs n_samples samples from latent distributions parametrized by dist_params.
    transfer_latent(z)
        Transforms latent distribution samples into samples of the distribution of
        parameters of the decoder.
    decode(sim)
        Decode parameters using decoder and reconstruct time series.
    forward(x, n_samples=1)
        Output n_samples samples of distribution of reconstructions from encoding
    time series x.
    point_estimate_rec(x, mode='random')
        Outputs the latent distribution parameters, a sample from the latent
    distribution,
        a sample from the decoder parameters distribution and a reconstruction.
        Samples can be random, the mode, the expectation, the median from distributions.
        This is selected by mode.
    """

    def __init__(self, config: SimVAEConfig):
        match config.lat_idx:
            case None:
                self.lat_idx = torch.tensor([])
            case torch.Tensor:
                self.lat_idx = config.lat_idx
            case _:
                self.lat_idx = torch.tensor([])
        if config.disabled_latent is None:
            config.disabled_latent = []
        if config.disabled_latent_values is None:
            config.disabled_latent_values = []
        super().__init__()
        # encoder
        self.encoder = config.encoder
        self.lat_space = config.lat_space
        self.sim_space = config.sim_space
        self.decoder = config.decoder
        self.reconstruction_loss = config.reconstruction_loss
        self.index_loss = config.index_loss
        self.encoder.eval()
        self.lat_space.eval()
        self.supervised = config.supervised
        self.device = config.device
        self.beta_kl = config.beta_kl
        self.eval()
        self.logger = logging.getLogger(config.logger_name)
        self.logger.info(
            f"Number of trainable parameters: {count_parameters(self.encoder)}"
        )
        self.beta_index = config.beta_index
        self.inference_mode = config.inference_mode
        self.hyper_prior = None
        self.lat_nll = config.lat_nll
        self.spatial_mode = self.encoder.get_spatial_encoding()
        self.deterministic = config.deterministic
        self.beta_cyclical = config.beta_cyclical

        self.snap_cyclical = config.snap_cyclical
        if self.snap_cyclical:
            self.lat_idx = torch.tensor([6]).int()
            self.lat_nll = "lai_nll"

    def set_hyper_prior(self, hyper_prior: nn.Module | None = None):
        self.hyper_prior = hyper_prior

    def change_device(self, device: str):
        """
        Changes all attributes to desired device
        """
        self.device = device
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

    def encode2lat_params(self, s2_r, s2_a, deterministic=False):
        """
        Uses encoder to encode data into latent distribution parameters
        """
        y, _ = self.encode(s2_r, s2_a)
        dist_params = self.lat_space.get_params_from_encoder(y)
        if deterministic:
            dist_params[..., 1] = 0.0
        return dist_params

    def sample_latent_from_params(self, dist_params, n_samples=1, deterministic=False):
        """
        Sample latent distribution
        """
        z = self.lat_space.sample_latent_from_params(
            dist_params, n_samples=n_samples, deterministic=deterministic
        )
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

    def freeze_weigths(self):
        """
        Freeze weights of the model
        """
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, angles=None, n_samples=1, apply_norm=None):
        """
        Forward pass through the VAE
        Returns:
            dist_params: distribution parameters
            z: latent space normalized
            sim: denormalized latent space
            rec: reconstruction of input
        """
        is_patch = check_is_patch(x)
        # encoding
        if angles is None:
            angles = x[:, -3:]
            x = x[:, :-3]
        batch_size = x.size(0)
        y, angles = self.encode(x, angles)
        dist_params = self.lat_space.get_params_from_encoder(y)
        if self.inference_mode:
            return dist_params, None, None, None
        # latent sampling
        z = self.sample_latent_from_params(
            dist_params, n_samples=n_samples, deterministic=self.deterministic
        )
        # transfer to simulator variable
        sim = self.transfer_latent(z)
        # decoding
        rec = self.decode(sim, angles, apply_norm=apply_norm)
        if is_patch:
            return (
                dist_params,
                unbatchify(z, batch_size=batch_size),
                unbatchify(sim, batch_size=batch_size),
                unbatchify(rec, batch_size=batch_size),
            )
        else:
            return dist_params, z, sim, rec

    def point_estimate_rec(self, x, angles, mode="random", apply_norm=False):
        """
        Forward pass with point estimate of latent distribution
        """
        is_patch = check_is_patch(x)
        if angles is None:
            angles = x[:, -3:]
            x = x[:, :-3]
        y, angles = self.encode(x, angles)
        dist_params = self.lat_space.get_params_from_encoder(y)
        if mode == "random":
            if self.inference_mode:
                return dist_params, None, None, None
            # latent sampling
            z = self.sample_latent_from_params(dist_params, n_samples=1)
            # transfer to simulator variable
            sim = self.transfer_latent(z)

        elif mode == "lat_mode":
            # latent mode
            z = self.lat_space.mode(dist_params)
            # transfer to simulator variable
            sim = self.transfer_latent(z.unsqueeze(2))

        elif mode == "sim_tg_mean":
            z = self.lat_space.expectation(dist_params)
            # transfer to simulator variable
            sim = self.transfer_latent(z.unsqueeze(2))

        elif mode == "sim_mode":
            lat_pdfs, lat_supports = self.lat_space.latent_pdf(dist_params)
            sim = self.sim_space.sim_mode(
                lat_pdfs, lat_supports, n_pdf_sample_points=5001
            )
            z = self.sim_space.sim2z(sim)
            # Quickfix for angle dimension:
            if len(angles.size()) == 4:
                angles = angles.permute(0, 2, 3, 1)
                angles = angles.reshape(-1, 3)

        elif mode == "sim_median":
            lat_pdfs, lat_supports = self.lat_space.latent_pdf(dist_params)
            sim = self.sim_space.sim_median(lat_pdfs, lat_supports, n_samples=5001)
            z = self.sim_space.sim2z(sim)

        elif mode == "sim_expectation":
            lat_pdfs, lat_supports = self.lat_space.latent_pdf(dist_params)
            sim = self.sim_space.sim_expectation(lat_pdfs, lat_supports, n_samples=5001)
            z = self.sim_space.sim2z(sim)

        else:
            raise NotImplementedError()
        rec = self.decode(sim, angles, apply_norm=apply_norm)
        if is_patch:  # and mode != 'random':
            batch_size = x.size(0)
            rec = unbatchify(rec, batch_size=batch_size)
            sim = unbatchify(sim, batch_size=batch_size)
            if not mode == "random":
                dist_params = unbatchify(dist_params, batch_size=batch_size)
        return dist_params, z, sim, rec

    def point_estimate_sim(self, x, angles, mode="random", unbatch=True):
        is_patch = check_is_patch(x)
        if angles is None:
            angles = x[:, -3:]
            x = x[:, :-3]
        y, angles = self.encode(x, angles)
        dist_params = self.lat_space.get_params_from_encoder(y)
        if mode == "random":
            if self.inference_mode:
                return dist_params, None, None, None
            # latent sampling
            z = self.sample_latent_from_params(dist_params, n_samples=1)

            # transfer to simulator variable
            sim = self.transfer_latent(z)

        elif mode == "lat_mode":
            # latent mode
            z = self.lat_space.mode(dist_params)
            # transfer to simulator variable
            sim = self.transfer_latent(z.unsqueeze(2))

        elif mode == "sim_tg_mean":
            z = self.lat_space.expectation(dist_params)
            # transfer to simulator variable
            sim = self.transfer_latent(z.unsqueeze(2))

        elif mode == "sim_mode":
            lat_pdfs, lat_supports = self.lat_space.latent_pdf(dist_params)
            sim = self.sim_space.sim_mode(
                lat_pdfs, lat_supports, n_pdf_sample_points=5001
            )
            z = self.sim_space.sim2z(sim)
            # Quickfix for angle dimension:
            if len(angles.size()) == 4:
                angles = angles.permute(0, 2, 3, 1)
                angles = angles.reshape(-1, 3)

        elif mode == "sim_median":
            lat_pdfs, lat_supports = self.lat_space.latent_pdf(dist_params)
            sim = self.sim_space.sim_median(lat_pdfs, lat_supports, n_samples=5001)
            z = self.sim_space.sim2z(sim)

        elif mode == "sim_expectation":
            lat_pdfs, lat_supports = self.lat_space.latent_pdf(dist_params)
            sim = self.sim_space.sim_expectation(lat_pdfs, lat_supports, n_samples=5001)
            z = self.sim_space.sim2z(sim)

        else:
            raise NotImplementedError()

        if is_patch and unbatch:  # and mode != 'random':
            batch_size = x.size(0)
            if mode == "random":
                return unbatchify(dist_params, batch_size=batch_size), z, sim
            return (
                unbatchify(dist_params, batch_size=batch_size),
                z,
                unbatchify(sim, batch_size=batch_size),
            )
        return dist_params, z, sim

    def crop_patch(
        self, s2_r: torch.Tensor, s2_a: torch.Tensor, z: torch.Tensor, rec: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert check_is_patch(rec)
        s2_r = crop_s2_input(s2_r, self.encoder.nb_enc_cropped_hw)
        s2_a = crop_s2_input(s2_a, self.encoder.nb_enc_cropped_hw)
        z = crop_s2_input(z, self.encoder.nb_enc_cropped_hw)

    def compute_rec_loss(self, s2_r: torch.Tensor, rec: torch.Tensor) -> torch.Tensor:
        if self.decoder.ssimulator.apply_norm:
            return self.reconstruction_loss(
                self.decoder.ssimulator.normalize(s2_r), rec
            )
        else:
            return self.reconstruction_loss(s2_r, rec)

    def compute_cyclical_loss(self, s2_r, s2_a, z, rec, batch_size, n_samples):
        sample_dim = self.reconstruction_loss.sample_dim
        feature_dim = self.reconstruction_loss.feature_dim

        if self.snap_cyclical:
            if self.spatial_mode:
                raise NotImplementedError
                snap_s2_r = s2_r
                snap_s2_a = s2_a
                if batch_size > 1:
                    snap_s2_r = torch.cat(snap_s2_r.split(1, dim=0), -1)
                    snap_s2_a = torch.cat(snap_s2_a.split(1, dim=0), -1)
                (snap_lai, snap_cab, snap_cw) = get_bvnet_biophyiscal_from_batch(
                    (snap_s2_r, snap_s2_a),
                    patch_size=s2_r.size(-1),
                    sensor="2A",
                    device=self.device,
                )
                snap_lai = snap_lai.reshape(-1, 1)
            else:
                (
                    snap_lai,
                    snap_cab,
                    snap_cw,
                ) = get_bvnet_biophyiscal_from_pixellic_batch(
                    (s2_r, s2_a), sensor="2A", device=self.device
                )
            snap_sim = torch.cat(
                (
                    torch.zeros_like(snap_lai),
                    torch.zeros_like(snap_lai),
                    torch.zeros_like(snap_lai),
                    torch.zeros_like(snap_lai),
                    torch.zeros_like(snap_lai),
                    torch.zeros_like(snap_lai),
                    snap_lai,
                    torch.zeros_like(snap_lai),
                    torch.zeros_like(snap_lai),
                    torch.zeros_like(snap_lai),
                    torch.zeros_like(snap_lai),
                ),
                1,
            ).to(self.device)
            snap_z = self.sim_space.sim2z(snap_sim)
            return (s2_r, s2_a, snap_z)

        else:
            rec_cyc = unstandardize(
                rec,
                self.encoder.bands_loc,
                self.encoder.bands_scale,
                dim=feature_dim,
            )
            # Fusing samples and batch dimensions together
            rec_cyc = rec_cyc.transpose(sample_dim, 1)
            rec_cyc = rec_cyc.reshape(-1, *rec_cyc.shape[2:])
            s2_a_cyc = s2_a.unsqueeze(sample_dim)
            s2_a_cyc = s2_a_cyc.tile(
                [
                    (n_samples if i == sample_dim else 1)
                    for i in range(len(s2_a_cyc.size()))
                ]
            )
            s2_a_cyc = s2_a_cyc.transpose(sample_dim, 1)
            s2_a_cyc = s2_a_cyc.reshape(-1, *s2_a_cyc.shape[2:])
            z_cyc = z.transpose(feature_dim, -1)
            z_cyc = z_cyc.reshape(-1, z_cyc.size(-1))
            return (rec_cyc, s2_a_cyc, z_cyc)

    def unsupervised_batch_loss(
        self, batch, normalized_loss_dict, len_loader=1, n_samples=1
    ):
        """
        Computes the unsupervised loss on batch (ELBO)
        """
        s2_r = batch[0]
        s2_a = batch[1]
        input_is_patch = check_is_patch(s2_r)
        batch_size = s2_r.size(0)
        if self.spatial_mode:  # self.decoder.loss_type=='spatial_nll':
            assert input_is_patch
        else:  # encoder is pixellic
            if input_is_patch:  # converting patch into batch
                s2_r = batchify_batch_latent(s2_r)
                s2_a = batchify_batch_latent(s2_a)
        # Forward Pass
        params, z, sim, rec = self.forward(s2_r, n_samples=n_samples, angles=s2_a)

        # cropping pixels lost to padding
        # TODO: FIX CROP PATCH DOESN4T RETURN ANYTHING
        if self.spatial_mode:
            s2_r, s2_a, z = self.crop_patch(s2_r, s2_a, z, rec)

        # Reconstruction term
        rec_loss = self.compute_rec_loss(s2_r, rec)

        loss_dict = {"rec_loss": rec_loss.item()}
        loss_sum = rec_loss

        if self.beta_cyclical > 0:
            cyclical_batch = self.compute_cyclical_loss(
                s2_r, s2_a, z, rec, batch_size, n_samples
            )
            cyclical_loss, _ = self.supervised_batch_loss(
                cyclical_batch, {}, ref_is_lat=True
            )
            loss_sum += self.beta_cyclical * cyclical_loss
            loss_dict["cyclical_loss"] = cyclical_loss.item()

        # Kl term
        if self.beta_kl > 0:
            if self.hyper_prior is None:  # KL Truncated Normal latent || Uniform prior
                kl_loss = (
                    self.beta_kl
                    * self.lat_space.kl(params, lat_idx=self.lat_idx).sum(1).mean()
                )
            else:  # KL Truncated Normal latent || Truncated Normal hyperprior
                s2_r_sup = s2_r
                s2_a_sup = s2_a
                if self.spatial_mode:  # if encoder 1 encodes patches
                    if self.hyper_prior.encoder.get_spatial_encoding():
                        # Case of a spatial hyperprior
                        raise NotImplementedError
                    s2_r_sup = batchify_batch_latent(s2_r_sup)
                    s2_a_sup = batchify_batch_latent(s2_a_sup)
                with torch.no_grad():
                    params_hyper = self.hyper_prior.encode2lat_params(
                        s2_r_sup, s2_a_sup
                    )
                kl_loss = (
                    self.beta_kl
                    * self.lat_space.kl(params, params_hyper, lat_idx=self.lat_idx)
                    .sum(1)
                    .mean()
                )  # sum over latent and mean over batch

            loss_sum += kl_loss
            loss_dict["kl_loss"] = kl_loss.item()

        if self.beta_index > 0:
            index_loss = self.beta_index * self.decoder.ssimulator.index_loss(
                s2_r,
                rec,
                lossfn=self.decoder.rec_loss_fn,
                normalize_idx=True,
                s2_r_bands_dim=1,
                rec_bands_dim=self.decoder.rec_loss_fn.feature_dim,
            )  # self.decoder.rec_loss_fn(s2_r, rec)
            loss_sum += index_loss
            loss_dict["index_loss"] = index_loss.item()

        loss_dict["loss_sum"] = loss_sum.item()
        for loss_type, loss in loss_dict.items():
            if loss_type not in normalized_loss_dict.keys():
                normalized_loss_dict[loss_type] = 0.0
            normalized_loss_dict[loss_type] += loss
        return loss_sum, normalized_loss_dict

    def supervised_batch_loss(
        self, batch, normalized_loss_dict, len_loader=1, ref_is_lat=False
    ):
        """
        Computes supervised loss on batch (gaussian NLL)
        """
        s2_r = batch[0].to(self.device)
        s2_a = batch[1].to(self.device)
        ref_lat = batch[2].to(self.device)
        if not ref_is_lat:
            ref_lat = self.sim_space.sim2z(ref_lat)
        encoder_output, _ = self.encode(s2_r, s2_a)
        if encoder_output.isnan().any() or encoder_output.isinf().any():
            nan_in_params = NaN_model_params(self)
            err_str = (
                "NaN encountered during encoding, "
                "but there is no NaN in network parameters!"
            )
            if nan_in_params:
                err_str = (
                    "NaN encountered during encoding, "
                    "there are NaN in network parameters!"
                )
            raise ValueError(err_str)
        params = self.lat_space.get_params_from_encoder(encoder_output=encoder_output)
        reduction_nll = "sum"
        if self.lat_nll == "lai_nll":
            reduction_nll = "lai"
        loss_sum = self.lat_space.supervised_loss(
            ref_lat, params, reduction_nll=reduction_nll
        )
        if loss_sum.isnan().any() or loss_sum.isinf().any():
            raise ValueError
        all_losses = {"lat_loss": loss_sum.item()}
        all_losses["loss_sum"] = loss_sum.item()
        for loss_type, loss in all_losses.items():
            if loss_type not in normalized_loss_dict.keys():
                normalized_loss_dict[loss_type] = 0.0
            normalized_loss_dict[loss_type] += loss
        return loss_sum, normalized_loss_dict

    def compute_lat_nlls_batch(self, batch):
        """
        Computes NLL loss on batch
        """
        s2_r = batch[0].to(self.device)
        s2_a = batch[1].to(self.device)
        ref_sim = batch[2].to(self.device)
        ref_lat = self.sim_space.sim2z(ref_sim)
        encoder_output, _ = self.encode(s2_r, s2_a)
        params = self.lat_space.get_params_from_encoder(encoder_output=encoder_output)
        nll = self.lat_space.supervised_loss(
            ref_lat, params, reduction=None, reduction_nll=None
        )
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
            for _, batch in zip(
                range(min(len(dataloader), batch_per_epoch)), dataloader, strict=False
            ):
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
        if self.hyper_prior is not None:  # Removing hyperprior before saving
            hyper_prior = self.hyper_prior  # Not a deep copy, but it seems to work...
            self.set_hyper_prior(None)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
            },
            path,
        )
        if hyper_prior is not None:
            self.set_hyper_prior(hyper_prior)

    def load_ae(self, path: str, optimizer=None, weights_only: bool = False):
        """
        Loads neural network weights from file.
        """
        # map_location = 'cuda:0' if self.device != torch.device('cpu') else 'cpu'
        hyper_prior = None
        if self.hyper_prior is not None:  # Removing hyperprior before saving
            hyper_prior = self.hyper_prior  # Not a deep copy, but it seems to work...
            self.set_hyper_prior(None)
        checkpoint = torch.load(
            path, map_location=self.device, weights_only=weights_only
        )
        try:
            self.load_state_dict(checkpoint["model_state_dict"])
        except Exception as exc:
            print("checkpoint state dict")
            print(checkpoint["model_state_dict"].keys())
            print("self state dict")
            print(self.state_dict().keys())
            print(exc)
            raise ValueError from exc
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if hyper_prior is not None:
            self.set_hyper_prior(hyper_prior)
        epoch = checkpoint["epoch"]
        loss = checkpoint["loss"]
        return epoch, loss

    def fit(self, dataloader, optimizer, n_samples=1, max_samples=None, accum_iter=1):
        """
        Computes loss and steps optimizer for a whole epoch
        """
        if max_samples is not None:
            accum_iter = min(accum_iter, max_samples)
        self.train()
        train_loss_dict = {}
        len_loader = len(dataloader.dataset)
        n_batches = 0
        for batch_idx, batch in enumerate(dataloader):
            if NaN_model_params(self):
                self.logger.debug("NaN model parameters at batch %d!", batch_idx)
            if max_samples is not None:
                if batch_idx == max_samples:
                    break
            try:
                n_batches += batch[0].size(0)
                if not self.supervised:
                    loss_sum, train_loss_dict = self.unsupervised_batch_loss(
                        batch,
                        train_loss_dict,
                        n_samples=n_samples,
                        len_loader=len_loader,
                    )
                else:
                    loss_sum, train_loss_dict = self.supervised_batch_loss(
                        batch, train_loss_dict, len_loader=len_loader
                    )
            except Exception as exc:
                self.logger.error("Couldn't compute loss at batch %d!", batch_idx)
                self.logger.error("s2_r : %d NaN", torch.isnan(batch[0]).sum().item())
                self.logger.error("s2_a : %d NaN", torch.isnan(batch[1]).sum().item())
                self.logger.error(exc)
                raise ValueError(
                    f"Couldn't compute loss at batch {batch_idx}!"
                ) from exc

            if torch.isnan(loss_sum).any():
                self.logger.error(
                    "NaN Loss encountered during training at batch %d!", batch_idx
                )
            loss_sum = loss_sum / accum_iter
            loss_sum.backward()
            if ((batch_idx + 1) % accum_iter == 0) or (
                batch_idx + 1 == len(dataloader)
            ):
                optimizer.step()

                if NaN_model_params(self):
                    self.logger.debug("NaN model parameters after batch %d!", batch_idx)
                optimizer.zero_grad()
        for loss_type, loss in train_loss_dict.items():
            train_loss_dict[loss_type] = loss / n_batches
        self.eval()
        return train_loss_dict

    def validate(self, dataloader, n_samples=1, batch_per_epoch=None, max_samples=None):
        """
        Computes loss for a whole epoch
        """
        self.eval()
        valid_loss_dict = {}
        len_loader = len(dataloader.dataset)
        n_batches = 0
        with torch.no_grad():
            if batch_per_epoch is None:
                batch_per_epoch = len(dataloader)
            for i, batch in zip(
                range(min(len(dataloader), batch_per_epoch)), dataloader, strict=False
            ):
                n_batches += batch[0].size(0)
                if max_samples is not None:
                    if i == max_samples:
                        break
                if not self.supervised:
                    loss_sum, _ = self.unsupervised_batch_loss(
                        batch,
                        valid_loss_dict,
                        n_samples=n_samples,
                        len_loader=len_loader,
                    )
                else:
                    loss_sum, _ = self.supervised_batch_loss(
                        batch, valid_loss_dict, len_loader=len_loader
                    )
            if torch.isnan(loss_sum).any():
                self.logger.error("NaN Loss encountered during validation !")
        for loss_type, loss in valid_loss_dict.items():
            valid_loss_dict[loss_type] = loss / n_batches
        return valid_loss_dict

    def get_cyclical_loss_from_batch(self, batch, n_samples=1):
        s2_r = batch[0].to(self.device)
        s2_a = batch[1].to(self.device)
        input_is_patch = check_is_patch(s2_r)
        if self.spatial_mode:  # self.decoder.loss_type=='spatial_nll':
            assert input_is_patch
        else:  # encoder is pixellic
            if input_is_patch:  # converting patch into batch
                s2_r = batchify_batch_latent(s2_r)
                s2_a = batchify_batch_latent(s2_a)
        # Forward Pass
        _, z, sim, rec = self.forward(s2_r, n_samples=n_samples, angles=s2_a)
        if self.spatial_mode:
            assert check_is_patch(rec)
            s2_a = crop_s2_input(s2_a, self.encoder.nb_enc_cropped_hw)
            z = crop_s2_input(z, self.encoder.nb_enc_cropped_hw)

        sample_dim = self.reconstruction_loss.sample_dim
        feature_dim = self.reconstruction_loss.feature_dim
        rec_cyc = unstandardize(
            rec, self.encoder.bands_loc, self.encoder.bands_scale, dim=feature_dim
        )
        rec_cyc = rec_cyc.transpose(sample_dim, 1)
        rec_cyc = rec_cyc.reshape(-1, *rec_cyc.shape[2:])
        s2_a_cyc = s2_a.unsqueeze(sample_dim)
        s2_a_cyc = s2_a_cyc.tile(
            [(n_samples if i == sample_dim else 1) for i in range(len(s2_a_cyc.size()))]
        )
        s2_a_cyc = s2_a_cyc.transpose(sample_dim, 1)
        s2_a_cyc = s2_a_cyc.reshape(-1, *s2_a_cyc.shape[2:])
        z_cyc = z.transpose(feature_dim, -1)
        z_cyc = z_cyc.reshape(-1, z_cyc.size(-1))
        cyclical_batch = (rec_cyc, s2_a_cyc, z_cyc)
        cyclical_loss, _ = self.supervised_batch_loss(
            cyclical_batch, {}, ref_is_lat=True
        )
        return cyclical_loss

    def get_cyclical_lai_squared_error_from_batch(
        self, batch, mode="lat_mode", lai_precomputed=False
    ):
        s2_r = batch[0].to(self.device)
        s2_a = batch[1].to(self.device)
        lai_idx = 6
        input_is_patch = check_is_patch(s2_r)
        if self.spatial_mode:  # self.decoder.loss_type=='spatial_nll':
            assert input_is_patch
        else:  # encoder is pixellic
            if input_is_patch:  # converting patch into batch
                s2_r = batchify_batch_latent(s2_r)
                s2_a = batchify_batch_latent(s2_a)
        sample_dim = self.reconstruction_loss.sample_dim
        feature_dim = self.reconstruction_loss.feature_dim
        if self.supervised:
            feature_dim = 1
            sample_dim = 2
        # Forward Pass
        if not lai_precomputed:
            _, z, sim, s2_r = self.point_estimate_rec(
                s2_r, angles=s2_a, mode=mode
            )  # computing reconstruction and prosail vars

            if self.spatial_mode:  # cropping encoder output due to padding
                assert check_is_patch(s2_r)
                s2_a = crop_s2_input(s2_a, self.encoder.nb_enc_cropped_hw)
                sim = crop_s2_input(sim, self.encoder.nb_enc_cropped_hw)

            s2_r = s2_r.transpose(sample_dim, 1)
            s2_r = s2_r.reshape(-1, *s2_r.shape[2:])

            s2_a = s2_a.unsqueeze(sample_dim)
            s2_a = s2_a.tile([1 for i in range(len(s2_a.size()))])
            s2_a = s2_a.transpose(sample_dim, 1)
            s2_a = s2_a.reshape(-1, *s2_a.shape[2:])
        else:
            sim = batch[2].to(self.device)
            if self.spatial_mode:
                sim = crop_s2_input(sim, self.encoder.nb_enc_cropped_hw)
            else:
                sim = batchify_batch_latent(sim)
                pass
        _, _, sim_cyc = self.point_estimate_sim(
            s2_r, s2_a, mode=mode
        )  # Predicting PROSAIL vars from reconstruction
        if self.spatial_mode:
            if len(sim.size()) < 5:
                sim = sim.unsqueeze(1)

        return (
            sim_cyc.select(feature_dim, lai_idx) - sim.select(feature_dim, lai_idx)
        ).pow(2)

    def get_cyclical_metrics_from_loader(
        self,
        dataloader,
        n_samples=1,
        batch_per_epoch=None,
        max_samples=None,
        lai_precomputed=False,
    ):
        """
        Computes loss for a whole epoch
        """
        self.eval()
        n_batches = 0
        with torch.no_grad():
            if batch_per_epoch is None:
                batch_per_epoch = len(dataloader)
                cyclical_loss = []
                cyclical_rmse = []
            for i, batch in zip(
                range(min(len(dataloader), batch_per_epoch)), dataloader, strict=False
            ):
                n_batches += batch[0].size(0)
                if max_samples is not None:
                    if i == max_samples:
                        break
                cyclical_rmse.append(
                    self.get_cyclical_lai_squared_error_from_batch(
                        batch, mode="lat_mode", lai_precomputed=lai_precomputed
                    ).unsqueeze(0)
                )
        # cyclical_loss = torch.cat(cyclical_loss).mean()
        cyclical_rmse = torch.cat(cyclical_rmse).mean().sqrt()
        return cyclical_loss, cyclical_rmse

    def get_cyclical_rmse_from_loader(
        self, dataloader, batch_per_epoch=None, max_samples=None, lai_precomputed=False
    ):
        """ """
        self.eval()
        n_batches = 0
        with torch.no_grad():
            if batch_per_epoch is None:
                batch_per_epoch = len(dataloader)
                cyclical_rmse = []
            for i, batch in zip(
                range(min(len(dataloader), batch_per_epoch)), dataloader, strict=False
            ):
                n_batches += batch[0].size(0)
                if max_samples is not None:
                    if i == max_samples:
                        break
                cyclical_rmse.append(
                    self.get_cyclical_lai_squared_error_from_batch(
                        batch, mode="lat_mode", lai_precomputed=lai_precomputed
                    ).reshape(-1)
                )

        cyclical_rmse = torch.cat(cyclical_rmse, 0).mean().sqrt()
        return cyclical_rmse

    def pvae_batch_extraction(self, batch: list) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts from batch , S2 refelectances and S2 angles

        INPUTS:
            batch: [reflectance bands, angles]. Shape: list[tensor, tensor]

        RETURNS:
            s2_r: S2 bands reflectances.
                  Shape: [batch, latent dim, width size, patch size]
            s2_a: S2 angles.
                  Shape: [batch, angles dim, patch size, patch size]
        """
        s2_r = batch[0]
        s2_a = batch[1]
        input_is_patch = check_is_patch(s2_r)
        if self.spatial_mode:  # self.decoder.loss_type=='spatial_nll':
            assert input_is_patch
        else:  # encoder is pixellic
            if input_is_patch:  # converting patch into batch
                s2_r = batchify_batch_latent(s2_r)
                s2_a = batchify_batch_latent(s2_a)
        return s2_r, s2_a

    def pvae_method(
        self, batch: list, n_samples: int = 70
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        pvae_method takes batch, passes it to VAE and outputs useful parameters

        INPUTS:
            batch: [reflectance bands, angles]. Shape: list[tensor, tensor]
        RETURNS:
            s2_r: S2 bands reflectances.
                  Shape: [batch, latent dim, width size, patch size]
            s2_a: S2 angles.
                  Shape: [batch, angles dim, patch size, patch size]
            distri_params: distribution paramaters of TN.
                           Shape: [(batch x width x height), variables, (mean sigma)]
            z: latent samples after sampling (normalized).
                           Shape: [(batch x width x height), variables, n_samples]
            sim: denormalized latent samples.
                           Shape: [(batch x width x height), variables, n_samples]
            rec: S2 reconstruction.
                           Shape: [(batch x width x height), bands, n_samples]
        """
        s2_r, s2_a = self.pvae_batch_extraction(batch)
        # Forward Pass
        distri_params, z, sim, rec = self.forward(
            s2_r, n_samples=n_samples, angles=s2_a
        )
        # TODO: change tuple for dataclass
        return s2_r, s2_a, distri_params, z, sim, rec

    def pvae_samples_2_distri_para(
        self, recs: torch.Tensor, sample_dim: int = 2
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes mean and var of each band. Method gotten from loss.gaussian_nll_loss

        INPUTS:
            recs: reconstruction from PM.
                Shape: [(batch x width x height), bands, n_samples]
        RETURNS:
            rec_mu: reconstruction mean.
                Shape: [(batch x width x height), bands, 0]
            rec_err_var: reconstruction variance.
                Shape: [(batch x width x height), bands, 0]
        """
        if len(recs.size()) < 3:
            raise ValueError("recs needs a batch, a feature and a sample dimension")
        if recs.size(sample_dim) == 1:
            rec_err_var = torch.tensor(0.0001).to(
                recs.device
            )  # constant variance, enabling computation even with 1 sample
            rec_mu = recs
        else:
            rec_err_var = recs.var(sample_dim, keepdim=True)  # .unsqueeze(sample_dim)
            rec_mu = recs.mean(sample_dim, keepdim=True)  # .unsqueeze(sample_dim)
        return rec_mu, rec_err_var

    def pvae_kl_elbo(
        self, s2_r: torch.Tensor, s2_a: torch.Tensor, distri_params: torch.Tensor
    ) -> torch.Tensor:
        """
        pvae_kl_term computes KL loss between output of encoder and prior

        INPUTS:
            s2_r: S2 reflectance. Shape: [width x height, bands]
            s2_a: S2 angles. Shape: [width x height, angles]
            z: latent samples after sampling (normalized)
               Shape: [(batch x width x height), variables, n_samples]
            distri_params: Truncated gaussian distribution parameters.
                           Shape: [(batch x width x height), variables, (mean sigma)]

        RETURNS:
            kl_loss: Sum of Kullbach-Leibler loss
        """
        # Kl term
        kl_loss = 0
        if self.beta_kl > 0:
            if self.hyper_prior is None:  # KL Truncated Normal latent || Uniform prior
                kl_loss = (
                    self.beta_kl
                    * self.lat_space.kl(distri_params, lat_idx=self.lat_idx)
                    .sum(1)
                    .mean()
                )
            else:  # KL Truncated Normal latent || Truncated Normal hyperprior
                s2_r_sup = s2_r
                s2_a_sup = s2_a
                if self.spatial_mode:  # if encoder 1 encodes patches
                    if self.hyper_prior.encoder.get_spatial_encoding():
                        # Case of a spatial hyperprior
                        raise NotImplementedError
                    s2_r_sup = batchify_batch_latent(s2_r_sup)
                    s2_a_sup = batchify_batch_latent(s2_a_sup)
                with torch.no_grad():
                    params_hyper = self.hyper_prior.encode2lat_params(
                        s2_r_sup, s2_a_sup
                    )
                kl_loss = (
                    self.beta_kl
                    * self.lat_space.kl(
                        distri_params, params_hyper, lat_idx=self.lat_idx
                    )
                    .sum(1)
                    .mean()
                )  # sum over latent and mean over batch
        return kl_loss
