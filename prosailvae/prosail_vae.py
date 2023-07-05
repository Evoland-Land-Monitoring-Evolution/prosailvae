#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 14:45:12 2022

@author: yoel
"""
from dataclasses import dataclass,field
import torch
from prosailvae.simvae import SimVAE
from prosailvae.encoders import get_encoder, EncoderConfig
from prosailvae.loss import LossConfig
# from prosailvae.decoders import TSSimulatorDecoder
from prosailvae.latentspace import TruncatedNormalLatent
from prosailvae.simspaces import LinearVarSpace
from prosailvae.ProsailSimus import (SensorSimulator, ProsailSimulator, get_z2prosailparams_offset,
                                     get_z2prosailparams_mat, get_prosailparams_pdf_span, 
                                     PROSAILVARS)
from prosailvae.decoders import ProsailSimulatorDecoder
from prosailvae.loss import NLLLoss

@dataclass
class ProsailVAEConfig:
    """
    Dataclass to hold all of PROSAIL_VAE configurations
    """
    encoder_config:EncoderConfig
    loss_config:LossConfig
    rsr_dir:str
    vae_load_file_path:str
    vae_save_file_path:str
    spatial_mode:bool=False
    load_vae:bool=False
    apply_norm_rec:bool = True
    inference_mode:bool = False
    prosail_bands:list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 7, 8, 11, 12])
    disabled_latent:list[int] = field(default_factory=lambda: [])
    disabled_latent_values:list[int] = field(default_factory=lambda: [])
    R_down:int=1

def get_prosail_vae_config(params, bands, io_coeffs,
                           inference_mode, prosail_bands, rsr_dir):
    """
    Get ProsailVAEConfig from params dict
    """
    # assert len(prosail_bands) == len(bands)
    n_idx = io_coeffs.idx.loc.size(0) if io_coeffs.idx.loc is not None else 0
    encoder_config = EncoderConfig(encoder_type=params['encoder_type'],
                                   input_size=len(bands) + 3 + n_idx,# + 2 * 3 + n_idx,
                                   output_size=len(PROSAILVARS),
                                   io_coeffs=io_coeffs,
                                   bands=bands,
                                   last_activation = None,
                                   n_latent_params=2,
                                   layer_sizes = params["layer_sizes"],
                                   kernel_sizes = params['kernel_sizes'],
                                   padding = "valid",
                                   first_layer_kernel = params["first_layer_kernel"],
                                   first_layer_size = params["first_layer_size"],
                                   block_layer_sizes = params["block_layer_sizes"],
                                   block_layer_depths = params["block_layer_depths"],
                                   block_kernel_sizes = params["block_kernel_sizes"],
                                   block_n = params["block_n"])
    spatial_encoder = get_encoder(encoder_config).get_spatial_encoding()
    if spatial_encoder:
        params["loss_type"] = "spatial_nll"
    assert len(bands) == len(params["rec_bands_loss_coeffs"])
    loss_config = LossConfig(supervised=params["supervised"],
                             beta_index=params['beta_index'],
                             beta_kl=params["beta_kl"],
                             loss_type=params["loss_type"],
                             reconstruction_bands_coeffs=torch.tensor(params["rec_bands_loss_coeffs"]).squeeze())

    return ProsailVAEConfig(encoder_config=encoder_config,
                            loss_config=loss_config,
                            rsr_dir=rsr_dir,
                            vae_load_file_path=params["vae_load_file_path"],
                            vae_save_file_path=params["vae_save_file_path"],
                            load_vae=params["load_model"],
                            apply_norm_rec=params["apply_norm_rec"],
                            inference_mode=inference_mode,
                            prosail_bands=prosail_bands,
                            disabled_latent=params["disabled_latent"],
                            disabled_latent_values=params["disabled_latent_values"],
                            R_down=params["R_down"])


def get_prosail_vae(pv_config:ProsailVAEConfig,
                    device:str='cpu',
                    logger_name:str='',
                    hyper_prior:SimVAE|None=None,
                    optimizer:torch.optim.Optimizer|None=None,
                    load_simulator=True,
                    freeze_weights=False):
    """
    Intializes an instance of prosail_vae
    """
    encoder = get_encoder(pv_config.encoder_config, device='cpu')
    if hyper_prior is not None:
        kl_type = "tntn"
    else:
        kl_type = "tnu"
    lat_space = TruncatedNormalLatent(device='cpu',
                                      latent_dim=pv_config.encoder_config.output_size,
                                      kl_type=kl_type,
                                      disabled_latent=pv_config.disabled_latent,
                                      disabled_latent_values = pv_config.disabled_latent_values)

    reconstruction_loss = NLLLoss(loss_type=pv_config.loss_config.loss_type, 
                                  feature_coeffs=pv_config.loss_config.reconstruction_bands_coeffs)

    z2sim_mat = get_z2prosailparams_mat()
    z2sim_offset = get_z2prosailparams_offset()
    sim_pdf_support_span = get_prosailparams_pdf_span()
    pheno_var_space = LinearVarSpace(latent_dim=pv_config.encoder_config.output_size,
                                     z2sim_mat=z2sim_mat,
                                     z2sim_offset=z2sim_offset,
                                     sim_pdf_support_span=sim_pdf_support_span,
                                     device='cpu')
    psimulator = ProsailSimulator(device='cpu', R_down=pv_config.R_down)
    if load_simulator:
        ssimulator = SensorSimulator(pv_config.rsr_dir + "/sentinel2.rsr", device='cpu',
                                     bands_loc=pv_config.encoder_config.io_coeffs.bands.loc,
                                     bands_scale=pv_config.encoder_config.io_coeffs.bands.scale,
                                     idx_loc=pv_config.encoder_config.io_coeffs.idx.loc,
                                     idx_scale=pv_config.encoder_config.io_coeffs.idx.scale,
                                     apply_norm=pv_config.apply_norm_rec,
                                     bands=pv_config.prosail_bands, R_down=pv_config.R_down)
    else:
        ssimulator = SensorSimulator(pv_config.rsr_dir + "/sentinel2.rsr", device='cpu',
                                    bands_loc=None,
                                    bands_scale=None,
                                    apply_norm=pv_config.apply_norm_rec,
                                    bands=pv_config.prosail_bands, R_down=pv_config.R_down)
    
    decoder = ProsailSimulatorDecoder(prosailsimulator=psimulator,
                                      ssimulator=ssimulator,
                                      loss_type=pv_config.loss_config.loss_type)

    prosail_vae = SimVAE(encoder=encoder, decoder=decoder,
                        lat_space=lat_space, sim_space=pheno_var_space, config=pv_config,
                        reconstruction_loss=reconstruction_loss,
                        supervised=pv_config.loss_config.supervised,
                        device='cpu',
                        beta_kl=pv_config.loss_config.beta_kl,
                        beta_index=pv_config.loss_config.beta_index,
                        logger_name=logger_name, inference_mode=pv_config.inference_mode,
                        lat_nll="lai_nll" if pv_config.loss_config.loss_type=="lai_nll" else "")
    prosail_vae.set_hyper_prior(hyper_prior)
    if pv_config.load_vae is not None and pv_config.vae_load_file_path is not None:
        _, _ = prosail_vae.load_ae(pv_config.vae_load_file_path, optimizer)

    prosail_vae.change_device(device)
    if freeze_weights:
        prosail_vae.freeze_weigths()
    return prosail_vae

def load_prosail_vae_with_hyperprior(logger_name:str,
                                     pv_config:ProsailVAEConfig,
                                     pv_config_hyper:ProsailVAEConfig|None=None):
    """
    Loads prosail vae with or without intializing weight, with optionnal hyperprior
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hyper_prior=None
    if pv_config_hyper is not None:
        hyper_prior = get_prosail_vae(pv_config_hyper, device=device,
                                      logger_name=logger_name,
                                      load_simulator=False, freeze_weights = True)

    prosail_vae = get_prosail_vae(pv_config, device=device,
                                        logger_name=logger_name,
                                        hyper_prior=hyper_prior)
    return prosail_vae