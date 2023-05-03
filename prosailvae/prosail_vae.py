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

def get_prosail_vae_config(params, bands, norm_mean, norm_std,
                           inference_mode, prosail_bands, rsr_dir, spatial_mode):
    """
    Get ProsailVAEConfig from params dict
    """
    # assert len(prosail_bands) == len(bands)
    encoder_config = EncoderConfig(encoder_type=params['encoder_type'],
                                   input_size=len(bands) + 2 * 3,
                                   output_size=len(PROSAILVARS),
                                   norm_mean=norm_mean,
                                   norm_std=norm_std,
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
                                   block_n = params["block_n"],
                                   spatial_mode=spatial_mode)
    
    loss_config = LossConfig(supervised=params["supervised"],
                             beta_index=params['beta_index'],
                             beta_kl=params["beta_kl"],
                             loss_type=params["loss_type"])

    return ProsailVAEConfig(encoder_config=encoder_config,
                            loss_config=loss_config,
                            rsr_dir=rsr_dir,
                            vae_load_file_path=params["vae_load_file_path"],
                            vae_save_file_path=params["vae_save_file_path"],
                            load_vae=params["load_model"],
                            apply_norm_rec=params["apply_norm_rec"],
                            inference_mode=inference_mode,
                            prosail_bands=prosail_bands)


def get_prosail_vae(pv_config:ProsailVAEConfig,
                    device:str='cpu',
                    logger_name:str='',
                    hyper_prior:SimVAE|None=None,
                    optimizer:torch.optim.Optimizer|None=None,
                    load_simulator=True):
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
                                      kl_type=kl_type)

    z2sim_mat = get_z2prosailparams_mat()
    z2sim_offset = get_z2prosailparams_offset()
    sim_pdf_support_span = get_prosailparams_pdf_span()
    pheno_var_space = LinearVarSpace(latent_dim=pv_config.encoder_config.output_size,
                                     z2sim_mat=z2sim_mat,
                                     z2sim_offset=z2sim_offset,
                                     sim_pdf_support_span=sim_pdf_support_span,
                                     device='cpu')
    psimulator = ProsailSimulator(device='cpu')
    if load_simulator:
        ssimulator = SensorSimulator(pv_config.rsr_dir + "/sentinel2.rsr", device='cpu',
                                    norm_mean=pv_config.encoder_config.norm_mean,
                                    norm_std=pv_config.encoder_config.norm_std,
                                    apply_norm=pv_config.apply_norm_rec,
                                    bands=pv_config.prosail_bands)
    else:
        ssimulator = SensorSimulator(pv_config.rsr_dir + "/sentinel2.rsr", device='cpu',
                                    norm_mean=None,
                                    norm_std=None,
                                    apply_norm=pv_config.apply_norm_rec,
                                    bands=pv_config.prosail_bands)
    decoder = ProsailSimulatorDecoder(prosailsimulator=psimulator,
                                        ssimulator=ssimulator,
                                        loss_type=pv_config.loss_config.loss_type)

    prosail_vae = SimVAE(encoder=encoder, decoder=decoder,
                        lat_space=lat_space, sim_space=pheno_var_space, config=pv_config,
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
                                      load_simulator=False)

    prosail_vae = get_prosail_vae(pv_config, device=device,
                                        logger_name=logger_name,
                                        hyper_prior=hyper_prior)
    return prosail_vae