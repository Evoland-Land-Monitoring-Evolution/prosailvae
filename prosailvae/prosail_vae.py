#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 14:45:12 2022

@author: yoel
"""

import torch
import prosailvae
from prosailvae.simvae import SimVAE
from prosailvae.encoders import ProsailNNEncoder
# from prosailvae.decoders import TSSimulatorDecoder
from prosailvae.latentspace import OrderedTruncatedGaussianLatent
from prosailvae.simspaces import LinearVarSpace
from prosailvae.ProsailSimus import SensorSimulator, ProsailSimulator, get_z2prosailparams_offset, get_z2prosailparams_mat, get_prosailparams_pdf_span
from prosailvae.decoders import ProsailSimulatorDecoder
import os
from dataset.loaders import get_simloader
import time
import torch.optim as optim

def get_prosail_VAE(rsr_dir, 
                    vae_params={"input_size":10,  
                                "hidden_layers_size":[400, 500, 300, 100], 
                                "encoder_last_activation":None, 
                                "supervised":False,  
                                "beta_kl":1,
                                "beta_index":1}, 
                    device='cpu',
                    refl_norm_mean=None,
                    refl_norm_std=None,
                    logger_name='',
                    patch_mode=False,
                    apply_norm_rec=True,
                    inference_mode=False):
    latent_dim = 11
    output_size = latent_dim * 2
    
    encoder = ProsailNNEncoder(s2refl_size=vae_params["input_size"], 
                        output_size=output_size, 
                        hidden_layers_size=vae_params["hidden_layers_size"], 
                        last_activation=vae_params["encoder_last_activation"], 
                        device=device,
                        norm_mean=refl_norm_mean,
                        norm_std=refl_norm_std)
    lat_space = OrderedTruncatedGaussianLatent(device=device, 
                                               latent_dim=latent_dim,
                                               max_matrix=None)
    
    z2sim_mat = get_z2prosailparams_mat()
    z2sim_offset = get_z2prosailparams_offset()
    sim_pdf_support_span = get_prosailparams_pdf_span()
    pheno_var_space = LinearVarSpace(latent_dim=latent_dim, 
                                     z2sim_mat=z2sim_mat, 
                                     z2sim_offset=z2sim_offset, 
                                     sim_pdf_support_span=sim_pdf_support_span,  
                                     device=device)
    psimulator = ProsailSimulator(device=device)
    ssimulator = SensorSimulator(rsr_dir + "/sentinel2.rsr", device=device,
                                 norm_mean=refl_norm_mean,
                                 norm_std=refl_norm_std,
                                 apply_norm=apply_norm_rec)
    sigmo_decoder = ProsailSimulatorDecoder(prosailsimulator=psimulator,
                                            ssimulator=ssimulator)
    
    prosailVAE = SimVAE(encoder=encoder, decoder=sigmo_decoder, 
                      lat_space=lat_space, sim_space=pheno_var_space, 
                      supervised=vae_params["supervised"],  
                      device=device, 
                      beta_kl=vae_params["beta_kl"],
                      beta_index=vae_params["beta_index"],
                      logger_name=logger_name, patch_mode=patch_mode, inference_mode=inference_mode)
    return prosailVAE

def load_prosailVAE(vae_params, vae_file_path, optimizer=None, device='cpu'):
    pheno_vae = get_prosail_VAE(vae_params, device=device).to(device)
    nb_epoch, loss = pheno_vae.load_ae(vae_file_path, optimizer)
    pheno_vae.eval()
    return pheno_vae, nb_epoch, loss

# if __name__ == "__main__":
#     data_dir = os.path.join(os.path.join(os.path.dirname(prosailvae.__file__),os.pardir),"data/")
#     prosailVAE = get_prosail_VAE(data_dir)
#     psimulator = ProsailSimulator()
#     ssimulator = SensorSimulator(data_dir + "/sentinel2.rsr")
    
#     test_loader = get_simloader(file_prefix="test_")
#     train_loader = get_simloader(file_prefix="train_")
#     valid_loader = get_simloader(file_prefix="valid_")
#     # t0=time.time()
#     # valid_loss_dict = prosailVAE.validate(dataloader=valid_loader)
#     # t1=time.time()
#     # print(f"simple validation is {t1-t0} s")
#     t0=time.time()
#     optimizer = optim.Adam(prosailVAE.parameters(), lr=0.0001)
#     train_loss_dict = prosailVAE.fit(dataloader=train_loader, optimizer=optimizer)
#     t1=time.time()
#     print(f"simple fit is {t1-t0} s")