#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 14:45:12 2022

@author: yoel
"""

import torch
import prosailvae
from prosailvae.simvae import SimVAE
from prosailvae.encoders import ProsailNNEncoder, ProsailRNNEncoder, ProsailDNNEncoder, ProsailCNNEncoder, ProsailRCNNEncoder
# from prosailvae.decoders import TSSimulatorDecoder
from prosailvae.latentspace import OrderedTruncatedGaussianLatent
from prosailvae.simspaces import LinearVarSpace
from prosailvae.ProsailSimus import SensorSimulator, ProsailSimulator, get_z2prosailparams_offset, get_z2prosailparams_mat, get_prosailparams_pdf_span
from prosailvae.decoders import ProsailSimulatorDecoder
import os
from dataset.loaders import  get_norm_coefs
import time
import torch.optim as optim
from prosailvae.utils import gaussian_nll_loss

def select_encoder(encoder_type, vae_params, device, refl_norm_mean, refl_norm_std, rnn_number, rnn_depth, latent_dim=11):
    output_size = latent_dim * 2
    if encoder_type=='nn':

        encoder = ProsailNNEncoder(s2refl_size=vae_params["input_size"], 
                            output_size=output_size, 
                            hidden_layers_size=vae_params["hidden_layers_size"], 
                            last_activation=vae_params["encoder_last_activation"], 
                            device=device,
                            norm_mean=refl_norm_mean,
                            norm_std=refl_norm_std)
    elif encoder_type=='rnn':
        encoder = ProsailRNNEncoder(s2refl_size=vae_params["input_size"], 
                            output_size=output_size, 
                            n_res_block = rnn_number,
                            res_block_layer_sizes=vae_params["hidden_layers_size"][0],
                            res_block_layer_depth=rnn_depth, 
                            last_activation=vae_params["encoder_last_activation"], 
                            device=device,
                            norm_mean=refl_norm_mean,
                            norm_std=refl_norm_std)
    elif encoder_type=='rcnn':
        encoder = ProsailRCNNEncoder(s2refl_size=vae_params["input_size"], 
                            output_size=latent_dim, 
                            device=device,
                            norm_mean=refl_norm_mean,
                            norm_std=refl_norm_std)
    elif encoder_type=='dnn':
        encoder = ProsailDNNEncoder(s2refl_size=vae_params["input_size"], 
                            output_size=output_size, 
                            n_res_block = rnn_number,
                            res_block_layer_sizes=vae_params["hidden_layers_size"][0],
                            res_block_layer_depth=rnn_depth, 
                            last_activation=vae_params["encoder_last_activation"], 
                            device=device,
                            norm_mean=refl_norm_mean,
                            norm_std=refl_norm_std)
    elif encoder_type=='cnn':
        encoder_sizes = [vae_params["input_size"] + 2 * 3] + vae_params["hidden_layers_size"] + [output_size]
        enc_kernel_sizes = [3] * len(encoder_sizes)
        encoder = ProsailCNNEncoder(encoder_sizes=encoder_sizes, enc_kernel_sizes=enc_kernel_sizes, device=device)
    else:
        raise NotImplementedError    
    return encoder

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
                    flat_patch_mode=False,
                    apply_norm_rec=True,
                    inference_mode=False,
                    loss_type="diag_nll",
                    supervised_model=None,
                    encoder_type='rnn',
                    rnn_depth=2,
                    rnn_number=3, 
                    bands = [1,2,3,4,5,6,7,8,11,12]):
    latent_dim=11
    encoder = select_encoder(encoder_type, vae_params, device, refl_norm_mean, refl_norm_std, rnn_number,rnn_depth, latent_dim=latent_dim)
    if supervised_model is not None:
        kl_type = "tntn"
    else:
        kl_type = "tnu"
    lat_space = OrderedTruncatedGaussianLatent(device=device, 
                                               latent_dim=latent_dim,
                                               max_matrix=None,
                                               kl_type=kl_type)
    
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
                                 apply_norm=apply_norm_rec, 
                                 bands=bands)
    sigmo_decoder = ProsailSimulatorDecoder(prosailsimulator=psimulator,
                                            ssimulator=ssimulator,
                                            loss_type=loss_type)
    
    prosailVAE = SimVAE(encoder=encoder, decoder=sigmo_decoder, 
                      lat_space=lat_space, sim_space=pheno_var_space, 
                      supervised=vae_params["supervised"],  
                      device=device, 
                      beta_kl=vae_params["beta_kl"],
                      beta_index=vae_params["beta_index"],
                      logger_name=logger_name, flat_patch_mode=flat_patch_mode, inference_mode=inference_mode,
                      supervised_model=supervised_model,
                      lat_nll="lai_nll" if loss_type=="lai_nll" else "")
    return prosailVAE

def load_prosailVAE(rsr_dir, vae_params, vae_file_path, optimizer=None, device='cpu',
                    refl_norm_mean=None, refl_norm_std=None,
                                            logger_name=None, flat_patch_mode=None,
                                            apply_norm_rec=None,
                                            loss_type=None, sup_prosail_vae=None, encoder_type='rnn',
                                            rnn_depth=2,
                                            rnn_number=3, bands=[1,2,3,4,5,6,7,8,11,12]):

    prosail_vae = get_prosail_VAE(rsr_dir, vae_params, device='cpu', 
                                    refl_norm_mean=refl_norm_mean, refl_norm_std=refl_norm_std,
                                    logger_name=logger_name, flat_patch_mode=flat_patch_mode,
                                    apply_norm_rec=apply_norm_rec, loss_type=loss_type, supervised_model=sup_prosail_vae, 
                                    encoder_type=encoder_type,
                                            rnn_depth=rnn_depth,
                                            rnn_number=rnn_number, bands=bands)

    nb_epoch, loss = prosail_vae.load_ae(vae_file_path, optimizer)
    prosail_vae.change_device(device)
    return prosail_vae, nb_epoch, loss

def load_PROSAIL_VAE_with_supervised_kl(params, rsr_dir, data_dir, logger_name, vae_file_path=None, 
                                        params_sup_kl_model=None, weiss_mode=False):
    bands = [1,2,3,4,5,6,7,8,11,12]
    if weiss_mode:
        bands = [2, 3, 4, 5, 6, 8, 11, 12]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sup_prosail_vae=None
    if params_sup_kl_model is not None:
        patch_mode_sup_model = not params_sup_kl_model["simulated_dataset"]
        vae_params={"input_size":len(bands),  
                "hidden_layers_size":params_sup_kl_model["hidden_layers_size"], 
                "encoder_last_activation":params_sup_kl_model["encoder_last_activation"],
                "supervised":params_sup_kl_model["supervised"],  
                "beta_kl":params_sup_kl_model["beta_kl"],
                "beta_index":params_sup_kl_model["beta_index"],
                }
        norm_mean, norm_std = get_norm_coefs(data_dir, params_sup_kl_model["dataset_file_prefix"])
        sup_prosail_vae, _, _ = load_prosailVAE(rsr_dir, vae_params=vae_params,vae_file_path=params_sup_kl_model['sup_model_weights_path'],
                                            optimizer=None, device=device,
                                            refl_norm_mean=norm_mean, refl_norm_std=norm_std,
                                            logger_name=logger_name, flat_patch_mode=patch_mode_sup_model,
                                            apply_norm_rec=params_sup_kl_model["apply_norm_rec"],
                                            loss_type=params_sup_kl_model["loss_type"],
                                            encoder_type=params_sup_kl_model["encoder_type"],
                                            rnn_depth=params_sup_kl_model["rnn_depth"],
                                            rnn_number=params_sup_kl_model["rnn_number"],
                                            bands=bands)
    
    vae_params={"input_size":len(bands),  
                "hidden_layers_size":params["hidden_layers_size"], 
                "encoder_last_activation":params["encoder_last_activation"],
                "supervised":params["supervised"],  
                "beta_kl":params["beta_kl"],
                "beta_index":params["beta_index"],
                }
    norm_mean, norm_std = get_norm_coefs(data_dir, params["dataset_file_prefix"])

    vae_flat_patch_mode = not params["simulated_dataset"] and params["encoder_type"] != "cnn"
    if vae_file_path is not None:
        prosail_vae, _, _ = load_prosailVAE(rsr_dir, vae_params=vae_params,vae_file_path=vae_file_path,
                                            optimizer=None, device=device,
                                            refl_norm_mean=norm_mean, refl_norm_std=norm_std,
                                            logger_name=logger_name, flat_patch_mode=vae_flat_patch_mode,
                                            apply_norm_rec=params["apply_norm_rec"],
                                            loss_type=params["loss_type"], sup_prosail_vae=sup_prosail_vae,
                                            encoder_type=params["encoder_type"],
                                            rnn_depth=params["rnn_depth"],
                                            rnn_number=params["rnn_number"],
                                            bands=bands)
    else:
        prosail_vae = get_prosail_VAE(rsr_dir, vae_params, device=device, 
                                    refl_norm_mean=norm_mean, refl_norm_std=norm_std,
                                    logger_name=logger_name, flat_patch_mode=vae_flat_patch_mode,
                                    apply_norm_rec=params["apply_norm_rec"], loss_type=params["loss_type"], 
                                    supervised_model=sup_prosail_vae,
                                    encoder_type=params["encoder_type"],
                                    rnn_depth=params["rnn_depth"],
                                    rnn_number=params["rnn_number"], bands=bands)
    
    return prosail_vae
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