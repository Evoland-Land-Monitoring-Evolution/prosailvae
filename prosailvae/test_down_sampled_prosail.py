#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 14:20:44 2022

@author: yoel
"""
import os
import sys
from prosailvae import __path__ as PPATH
TOP_PATH = os.path.join(PPATH[0], os.pardir)
from dataclasses import dataclass
import shutil
import logging
import logging.config
import traceback
import time
from prosail_vae import (load_prosail_vae_with_hyperprior, get_prosail_vae_config, ProsailVAEConfig)
from torch_lr_finder import get_PROSAIL_VAE_lr
from dataset.loaders import  (get_simloader, lr_finder_loader, get_train_valid_test_loader_from_patches)
from metrics.results import save_results, save_results_2d, get_res_dir_path
from utils.utils import load_dict, save_dict, get_RAM_usage, get_total_RAM, plot_grad_flow
from ProsailSimus import get_bands_idx, BANDS
import argparse
import pandas as pd
from dataset.generate_dataset import np_simulate_prosail_dataset

import socket
import os
import numpy as np
import torch.optim as optim
import torch
from tqdm import trange
from tqdm.contrib.logging import logging_redirect_tqdm

torch.autograd.set_detect_anomaly(True)
import matplotlib.pyplot as plt

CUDA_LAUNCH_BLOCKING=1
LOGGER_NAME = 'PROSAIL-VAE logger'



@dataclass
class DatasetConfig:
    dataset_file_prefix:str="small_test_"

@dataclass
class TrainingConfig:
    batch_size:int=128
    epochs:int=1
    lr:float=0.0001
    test_size:float=0.01
    valid_ratio:float=0.01
    k_fold:int=0
    n_fold:int|None=None
    n_samples:int=2

def get_training_config(params):
    return TrainingConfig(batch_size=params["batch_size"],
                          epochs=params["epochs"],
                          lr=params["lr"],
                          test_size = params["test_size"],
                          valid_ratio=params["valid_ratio"],
                          k_fold=params["k_fold"],
                          n_fold=params["n_fold"],
                          n_samples=params['n_samples'])

@dataclass
class ModelConfig:
    supervised:bool=False
    beta_kl:float=1
    beta_index:float=1


def get_prosailvae_train_parser():
    """
    Creates a new argument parser.
    """
    parser = argparse.ArgumentParser(description='Parser for data generation')

    parser.add_argument("-n", dest="n_fold",
                        help="number k of fold",
                        type=int, default=0)

    parser.add_argument("-c", dest="config_file",
                        help="name of config json file on config directory.",
                        type=str, default="config.json")

    parser.add_argument("-x", dest="n_xp",
                        help="Number of experience (to use in case of kfold)",
                        type=int, default=1)

    parser.add_argument("-o", dest="overwrite_xp",
                        help="Allow overwrite of experiment (fold)",
                        type=bool, default=True)

    parser.add_argument("-d", dest="data_dir",
                        help="path to data direcotry",
                        type=str, default="/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/")

    parser.add_argument("-r", dest="root_results_dir",
                        help="path to root results direcotry",
                        type=str, default="")

    parser.add_argument("-rsr", dest="rsr_dir",
                        help="directory of rsr_file",
                        type=str, default='/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/')

    parser.add_argument("-t", dest="tensor_dir",
                        help="directory of mmdc tensor files",
                        type=str,
                        default="/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/real_data/torchfiles/")

    parser.add_argument("-a", dest="xp_array",
                        help="array training (false for single xp) ",
                        type=bool, default=False)

    parser.add_argument("-p", dest="plot_results",
                        help="toggle results plotting",
                        type=bool, default=False)

    parser.add_argument("-w", dest="weiss_mode",
                        help="removes B2 and B8 bands for validation with weiss data",
                        type=bool, default=False)
    return parser

def recompute_lr(lr_scheduler, PROSAIL_VAE, epoch, lr_recompute, exp_lr_decay, logger, optimizer, lrtrainloader, 
                 old_lr=1.0, n_samples=1):
    new_lr=old_lr
    if epoch > 0 and lr_recompute is not None:
        if epoch % lr_recompute == 0:
            try:
                new_lr = get_PROSAIL_VAE_lr(PROSAIL_VAE, lrtrainloader, 
                                            old_lr=old_lr, old_lr_max_ratio=10, n_samples=n_samples)
                optimizer = optim.Adam(PROSAIL_VAE.parameters(), lr=new_lr, weight_decay=1e-2)
                if exp_lr_decay>0:
                    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, 
                                                                          gamma=exp_lr_decay)
            except Exception as exc:
                traceback.print_exc()
                print(exc)
                logger.error(f"Couldn't recompute lr at epoch {epoch} !")
                logger.error(f"{exc}")
                print(f"Couldn't recompute lr at epoch {epoch} !")
    return lr_scheduler, optimizer, new_lr

def switch_loss(epoch, n_epoch, PROSAIL_VAE, swith_ratio = 0.75):
    loss_type = PROSAIL_VAE.decoder.loss_type
    if loss_type == "hybrid_nll":
        if epoch > swith_ratio * n_epoch:
            PROSAIL_VAE.decoder.loss_type = "full_nll"


def load_params(config_dir, config_file, parser=None):
    """
    Load parameter dict form prosail vae and training
    """
    params = load_dict(config_dir + config_file)
    if params["supervised"]:
        params["simulated_dataset"]=True
    if not "load_model" in params.keys():
        params["load_model"]=False
    if not "vae_load_dir_path" in params.keys():
        params["vae_load_dir_path"]=None
    if not "lr_recompute_mode" in params.keys():
        params["lr_recompute_mode"]=False
    if not "init_model" in params.keys():
        params["init_model"] = False
    if parser is not None:
        params["k_fold"] = parser.n_xp
        params["n_fold"] = parser.n_fold if params["k_fold"] > 1 else None
    if "layer_sizes" not in params.keys():
        params["layer_sizes"] = [512, 512]
    if "kernel_sizes" not in params.keys():
        params['kernel_sizes'] = [3, 3]
    if "first_layer_kernel" not in params.keys():
        params["first_layer_kernel"] = 3
    if "first_layer_size" not in params.keys():
        params["first_layer_size"] = 128
    if "block_layer_sizes" not in params.keys():
        params["block_layer_sizes"] = [128, 128]
    if "block_layer_depths" not in params.keys():
        params["block_layer_depths"] = [2, 2]
    if "block_kernel_sizes" not in params.keys():
        params["block_kernel_sizes"] = [3, 1]
    if "block_n" not in params.keys():
        params["block_n"] = [1,3]
    if "supervised_kl" not in params.keys():
        params["supervised_kl"] = False
    if "weiss_bands" not in params.keys():
        params["weiss_bands"] = False
    params["vae_save_file_path"] = None
    if "supervised_config_file" not in params.keys():
        params["supervised_config_file"] = None
    if "supervised_weight_file" not in params.keys():
        params["supervised_weight_file"] = None
    if "disabled_latent" not in params.keys():
        params["disabled_latent"] = []
    if "disabled_latent_values" not in params.keys():
        params["disabled_latent_values"] = []
    if "cycle_training" not in params.keys():
        params["cycle_training"] = False
    if "R_down" not in params.keys():
        params["R_down"] = 1
    return params

def setup_training():
    """
    Read parser and config files to launch training
    """
    if socket.gethostname()=='CELL200973':
        args=["-n", "0",
              "-c", "config_dev.json",
              "-x", "1",
              "-o", "True",
              "-d", "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/sim_data/",#patches/",
              "-r", "",
              "-rsr", '/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/',
              "-t", "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/validation_tiles/",
              "-a", "False",
              "-p", "False"]
        parser = get_prosailvae_train_parser().parse_args(args)
        frm4veg_data_dir = "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/frm4veg_validation"
        frm4veg_barrax_filename = "2B_20180516_FRM_Veg_Barrax_20180605"
        frm4veg_wytham_filename = None #"2A_20180629_FRM_Veg_Wytham_20180703"
        belsar_dir = "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/belSAR_validation"
        
    else:
        parser = get_prosailvae_train_parser().parse_args()
        frm4veg_data_dir = "/work/scratch/zerahy/prosailvae/data/frm4veg_validation"
        belsar_dir = "/work/scratch/zerahy/prosailvae/data/belSAR_validation"
        # silvia_filename = "FRM_Veg_Barrax_20180605"
        frm4veg_barrax_filename = "2B_20180516_FRM_Veg_Barrax_20180605"
        frm4veg_wytham_filename = None #"2A_20180629_FRM_Veg_Wytham_20180703"
    list_belsar_filenames = ["2A_20180508_both_BelSAR_agriculture_database",
                            "2A_20180518_both_BelSAR_agriculture_database",
                            "2A_20180528_both_BelSAR_agriculture_database",
                            "2A_20180620_both_BelSAR_agriculture_database",
                            "2A_20180627_both_BelSAR_agriculture_database",
                            "2B_20180715_both_BelSAR_agriculture_database",
                            "2B_20180722_both_BelSAR_agriculture_database",
                            "2A_20180727_both_BelSAR_agriculture_database",
                            "2B_20180804_both_BelSAR_agriculture_database"]    
    root_dir = TOP_PATH
    xp_array = parser.xp_array
    job_array_dir = None
    if xp_array:
        job_array_dir = os.path.join(parser.root_results_dir, os.pardir)
    config_dir = os.path.join(root_dir,"config/")

    params = load_params(config_dir, config_file=parser.config_file, parser=parser)
    if "data_dir" not in params.keys():
        data_dir = os.path.join(root_dir,"data/")
    else:
        data_dir = params["data_dir"]
    assert parser.n_fold < parser.n_xp
    if len(parser.root_results_dir)==0:
        root_results_dir = os.path.join(TOP_PATH,"results/")
    else:
        root_results_dir = parser.root_results_dir
    res_dir = get_res_dir_path(root_results_dir, params, parser.n_xp, parser.overwrite_xp)
    save_dict(params, res_dir+"/config.json")
    params["vae_save_file_path"] = res_dir + "/prosailvae_weights.tar"

    logging.basicConfig(filename=res_dir+'/training_log.log',
                              level=logging.INFO, force=True)
    logger_name = 'PROSAIL-VAE logger'
    # create logger
    logger = logging.getLogger(logger_name)
    logger.info('Starting training of PROSAIL-VAE.')
    logger.info('========================================================================')
    logger.info('Parameters are : ')
    for _, key in enumerate(params):
        logger.info(f'{key} : {params[key]}')
    logger.info('========================================================================')
    if params["supervised_kl"]:
        logger.info("Supervised KL loss (hyperprior) enabled.")
        logger.info(f"copying {params['supervised_config_file']} into {res_dir+'/sup_kl_model_config.json'}")
        logger.info(f"copying {params['supervised_weight_file']} into {res_dir+'/sup_kl_model_weights.tar'}")
        logger.info(f"copying {os.path.join(os.path.dirname(params['supervised_config_file']), 'norm_mean.pt')} into {res_dir+'/sup_kl_norm_mean.pt'}")
        shutil.copyfile(params['supervised_config_file'], res_dir+"/sup_kl_model_config.json")
        shutil.copyfile(params['supervised_weight_file'], res_dir+"/sup_kl_model_weights.tar")
        shutil.copyfile(os.path.join(os.path.dirname(params["supervised_config_file"]), "norm_mean.pt"),
                        res_dir+"/sup_kl_norm_mean.pt")
        shutil.copyfile(os.path.join(os.path.dirname(params["supervised_weight_file"]), "norm_std.pt"),
                        res_dir+"/sup_kl_norm_std.pt")
        params_sup_kl_model = load_params(res_dir, "/sup_kl_model_config.json", parser=None)
        params_sup_kl_model['vae_load_file_path'] = res_dir + "/sup_kl_model_weights.tar"
        params_sup_kl_model["load_model"] = True
        sup_norm_mean = torch.load(res_dir + "/sup_kl_norm_mean.pt")
        sup_norm_std =torch.load(res_dir + "/sup_kl_norm_std.pt")
    else:
        params_sup_kl_model = None
        sup_norm_mean = None
        sup_norm_std = None
    return (params, parser, res_dir, data_dir, params_sup_kl_model, job_array_dir, sup_norm_mean, 
            sup_norm_std, frm4veg_data_dir, frm4veg_barrax_filename, frm4veg_wytham_filename, 
            belsar_dir, list_belsar_filenames)

def train_prosailvae(params, parser, res_dir, data_dir:str, params_sup_kl_model,
                     sup_norm_mean=None, sup_norm_std=None):
    """
    Intializes and trains a prosail instance
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.info(f"Loading training and validation loader in"
                f" {data_dir}/{params['dataset_file_prefix']}...")
    bands, prosail_bands = get_bands_idx(params["weiss_bands"])
    if params["simulated_dataset"]:
        if params["batch_size"] < 2:
            raise ValueError("With simulated data, batch_size cannot be inferior to 2.")
        train_loader, valid_loader = get_simloader(valid_ratio=params["valid_ratio"],
                                                    file_prefix=params["dataset_file_prefix"],
                                                    sample_ids=None,
                                                    batch_size=params["batch_size"],
                                                    data_dir=data_dir)
    else:
        train_loader, valid_loader, _ = get_train_valid_test_loader_from_patches(data_dir, batch_size=1,
                                                                                 num_workers=0, max_valid_samples=100)

    if params["apply_norm_rec"]:
        norm_mean = torch.load(os.path.join(data_dir, params["dataset_file_prefix"] + "norm_mean.pt"))#[bands]
        norm_std = torch.load(os.path.join(data_dir, params["dataset_file_prefix"] + "norm_std.pt"))#[bands]
        if isinstance(norm_mean, np.ndarray):
            norm_mean = torch.from_numpy(norm_mean)
        if isinstance(norm_std, np.ndarray):
            norm_std = torch.from_numpy(norm_std)
    else:
        norm_mean = torch.zeros(1, len(bands))
        norm_std = torch.ones(1, len(bands))
    torch.save(norm_mean, res_dir + "/norm_mean.pt")
    torch.save(norm_std, res_dir + "/norm_std.pt")
    logger.info(f'Training ({len(train_loader.dataset)} samples) '
                f'and validation ({len(valid_loader.dataset)} samples) loaders, loaded.')
    print(f"Weiss mode : {parser.weiss_mode}")
    if params["load_model"] : # and not socket.gethostname()=='CELL200973' :
        #"/home/uz/zerahy/scratch/prosailvae/results/cnn_39950033_jobarray/1_d2023_03_31_05_24_16_supervised_False_weiss_/prosailvae_weights.tar"
        vae_load_file_path = params["vae_load_dir_path"] + "/prosailvae_weights.tar"
        norm_mean = torch.load(os.path.join(params["vae_load_dir_path"], "norm_mean.pt"))
        norm_std = torch.load(os.path.join(params["vae_load_dir_path"], "norm_std.pt"))
        torch.save(norm_mean, res_dir + "/norm_mean.pt")
        torch.save(norm_std, res_dir + "/norm_std.pt")
    else:
        vae_load_file_path = None
    params["vae_load_file_path"] = vae_load_file_path
    training_config = get_training_config(params)
    params["R_down"]=1
    pv_config_1 = get_prosail_vae_config(params, bands = bands, prosail_bands=prosail_bands,
                                       inference_mode = False, rsr_dir=parser.rsr_dir,
                                       norm_mean = norm_mean, norm_std=norm_std)
    pv_config_hyper=None
    if params_sup_kl_model is not None:
        bands_hyper, prosail_bands_hyper = get_bands_idx(params_sup_kl_model["weiss_bands"])
        pv_config_hyper = get_prosail_vae_config(params_sup_kl_model, bands=bands_hyper,
                                                prosail_bands=prosail_bands_hyper,
                                                inference_mode=True, rsr_dir=parser.rsr_dir,
                                                norm_mean=sup_norm_mean, norm_std=sup_norm_std)
    prosail_vae_1 = load_prosail_vae_with_hyperprior(pv_config=pv_config_1,
                                                     pv_config_hyper=pv_config_hyper,
                                                     logger_name=LOGGER_NAME)
    pvae_down = {}
    for R_down in [2,3,4,5,6,7,10,12,14,15,20]:
        params["R_down"]=R_down
        pv_config_2 = get_prosail_vae_config(params, bands = bands, prosail_bands=prosail_bands,
                                        inference_mode = False, rsr_dir=parser.rsr_dir,
                                        norm_mean = norm_mean, norm_std=norm_std)



        prosail_vae = load_prosail_vae_with_hyperprior(pv_config=pv_config_2,
                                                        pv_config_hyper=pv_config_hyper,
                                                        logger_name=LOGGER_NAME)
        pvae_down[str(R_down)] = prosail_vae
    # x, angles = train_loader.dataset[0]
    # x = x.unsqueeze(0)
    # angles = angles.unsqueeze(0)
    # y, angles = prosail_vae_1.encode(x, angles)
    # dist_params = prosail_vae_1.lat_space.get_params_from_encoder(y)
    # # latent mode
    # z = prosail_vae_1.lat_space.mode(dist_params)
    # # transfer to simulator variable
    # sim = prosail_vae_1.transfer_latent(z.unsqueeze(2))

    p_vars, p_s2r = np_simulate_prosail_dataset(nb_simus=1024, noise=0, psimulator=prosail_vae_1.decoder.prosailsimulator,
                                                ssimulator=prosail_vae_1.decoder.ssimulator,
                                                n_samples_per_batch=1024, uniform_mode=False, lai_corr=True)

    # decoding
    sim = torch.from_numpy(p_vars[:,:11]).unsqueeze(2).float()

    angles = torch.from_numpy(p_vars[:,11:]).float()
    p_s2r = torch.from_numpy(p_s2r).float()


    # sim_zero_hspot = sim
    # sim_zero_hspot[:,8,:] = 0
    # rec_1_0_hspot = prosail_vae_1.decode(sim, angles, apply_norm=False).detach()
    # fig, axs = plt.subplots(2,5, dpi=150, tight_layout=True, figsize = (12,6))
    # for i in range(10):
    #     row = i % 2
    #     col = i//2
    #     err = (rec_1[:,i,:] - rec_1_0_hspot[:,i,:]).abs().squeeze()
    #     axs[row, col].boxplot(err, showfliers=False)
    #     # axs[1, col].set_xlabel("Down-sampling")
    #     axs[row, col].set_xticks([])
    #     axs[row, col].ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)
    #     axs[row, col].set_title(BANDS[i])
    # axs[0, 0].set_ylabel('Absolute Error')
    # axs[1, 0].set_ylabel('Absolute Error')
    # fig.savefig("hspot_boxplots_bands_error.png")
    # fig, axs = plt.subplots(dpi=150, tight_layout=True, figsize = (12,6))
    recs_rdown = {}
    soil_spectrums = {}
    lambdas = {}
    n_s2 = {}
    for key, pvae in pvae_down.items():
        R_down = pvae.decoder.prosailsimulator.R_down
        soil_spectrums[key] = pvae.decoder.prosailsimulator.soil_spectrum1
        lambdas[key] = pvae.decoder.prosailsimulator.lambdas
        n_s2[key] = pvae.decoder.ssimulator.s2norm_factor_n
        recs_rdown[key] = pvae.decode(sim, angles, apply_norm=False).detach()
    rec_1 = prosail_vae_1.decode(sim, angles, apply_norm=False).detach()
    soil_spectrum1 = prosail_vae_1.decoder.prosailsimulator.soil_spectrum1
    lambda1 = prosail_vae_1.decoder.prosailsimulator.lambdas
    fig, ax = plt.subplots(dpi=150, tight_layout=True, figsize = (12,6))
    ax.plot(lambda1, soil_spectrum1, label = f"R_down = 1")
    for j, (key, pvae) in enumerate(pvae_down.items()):
        ax.plot(lambdas[key], soil_spectrums[key], label = f"R_down = {pvae.decoder.prosailsimulator.R_down}")
    ax.legend()
    fig, ax = plt.subplots(dpi=150, tight_layout=True, figsize = (12,6))
    ax.plot(lambda1, prosail_vae_1.decoder.ssimulator.s2norm_factor_n[0,0,:], label = f"R_down = 1")
    for j, (key, pvae) in enumerate(pvae_down.items()):
        ax.plot(lambdas[key], n_s2[key][0,0,:], label = f"R_down = {pvae.decoder.prosailsimulator.R_down}")
    ax.legend()
    fig, ax = plt.subplots(dpi=150, tight_layout=True, figsize = (12,6))
    ax.plot(rec_1[0,:,0], label = f"R_down = 1")
    for j, (key, pvae) in enumerate(pvae_down.items()):
        ax.plot(recs_rdown[key][0,:,0], label = f"R_down = {pvae.decoder.prosailsimulator.R_down}")
    ax.legend()

    fig, axs = plt.subplots(2,5, dpi=150, tight_layout=True, figsize = (12,6))
    for i in range(10):
        row = i % 2
        col = i // 2
        for j, (key, rec) in enumerate(recs_rdown.items()):
            err = (rec_1[:,i,:] - rec[:,i,:]).abs().squeeze()
            axs[row, col].boxplot(err,positions=[j], showfliers=False)
        axs[1, col].set_xlabel("Down-sampling")
        axs[row, col].set_xticklabels(recs_rdown.keys())
        axs[row, col].ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)
        axs[row, col].set_title(BANDS[i])
    axs[0, 0].set_ylabel('Absolute Error')
    axs[1, 0].set_ylabel('Absolute Error')
    fig.savefig("Prosail_down_sampling_error_boxplot.png")
    # n_samples = sim.size(2)
    # batch_size = sim.size(0)
    # sim_input = torch.concat((sim, angles.unsqueeze(2).repeat(1,1,n_samples)), 
    #                             axis=1).transpose(1,2).reshape(n_samples*batch_size, -1)
    # prosail_output_1 = prosail_vae_1.decoder.prosailsimulator(sim_input).detach()
    # prosail_output_2 = prosail_vae_2.decoder.prosailsimulator(sim_input).detach()
    
    # fix, ax = plt.subplots(dpi=150)
    # ax.plot(np.arange(400,2501), prosail_output_1[0,:])
    # ax.plot(np.arange(400,2500, 10), prosail_output_2[0,:])
    # fig, ax = plt.subplots(10,1, dpi=150)
    # for idx_band in range(10):
    #     xmin = min(rec_1[:,idx_band, :].min(), rec_2[:,idx_band, :].min())
    #     xmax = max(rec_1[:,idx_band, :].max(), rec_2[:,idx_band, :].max())
    #     ax[idx_band].scatter(rec_1[:,idx_band, :], rec_2[:,idx_band, :], s=1)
    #     ax[idx_band].set_title(idx_band)
    #     ax[idx_band].plot([xmin, xmax], [xmin, xmax], "k--")
    
    # for idx_band in range(10):
    #     fig, ax = plt.subplots(dpi=150)
    #     xmin = min(rec_1[:,idx_band, :].min(), rec_2[:,idx_band, :].min())
    #     xmax = max(rec_1[:,idx_band, :].max(), rec_2[:,idx_band, :].max())
    #     ax.scatter(rec_1[:,idx_band, :], rec_2[:,idx_band, :], s=1)
    #     ax.set_title(idx_band)
    #     ax.plot([xmin, xmax], [xmin, xmax], "k--")
    #     ax.set_xlabel("R_down = 1")
    #     ax.set_ylabel("R_down = 10")
def configureEmissionTracker(parser):
    logger = logging.getLogger(LOGGER_NAME)
    try:
        from codecarbon import OfflineEmissionsTracker
        tracker = OfflineEmissionsTracker(country_iso_code="FRA", output_dir=parser.root_results_dir)
        tracker.start()
        useEmissionTracker = True
    except:
        logger.error("Couldn't start codecarbon ! Emissions not tracked for this execution.")
        useEmissionTracker = False
        tracker = None
    return tracker, useEmissionTracker

def save_array_xp_path(job_array_dir, res_dir):
    if job_array_dir is not None:
        if not os.path.isfile(job_array_dir + "/results_directory_names.txt"):
            with open(job_array_dir + "/results_directory_names.txt", 'w') as outfile:
                outfile.write(f"{res_dir}\n")
        else:
            with open(job_array_dir + "/results_directory_names.txt", 'a') as outfile:
                outfile.write(f"{res_dir}\n")

def main():
    (params, parser, res_dir, data_dir, params_sup_kl_model,
     job_array_dir, sup_norm_mean, sup_norm_std,
     frm4veg_data_dir, frm4veg_barrax_filename, frm4veg_wytham_filename,
     belsar_dir, list_belsar_filenames) = setup_training()
    tracker, useEmissionTracker = configureEmissionTracker(parser)
    spatial_encoder_types = ['cnn', 'rcnn']
    try:
        train_prosailvae(params, parser, res_dir, data_dir, params_sup_kl_model,
                                     sup_norm_mean=sup_norm_mean, sup_norm_std=sup_norm_std)
        if params["k_fold"] > 1:
            save_array_xp_path(os.path.join(res_dir, os.path.pardir), res_dir)
    except Exception as exc:
        traceback.print_exc()
        print(exc)
    if useEmissionTracker:
        tracker.stop()
    pass

if __name__ == "__main__":
    main()
