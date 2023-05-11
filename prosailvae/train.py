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
from ProsailSimus import get_bands_idx
import argparse
import pandas as pd


import socket
import os
import numpy as np
import torch.optim as optim
import torch
from tqdm import trange
from tqdm.contrib.logging import logging_redirect_tqdm

torch.autograd.set_detect_anomaly(True)


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

def initialize_by_training(n_models:int,
                           n_epochs:int,
                           n_samples:int,
                           train_loader,
                           valid_loader,
                           lr:float,
                           logger,
                           pv_config:ProsailVAEConfig,
                           pv_config_hyper:ProsailVAEConfig|None=None,
                           ):
    """
    Initialize prosial_vae by running a few models for several epochs at high lr,
      and selecting the best.
    """
    min_valid_loss = torch.inf
    logger.info(f"Intializing by training {n_models} models for {n_epochs} epochs:")
    for _ in range(n_models):
        prosail_vae = load_prosail_vae_with_hyperprior(pv_config=pv_config,
                                                       pv_config_hyper=pv_config_hyper,
                                                       logger_name=LOGGER_NAME)
        optimizer = optim.Adam(prosail_vae.parameters(), lr=lr, weight_decay=1e-2)
        _, all_valid_loss_df, _ = training_loop(prosail_vae,
                                                optimizer,
                                                n_epochs,
                                                train_loader,
                                                valid_loader,
                                                lrtrainloader=None,
                                                res_dir=None,
                                                n_samples=n_samples,
                                                lr_recompute=None,
                                                exp_lr_decay=-1,
                                                plot_gradient=False,#parser.plot_results,
                                                lr_recompute_mode=False)
        model_min_loss = all_valid_loss_df['loss_sum'].values.min()
        if min_valid_loss > model_min_loss:
            prosail_vae.save_ae(n_epochs, optimizer, model_min_loss, pv_config.vae_save_file_path)
    # best_prosail_vae = load_prosail_vae_with_hyperprior(pv_config=pv_config,
    #                                                     pv_config_hyper=pv_config_hyper,
    #                                                     logger_name=LOGGER_NAME)
    # return best_prosail_vae

def training_loop(prosail_vae, optimizer, n_epoch, train_loader, valid_loader, lrtrainloader,
                  res_dir=None, n_samples=20, lr_recompute=None, exp_lr_decay=0,
                  plot_gradient=False, lr_recompute_mode=True):
    logger = logging.getLogger(LOGGER_NAME)
    if prosail_vae.decoder.loss_type=='mse':
        n_samples=1
        logger.info('MSE Loss enabled, setting number of monte-carlo samples to 1')
    all_train_loss_df = pd.DataFrame()
    all_valid_loss_df = pd.DataFrame()
    info_df = pd.DataFrame()
    best_val_loss = torch.inf
    total_ram = get_total_RAM()
    old_lr = optimizer.param_groups[0]['lr']
    if exp_lr_decay > 0:
        if lr_recompute_mode :
            lr_scheduler =  torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer,
                                                                   gamma=exp_lr_decay)
        else:
            if lr_recompute is not None:
                lr_scheduler =  torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                                           patience=lr_recompute,
                                                                           threshold=0.01)

    max_train_samples_per_epoch = 100
    max_valid_samples_per_epoch = None
    if socket.gethostname()=='CELL200973':
        max_train_samples_per_epoch = 5
        max_valid_samples_per_epoch = 2
    with logging_redirect_tqdm():
        for epoch in trange(n_epoch, desc='PROSAIL-VAE training', leave=True):
            t0=time.time()
            if optimizer.param_groups[0]['lr'] < 5e-8:
                break #stop training if lr too low

            switch_loss(epoch, n_epoch, prosail_vae, swith_ratio=0.75)
            if lr_recompute_mode:
                raise NotImplementedError
                # lr_scheduler, optimizer, old_lr = recompute_lr(lr_scheduler, prosail_vae, epoch,
                #                                                lr_recompute, exp_lr_decay, logger,
                #                                                 optimizer, old_lr=old_lr,
                #                                                 lrtrainloader=lrtrainloader,
                #                                                 weiss_mode=weiss_mode,
                #                                                 n_samples=n_samples)
            info_df = pd.concat([info_df, pd.DataFrame({'epoch':epoch,
                                                        "lr": optimizer.param_groups[0]['lr']}, 
                                                        index=[0])],ignore_index=True)
            try:
                train_loss_dict = prosail_vae.fit(train_loader, optimizer,
                                                  n_samples=n_samples,
                                                  max_samples=max_train_samples_per_epoch)
                if plot_gradient and res_dir is not None:
                    if not os.path.isdir(res_dir + "/gradient_flows"):
                        os.makedirs(res_dir + "/gradient_flows")
                    plot_grad_flow(prosail_vae, 
                                   savefile=res_dir+f"/gradient_flows/grad_flow_{epoch}.svg")
            except Exception as exc:
                logger.error(f"Error during Training at epoch {epoch} !")
                logger.error('Original error :')
                logger.error(str(exc))
                print(f"Error during Training at epoch {epoch} !")
                print('Original error :')
                print(str(exc))
                traceback.print_exc()
                break
            try:
                valid_loss_dict = prosail_vae.validate(valid_loader, n_samples=n_samples,
                                                       max_samples=max_valid_samples_per_epoch)
                if exp_lr_decay>0:
                    if lr_recompute_mode:
                        lr_scheduler.step()
                    else:
                        lr_scheduler.step(valid_loss_dict['loss_sum'])

            except Exception as exc:
                logger.error(f"Error during Validation at epoch {epoch} !")
                logger.error('Original error :')
                logger.error(str(exc))
                print(f"Error during Validation at epoch {epoch} !")
                print('Original error :')
                print(str(exc))
                traceback.print_exc()
            t1=time.time()
            ram_usage = get_RAM_usage()
            train_loss_info = '- '.join([f"{key}: {'{:.2E}'.format(train_loss_dict[key])} " for key in train_loss_dict.keys()])
            valid_loss_info = '- '.join([f"{key}: {'{:.2E}'.format(valid_loss_dict[key])} " for key in valid_loss_dict.keys()])
            logger.info(f"{epoch} -- RAM: {ram_usage} / {total_ram} -- lr: {'{:.2E}'.format(optimizer.param_groups[0]['lr'])} -- {'{:.1f}'.format(t1-t0)} s -- {train_loss_info} -- {valid_loss_info}")
            train_loss_dict['epoch'] = epoch
            valid_loss_dict['epoch'] = epoch
            all_train_loss_df = pd.concat([all_train_loss_df,
                       pd.DataFrame(train_loss_dict, index=[0])],ignore_index=True)
            all_valid_loss_df = pd.concat([all_valid_loss_df, 
                       pd.DataFrame(valid_loss_dict, index=[0])],ignore_index=True)
            if valid_loss_dict['loss_sum'] < best_val_loss:
                best_val_loss = valid_loss_dict['loss_sum']
                if res_dir is not None:
                    prosail_vae.save_ae(epoch, optimizer, best_val_loss,
                                        res_dir + "/prosailvae_weights.tar")
    if n_epoch < 1: # In case we just want to plot results
        all_train_loss_df = pd.DataFrame(data={"loss_sum":10000, "epoch":0}, index=[0])
        all_valid_loss_df = pd.DataFrame(data={"loss_sum":10000, "epoch":0}, index=[0])
        info_df = pd.DataFrame(data={"lr":10000, "epoch":0}, index=[0])
    return all_train_loss_df, all_valid_loss_df, info_df


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
              "-d", "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/patches/",
              "-r", "",
              "-rsr", '/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/',
              "-t", "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/validation_tiles/",
              "-a", "False",
              "-p", "False"]
        parser = get_prosailvae_train_parser().parse_args(args)
    else:
        parser = get_prosailvae_train_parser().parse_args()
    root_dir = TOP_PATH
    xp_array = parser.xp_array
    job_array_dir = None
    if xp_array:
        job_array_dir = os.path.join(parser.root_results_dir, os.pardir)
    config_dir = os.path.join(root_dir,"config/")
    if len(parser.data_dir)==0:
        data_dir = os.path.join(root_dir,"data/")
    else:
        data_dir = parser.data_dir
    assert parser.n_fold < parser.n_xp
    params = load_params(config_dir, config_file=parser.config_file, parser=parser)
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
    return params, parser, res_dir, data_dir, params_sup_kl_model, job_array_dir, sup_norm_mean, sup_norm_std

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
        train_loader, valid_loader = get_simloader(valid_ratio=params["valid_ratio"],
                                                    file_prefix=params["dataset_file_prefix"],
                                                    sample_ids=None,
                                                    batch_size=params["batch_size"],
                                                    data_dir=data_dir)
    else:
        train_loader, valid_loader, _ = get_train_valid_test_loader_from_patches(data_dir, batch_size=1, 
                                                                                 num_workers=0)

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
    pv_config = get_prosail_vae_config(params, bands = bands, prosail_bands=prosail_bands,
                                       inference_mode = False, rsr_dir=parser.rsr_dir,
                                       norm_mean = norm_mean, norm_std=norm_std)
    pv_config_hyper=None
    if params_sup_kl_model is not None:
        bands_hyper, prosail_bands_hyper = get_bands_idx(params_sup_kl_model["weiss_bands"])
        pv_config_hyper = get_prosail_vae_config(params_sup_kl_model, bands=bands_hyper,
                                                 prosail_bands=prosail_bands_hyper,
                                                 inference_mode=True, rsr_dir=parser.rsr_dir,
                                                 norm_mean=sup_norm_mean, norm_std=sup_norm_std)
    if params['init_model']:
        n_models=10
        lr = 1e-4
        n_epochs=5
        if socket.gethostname()=='CELL200973':
            n_epochs = 1
            n_models = 2

        initialize_by_training(n_models=n_models,
                               n_epochs=n_epochs,
                                train_loader=train_loader,
                                valid_loader=valid_loader,
                                lr=lr,
                                logger=logger,
                                n_samples=training_config.n_samples,
                                pv_config=pv_config,
                                pv_config_hyper=pv_config_hyper
                            )
        # Changing config to load the best model intialized
        params["load_model"] = True
        params["vae_load_file_path"] = params["vae_save_file_path"]
        pv_config = get_prosail_vae_config(params, bands=bands, prosail_bands=prosail_bands,
                                            inference_mode=False, rsr_dir=parser.rsr_dir,
                                            norm_mean=norm_mean, norm_std=norm_std)

    prosail_vae = load_prosail_vae_with_hyperprior(pv_config=pv_config,
                                                    pv_config_hyper=pv_config_hyper,
                                                    logger_name=LOGGER_NAME)
    lr = params['lr']
    lrtrainloader = None
    tensor_dir=None
    if not params["simulated_dataset"]:
        tensor_dir = parser.tensor_dir
    lrtrainloader = lr_finder_loader(
                    file_prefix=params["dataset_file_prefix"],
                    sample_ids=None,
                    batch_size=64,
                    data_dir=data_dir,
                    supervised=prosail_vae.supervised,
                    tensors_dir=tensor_dir)
    lr_recompute_mode = params["lr_recompute_mode"]
    if lr is None:
        try:
            # raise NotImplementedError
            lr = get_PROSAIL_VAE_lr(prosail_vae, lrtrainloader, n_samples=params["n_samples"], 
                                    disable_tqdm=not socket.gethostname()=='CELL200973')
            logger.info('LR computed ! using lr = {:.2E}'.format(lr))
        except Exception as exc:
            traceback.print_exc()
            print(exc)
            lr = 1e-4
            logger.error(f"Couldn't recompute lr at initialization ! Using lr={lr}")
            logger.error(f"{exc}")
            print(f"Couldn't recompute lr at initialization ! Using lr={lr}")

    optimizer = optim.Adam(prosail_vae.parameters(), lr=lr, weight_decay=1e-2)
    logger.info('PROSAIL-VAE and optimizer initialized.')
    
    # Training
    logger.info(f"Starting Training loop for {params['epochs']} epochs.")

    all_train_loss_df, all_valid_loss_df, info_df = training_loop(prosail_vae,
                                                         optimizer,
                                                         params['epochs'],
                                                         train_loader,
                                                         valid_loader,
                                                         lrtrainloader,
                                                         res_dir=res_dir,
                                                         n_samples=params["n_samples"],
                                                         lr_recompute=params['lr_recompute'],
                                                         exp_lr_decay=params["exp_lr_decay"],
                                                         plot_gradient=False,#parser.plot_results,
                                                         lr_recompute_mode=lr_recompute_mode)
    logger.info("Training Completed !")

    return prosail_vae, all_train_loss_df, all_valid_loss_df, info_df

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
     job_array_dir, sup_norm_mean, sup_norm_std) = setup_training()
    tracker, useEmissionTracker = configureEmissionTracker(parser)
    spatial_encoder_types = ['cnn', 'rcnn']

    try:
        (prosail_vae, all_train_loss_df, all_valid_loss_df,
         info_df) = train_prosailvae(params, parser, res_dir, data_dir, params_sup_kl_model,
                                     sup_norm_mean=sup_norm_mean, sup_norm_std=sup_norm_std)
        if params['encoder_type'] in spatial_encoder_types:
            _, _, test_loader = get_train_valid_test_loader_from_patches(data_dir, bands = torch.arange(10),
                                                                            batch_size=1, num_workers=0)
            info_test_data = np.load(os.path.join(data_dir,"test_info.npy"))
            save_results_2d(prosail_vae, test_loader, res_dir, 
                            all_train_loss_df, all_valid_loss_df, info_df, LOGGER_NAME=LOGGER_NAME, 
                            plot_results=parser.plot_results, info_test_data=info_test_data)
        else:
            save_results(prosail_vae, res_dir, data_dir, all_train_loss_df,
                         all_valid_loss_df, info_df, LOGGER_NAME=LOGGER_NAME,
                         plot_results=parser.plot_results, weiss_mode=parser.weiss_mode, n_samples=params["n_samples"])
        save_array_xp_path(job_array_dir, res_dir)
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
