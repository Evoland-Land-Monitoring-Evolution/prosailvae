#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 14:20:44 2022

@author: yoel
"""
from prosailvae.prosail_vae import load_PROSAIL_VAE_with_supervised_kl
from prosailvae.torch_lr_finder import get_PROSAIL_VAE_lr
from dataset.loaders import  get_simloader, get_norm_coefs, get_mmdc_loaders
# from prosailvae.ProsailSimus import get_ProsailVarsIntervalLen
from metrics.results import save_results, get_res_dir_path

import torch.optim as optim
import torch
import pandas as pd
import argparse
import os 

import prosailvae
from tqdm import trange
from tqdm.contrib.logging import logging_redirect_tqdm
import shutil


from prosailvae.ProsailSimus import PROSAILVARS, BANDS
from prosailvae.utils import load_dict, save_dict, get_RAM_usage, get_total_RAM
torch.autograd.set_detect_anomaly(True)
import logging
import logging.config
import time
import traceback
import socket
CUDA_LAUNCH_BLOCKING=1
LOGGER_NAME = 'PROSAIL-VAE logger'


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

    parser.add_argument("-cs", dest="supervised_config_file",
                        help="path to config for supervised model kl json file on config directory.",
                        type=str, default="config.json")

    parser.add_argument("-ws", dest="supervised_weight_file",
                        help="path to model weights used to supervise the KL loss",
                        type=str, default="model.tar")
    
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
                        type=str, default="/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/real_data/torchfiles/")
    parser.add_argument("-a", dest="xp_array",
                        help="array training (false for single xp) ",
                        type=int, default=False)
    return parser

def recompute_lr(lr_scheduler, PROSAIL_VAE, epoch, lr_recompute, exp_lr_decay, logger, data_dir, optimizer):
    if epoch > 0 and lr_recompute is not None:
        if epoch % lr_recompute == 0:
            try:
                new_lr = get_PROSAIL_VAE_lr(PROSAIL_VAE, data_dir=data_dir)
                optimizer = optim.Adam(PROSAIL_VAE.parameters(), lr=new_lr, weight_decay=1e-2)
                if exp_lr_decay>0:
                    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.2, 
                    #                                         patience=exp_lr_decay, threshold=0.0001, 
                    #                                         threshold_mode='rel', cooldown=0, min_lr=1e-8, 
                    #                                         eps=1e-08, verbose=False)
                    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=exp_lr_decay)
            except:
                logger.error(f"Couldn't recompute lr at epoch {epoch} !")
                print(f"Couldn't recompute lr at epoch {epoch} !")
    return lr_scheduler, optimizer

def switch_loss(epoch, n_epoch, PROSAIL_VAE, swith_ratio = 0.75):
    loss_type = PROSAIL_VAE.decoder.loss_type
    if loss_type == "hybrid_nll":
        if epoch > swith_ratio * n_epoch:
            PROSAIL_VAE.decoder.loss_type = "full_nll"
    pass

def training_loop(PROSAIL_VAE, optimizer, n_epoch, train_loader, valid_loader, 
                  res_dir, n_samples=20, lr_recompute=None, data_dir="", exp_lr_decay=0):


    logger = logging.getLogger(LOGGER_NAME)
    if PROSAIL_VAE.decoder.loss_type=='mse':
        n_samples=1
        logger.info('MSE Loss enabled, setting number of monte-carlo samples to 1')
    all_train_loss_df = pd.DataFrame()
    all_valid_loss_df = pd.DataFrame()
    info_df = pd.DataFrame()
    best_val_loss = torch.inf
    total_ram = get_total_RAM()
    if exp_lr_decay > 0:
        # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.2, 
        #                                                             patience=exp_lr_decay, threshold=0.0001, 
        #                                                             threshold_mode='rel', cooldown=0, min_lr=1e-8, 
        #                                                             eps=1e-08, verbose=False)
        lr_scheduler =  torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=exp_lr_decay)
    with logging_redirect_tqdm():
        for epoch in trange(n_epoch, desc='PROSAIL-VAE training', leave=True):
            t0=time.time()
            switch_loss(epoch, n_epoch, PROSAIL_VAE, swith_ratio=0.75)
            lr_scheduler, optimizer = recompute_lr(lr_scheduler, PROSAIL_VAE, epoch, lr_recompute, exp_lr_decay, logger, data_dir, optimizer)
            info_df = pd.concat([info_df, pd.DataFrame({'epoch':epoch, "lr": optimizer.param_groups[0]['lr']}, index=[0])],ignore_index=True)
            try:
                train_loss_dict = PROSAIL_VAE.fit(train_loader, optimizer, n_samples=n_samples)
            except Exception as e:
                logger.error(f"Error during Training at epoch {epoch} !")
                logger.error('Original error :')
                logger.error(str(e))
                print(f"Error during Training at epoch {epoch} !")
                print('Original error :')
                print(str(e))
                traceback.print_exc()
                break
            try:
                valid_loss_dict = PROSAIL_VAE.validate(valid_loader, n_samples=n_samples)
                if exp_lr_decay>0:
                    # lr_scheduler.step(valid_loss_dict['loss_sum'])
                    lr_scheduler.step()
            except Exception as e:
                logger.error(f"Error during Training at epoch {epoch} !")
                logger.error('Original error :')
                logger.error(str(e))
                print(f"Error during Validation at epoch {epoch} !")
                print('Original error :')
                print(str(e))
                traceback.print_exc()
            
            t1=time.time()
            ram_usage = get_RAM_usage()
            
            train_loss_info = '- '.join([f"{key}: {'{:.2E}'.format(train_loss_dict[key])} " for key in train_loss_dict.keys()])
            valid_loss_info = '- '.join([f"{key}: {'{:.2E}'.format(valid_loss_dict[key])} " for key in valid_loss_dict.keys()])
            logger.info(f"{epoch} -- RAM: {ram_usage} / {total_ram} -- lr: {'{:.2E}'.format(optimizer.param_groups[0]['lr'])} -- {'{:.1f}'.format(t1-t0)} s -- {train_loss_info} -- {valid_loss_info}")
            train_loss_dict['epoch']=epoch
            valid_loss_dict['epoch']=epoch
            all_train_loss_df = pd.concat([all_train_loss_df, 
                       pd.DataFrame(train_loss_dict, index=[0])],ignore_index=True)
            all_valid_loss_df = pd.concat([all_valid_loss_df, 
                       pd.DataFrame(valid_loss_dict, index=[0])],ignore_index=True)
            
            if valid_loss_dict['loss_sum'] < best_val_loss:
                best_val_loss = valid_loss_dict['loss_sum'] 
                PROSAIL_VAE.save_ae(epoch, optimizer, best_val_loss, res_dir + "/prosailvae_weights.tar")
    return all_train_loss_df, all_valid_loss_df, info_df



def setupTraining():
    if socket.gethostname()=='CELL200973':
        args=["-n", "0",
              "-c", "config_dev.json",
              "-cs", "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/results/sup_kl_model_config.json",
              "-ws", "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/results/sup_kl_model_weights.tar",
              "-x", "1",
              "-o", "True",
              "-d", "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/",
              "-r", "",
              "-rsr", '/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/',
              "-t", "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/real_data/torchfiles/",
              "-a", "False"]
        
        parser = get_prosailvae_train_parser().parse_args(args)    
    else:
        parser = get_prosailvae_train_parser().parse_args()
    root_dir = os.path.join(os.path.dirname(prosailvae.__file__), os.pardir)
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
    params = load_dict(config_dir + parser.config_file)
    if params["supervised"]:
        params["simulated_dataset"]=True


    params["n_fold"] = parser.n_fold if params["k_fold"] > 1 else None
    if len(parser.root_results_dir)==0:
        root_results_dir = os.path.join(os.path.join(os.path.dirname(prosailvae.__file__),
                                                     os.pardir),"results/")
    else:
        root_results_dir = parser.root_results_dir
    res_dir = get_res_dir_path(root_results_dir, params, 
                               parser.n_xp, parser.overwrite_xp)
    save_dict(params, res_dir+"/config.json")

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
        logger.info("Supervised KL loss enabled.")
        logger.info(f"copying {parser.supervised_config_file} into {res_dir+'/sup_kl_model_config.json'}")
        logger.info(f"copying {parser.supervised_weight_file} into {res_dir+'/sup_kl_model_weights.tar'}")
        shutil.copyfile(parser.supervised_config_file, res_dir+"/sup_kl_model_config.json")
        shutil.copyfile(parser.supervised_weight_file, res_dir+"/sup_kl_model_weights.tar")
        params_sup_kl_model = load_dict(res_dir+"/sup_kl_model_config.json")
        params_sup_kl_model['sup_model_weights_path'] = res_dir+"/sup_kl_model_weights.tar"
    else:
        params_sup_kl_model = None
    return params, parser, res_dir, data_dir, params_sup_kl_model, job_array_dir

def trainProsailVae(params, parser, res_dir, data_dir, params_sup_kl_model=None):
    logger = logging.getLogger(LOGGER_NAME)
    logger.info(f'Loading training and validation loader in {data_dir}/{params["dataset_file_prefix"]}...')
    if params["simulated_dataset"]:
        train_loader, valid_loader = get_simloader(valid_ratio=params["valid_ratio"], 
                            file_prefix=params["dataset_file_prefix"], 
                            sample_ids=None,
                            batch_size=params["batch_size"],
                            data_dir=data_dir)
    else:
        train_loader, valid_loader, _ = get_mmdc_loaders(tensors_dir=parser.tensor_dir,
                                                         batch_size=1,
                                                         max_open_files=4,
                                                         num_workers=1,
                                                         pin_memory=False)
    
    logger.info(f'Training ({len(train_loader.dataset)} samples) '
                f'and validation ({len(valid_loader.dataset)} samples) loaders, loaded.')
    
    

    PROSAIL_VAE = load_PROSAIL_VAE_with_supervised_kl(params, parser, data_dir, logger_name=LOGGER_NAME,
                                                        vae_file_path=None, params_sup_kl_model=params_sup_kl_model)
    lr = params['lr']
    if lr is None:
        try:
            lr = get_PROSAIL_VAE_lr(PROSAIL_VAE, data_dir=data_dir,n_samples=params["n_samples"])
        except:
            lr = 1e-3
            logger.error(f"Couldn't recompute lr at initialization ! Using lr={lr}")
            print(f"Couldn't recompute lr at initialization ! Using lr={lr}")

    optimizer = optim.Adam(PROSAIL_VAE.parameters(), lr=lr, weight_decay=1e-2)
    # PROSAIL_VAE.load_ae("/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/results/" + "/prosailvae_weigths.tar", optimizer=optimizer)
    # all_train_loss_df = pd.DataFrame([])
    # all_valid_loss_df = pd.DataFrame([])
    # info_df = pd.DataFrame([])
    logger.info('PROSAIL-VAE and optimizer initialized.')
    
    # Trainingget_PROSAIL_VAE_lr
    logger.info(f"Starting Training loop for {params['epochs']} epochs.")

    all_train_loss_df, all_valid_loss_df, info_df = training_loop(PROSAIL_VAE, 
                                                         optimizer, 
                                                         params['epochs'],
                                                         train_loader, 
                                                         valid_loader,
                                                         res_dir=res_dir,
                                                         n_samples=params["n_samples"],
                                                         lr_recompute=params['lr_recompute'],
                                                         data_dir=data_dir, 
                                                         exp_lr_decay=params["exp_lr_decay"]) 
    logger.info("Training Completed !")

    return PROSAIL_VAE, all_train_loss_df, all_valid_loss_df, info_df

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
    params, parser, res_dir, data_dir, params_sup_kl_model, job_array_dir = setupTraining()
    tracker, useEmissionTracker = configureEmissionTracker(parser)
    try:
        PROSAIL_VAE, all_train_loss_df, all_valid_loss_df, info_df = trainProsailVae(params, parser, res_dir, data_dir, params_sup_kl_model)
        save_results(PROSAIL_VAE, res_dir, data_dir, all_train_loss_df, all_valid_loss_df, info_df, LOGGER_NAME=LOGGER_NAME)
        save_array_xp_path(job_array_dir, res_dir)
    except Exception as e:
        traceback.print_exc()
        print(e)
    if useEmissionTracker:
        tracker.stop()
    pass

if __name__ == "__main__":
    main()
    
    