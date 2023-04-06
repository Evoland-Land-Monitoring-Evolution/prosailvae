#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 14:20:44 2022

@author: yoel
"""
from prosailvae.prosail_vae import load_PROSAIL_VAE_with_supervised_kl
from prosailvae.torch_lr_finder import get_PROSAIL_VAE_lr
from dataset.loaders import  get_simloader, get_norm_coefs, get_mmdc_loaders, get_loaders_from_image, lr_finder_loader, get_bands_norm_factors_from_loaders, get_train_valid_test_loader_from_patches
# from prosailvae.ProsailSimus import get_ProsailVarsIntervalLen
from metrics.results import save_results, save_results_2d, get_res_dir_path

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
from prosailvae.utils import load_dict, save_dict, get_RAM_usage, get_total_RAM, plot_grad_flow
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
                        type=bool, default=False)

    parser.add_argument("-p", dest="plot_results",
                        help="toggle results plotting",
                        type=bool, default=False)
    parser.add_argument("-w", dest="weiss_mode",
                        help="removes B2 and B8 bands for validation with weiss data",
                        type=bool, default=False)             
    return parser

def recompute_lr(lr_scheduler, PROSAIL_VAE, epoch, lr_recompute, exp_lr_decay, logger, optimizer, lrtrainloader, 
                 old_lr=1.0, weiss_mode=False, n_samples=1):
    new_lr=old_lr
    if epoch > 0 and lr_recompute is not None:
        if epoch % lr_recompute == 0:
            try:
                new_lr = get_PROSAIL_VAE_lr(PROSAIL_VAE, lrtrainloader, old_lr=old_lr, old_lr_max_ratio=10, n_samples=n_samples)
                optimizer = optim.Adam(PROSAIL_VAE.parameters(), lr=new_lr, weight_decay=1e-2)
                if exp_lr_decay>0:
                    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=exp_lr_decay)
            except Exception as e:
                traceback.print_exc()
                print(e)
                logger.error(f"Couldn't recompute lr at epoch {epoch} !")
                logger.error(f"{e}")
                print(f"Couldn't recompute lr at epoch {epoch} !")
    return lr_scheduler, optimizer, new_lr

def switch_loss(epoch, n_epoch, PROSAIL_VAE, swith_ratio = 0.75):
    loss_type = PROSAIL_VAE.decoder.loss_type
    if loss_type == "hybrid_nll":
        if epoch > swith_ratio * n_epoch:
            PROSAIL_VAE.decoder.loss_type = "full_nll"
    pass

def training_loop(PROSAIL_VAE, optimizer, n_epoch, train_loader, valid_loader, lrtrainloader,
                  res_dir, n_samples=20, lr_recompute=None, exp_lr_decay=0, 
                  plot_gradient=False, mmdc_dataset=False, weiss_mode=False, lr_recompute_mode=True):


    logger = logging.getLogger(LOGGER_NAME)
    if PROSAIL_VAE.decoder.loss_type=='mse':
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
            lr_scheduler =  torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=exp_lr_decay)
        else:
            if lr_recompute is None:
                lr_recompute = 20
            lr_scheduler =  torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=lr_recompute,threshold=0.01)
    
    
    max_train_samples_per_epoch = 100
    max_valid_samples_per_epoch = None
    if socket.gethostname()=='CELL200973':
        max_train_samples_per_epoch = 5
        max_valid_samples_per_epoch = 2
    with logging_redirect_tqdm():
        for epoch in trange(n_epoch, desc='PROSAIL-VAE training', leave=True):
            t0=time.time()
            switch_loss(epoch, n_epoch, PROSAIL_VAE, swith_ratio=0.75)
            if lr_recompute_mode:
                lr_scheduler, optimizer, old_lr = recompute_lr(lr_scheduler, PROSAIL_VAE, epoch, lr_recompute, exp_lr_decay, logger, 
                                                            optimizer, old_lr=old_lr, lrtrainloader=lrtrainloader, weiss_mode=weiss_mode,
                                                            n_samples=n_samples)
            info_df = pd.concat([info_df, pd.DataFrame({'epoch':epoch, "lr": optimizer.param_groups[0]['lr']}, index=[0])],ignore_index=True)
            try:
                
                train_loss_dict = PROSAIL_VAE.fit(train_loader, optimizer, n_samples=n_samples, mmdc_dataset=mmdc_dataset, 
                                                  max_samples=max_train_samples_per_epoch)
                if plot_gradient:
                    if not os.path.isdir(res_dir + "/gradient_flows"):
                        os.makedirs(res_dir + "/gradient_flows")
                    plot_grad_flow(PROSAIL_VAE, savefile=res_dir+f"/gradient_flows/grad_flow_{epoch}.svg")
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
                valid_loss_dict = PROSAIL_VAE.validate(valid_loader, n_samples=n_samples, mmdc_dataset=mmdc_dataset, 
                                                       max_samples=max_valid_samples_per_epoch)
                if exp_lr_decay>0:
                    
                    if lr_recompute_mode:
                        lr_scheduler.step()
                    else:
                        lr_scheduler.step(valid_loss_dict['loss_sum'])
            except Exception as e:
                logger.error(f"Error during Validation at epoch {epoch} !")
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
              "-d", "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/patches/",
              "-r", "",
              "-rsr", '/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/',
              "-t", "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/validation_tiles/",
              "-a", "False",
              "-p", "False",
              "-w", ""]
        
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
    if not "load_model" in params.keys():
        params["load_model"]=None
    params["k_fold"] = parser.n_xp
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
    bands = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12]
    if parser.weiss_mode:
        bands = [2, 3, 4, 5, 6, 8, 11, 12]
    if params["simulated_dataset"]:
        train_loader, valid_loader = get_simloader(valid_ratio=params["valid_ratio"], 
                            file_prefix=params["dataset_file_prefix"], 
                            sample_ids=None,
                            batch_size=params["batch_size"],
                            data_dir=data_dir)
        
    else:

        # train_loader, valid_loader, _ = get_mmdc_loaders(tensors_dir=parser.tensor_dir,
        #                                                  batch_size=1,
        #                                                  max_open_files=4,
        #                                                  num_workers=1,
        #                                                  pin_memory=False)
        # path_to_image = parser.tensor_dir + "/after_SENTINEL2B_20171127-105827-648_L2A_T31TCJ_C_V2-2_roi_0.pth"
        # n_patches_max = 100
        if socket.gethostname()=='CELL200973':
            n_patches_max = 10
        # bands_image = torch.tensor([0,1,2,3,4,5,6,7,8,9])
        bands_image = torch.tensor([0,1,2,4,5,6,3,7,8,9])
        if parser.weiss_mode:
            bands_image = torch.tensor([1,2,3,4,5,7,8,9])
            raise NotImplementedError
        train_loader, valid_loader, _ = get_train_valid_test_loader_from_patches(data_dir, bands = torch.arange(10), 
                                                                                 batch_size=1, num_workers=0)
        # train_loader, valid_loader, _ = get_loaders_from_image(path_to_image, patch_size=32, train_ratio=0.8, valid_ratio=0.1, 
        #                                                         bands=bands_image, n_patches_max = n_patches_max, 
        #                                                         batch_size=1, num_workers=0)
    if params["apply_norm_rec"]:
        # norm_mean, norm_std = get_bands_norm_factors_from_loaders(train_loader, bands_dim=1, max_samples=1000000, n_bands=len(bands))
        norm_mean = torch.load(os.path.join(data_dir, "norm_mean.pt"))
        norm_std = torch.load(os.path.join(data_dir, "norm_std.pt"))
    
    else:
        norm_mean = torch.zeros(1, len(bands))
        norm_std = torch.ones(1, len(bands))
    torch.save(norm_mean, res_dir + "/norm_mean.pt")
    torch.save(norm_std, res_dir + "/norm_std.pt")
    if params_sup_kl_model is not None:
        raise NotImplementedError
    logger.info(f'Training ({len(train_loader.dataset)} samples) '
                f'and validation ({len(valid_loader.dataset)} samples) loaders, loaded.')
    
    
    print(f"Weiss mode : {parser.weiss_mode}")
    PROSAIL_VAE = load_PROSAIL_VAE_with_supervised_kl(params, parser.rsr_dir, logger_name=LOGGER_NAME,
                                                        vae_file_path=None, params_sup_kl_model=params_sup_kl_model, 
                                                        bands=bands, norm_mean=norm_mean, norm_std=norm_std)
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
                    supervised=PROSAIL_VAE.supervised,
                    tensors_dir=tensor_dir)
    lr_recompute_mode=False
    if lr is None:
        try:
            # raise NotImplementedError
            lr = get_PROSAIL_VAE_lr(PROSAIL_VAE, lrtrainloader, n_samples=params["n_samples"], 
                                    disable_tqdm=not socket.gethostname()=='CELL200973')
            logger.info('LR computed ! using lr = {:.2E}'.format(lr))
        except Exception as e:
            traceback.print_exc()
            print(e)
            lr = 1e-4
            logger.error(f"Couldn't recompute lr at initialization ! Using lr={lr}")
            logger.error(f"{e}")
            print(f"Couldn't recompute lr at initialization ! Using lr={lr}")

    optimizer = optim.Adam(PROSAIL_VAE.parameters(), lr=lr, weight_decay=1e-2)
    # PROSAIL_VAE.load_ae("/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/results/" + "/prosailvae_weigths.tar", optimizer=optimizer)
    if not socket.gethostname()=='CELL200973' and params["load_model"] is not None:
        #"/home/uz/zerahy/scratch/prosailvae/results/cnn_39950033_jobarray/1_d2023_03_31_05_24_16_supervised_False_weiss_/prosailvae_weights.tar"
        vae_path = params["load_model"]
        original_device = PROSAIL_VAE.device
        PROSAIL_VAE.change_device('cpu')
        PROSAIL_VAE.load_ae(vae_path, optimizer=None)
        PROSAIL_VAE.change_device(original_device)
        print(f"loading VAE {vae_path}") 
        logger.info(f"loading VAE {vae_path}")
    # all_train_loss_df = pd.DataFrame([])
    # all_valid_loss_df = pd.DataFrame([])
    # info_df = pd.DataFrame([])
    logger.info('PROSAIL-VAE and optimizer initialized.')
    
    # Training
    logger.info(f"Starting Training loop for {params['epochs']} epochs.")

    all_train_loss_df, all_valid_loss_df, info_df = training_loop(PROSAIL_VAE, 
                                                         optimizer, 
                                                         params['epochs'],
                                                         train_loader, 
                                                         valid_loader,
                                                         lrtrainloader,
                                                         res_dir=res_dir,
                                                         n_samples=params["n_samples"],
                                                         lr_recompute=params['lr_recompute'],
                                                         exp_lr_decay=params["exp_lr_decay"],
                                                         plot_gradient=parser.plot_results,
                                                         mmdc_dataset = not params["simulated_dataset"],
                                                         weiss_mode=parser.weiss_mode, 
                                                         lr_recompute_mode=lr_recompute_mode) 
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
    spatial_encoder_types = ['cnn', 'rcnn']
    try:
        PROSAIL_VAE, all_train_loss_df, all_valid_loss_df, info_df = trainProsailVae(params, parser, res_dir, data_dir, params_sup_kl_model)
        if params['encoder_type'] in spatial_encoder_types:
            # _,_, test_loader = get_mmdc_loaders(tensors_dir=parser.tensor_dir,
            #                                              batch_size=1,
            #                                              max_open_files=4,
            #                                              num_workers=1,
            #                                              pin_memory=False)
            # path_to_image = parser.tensor_dir + "/after_SENTINEL2B_20171127-105827-648_L2A_T31TCJ_C_V2-2_roi_0.pth"

            _, _, test_loader = get_train_valid_test_loader_from_patches(data_dir, bands = torch.tensor([0,1,2,4,5,6,3,7,8,9]),
                                                                            batch_size=1, num_workers=0)
            # _, _, test_loader = get_loaders_from_image(path_to_image, patch_size=32, train_ratio=0.8, valid_ratio=0.1, 
            #                 bands = torch.tensor([0,1,2,4,5,6,3,7,8,9]), n_patches_max = 100, 
            #                 batch_size=1, num_workers=0)
            save_results_2d(PROSAIL_VAE, test_loader, res_dir, parser.tensor_dir, all_train_loss_df, all_valid_loss_df, info_df, LOGGER_NAME=LOGGER_NAME, plot_results=parser.plot_results)
        else:
            save_results(PROSAIL_VAE, res_dir, data_dir, all_train_loss_df, all_valid_loss_df, info_df, LOGGER_NAME=LOGGER_NAME, plot_results=parser.plot_results, weiss_mode=parser.weiss_mode)
        save_array_xp_path(job_array_dir, res_dir)
        if params["k_fold"] > 1:
            save_array_xp_path(os.path.join(res_dir,os.path.pardir), res_dir)
    except Exception as e:
        traceback.print_exc()
        print(e)
    if useEmissionTracker:
        tracker.stop()
    pass

if __name__ == "__main__":
    main()
    
    