#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 14:20:44 2022

@author: yoel
"""
from prosailvae.prosail_vae import get_prosail_VAE
from dataset.loaders import  get_simloader, get_norm_coefs
# from prosailvae.ProsailSimus import get_ProsailVarsIntervalLen
from metrics.metrics import get_metrics, save_metrics
from metrics.prosail_plots import plot_metrics, plot_rec_and_latent, loss_curve, plot_param_dist, plot_pred_vs_tgt, plot_refl_dist, pair_plot
from datetime import datetime 
import torch.optim as optim
import torch
import pandas as pd
import argparse
import os 
import shutil
import prosailvae
from tqdm import trange
import warnings
from time import sleep
from prosailvae.ProsailSimus import PROSAILVARS, BANDS
from prosailvae.utils import load_dict, save_dict


def check_fold_res_dir(fold_dir, n_xp, params):
    same_fold = ""
    all_dirs = os.listdir(fold_dir)
    for d in all_dirs:
        if d.startswith(f"{n_xp}_kfold_{params['k_fold']}_n_{params['n_fold']}") :
            same_fold = d
    return same_fold

def get_res_dir_path(root_results_dir, params, n_xp=None, overwrite_xp=False):
    
    if not os.path.exists(root_results_dir):
        os.makedirs(root_results_dir)
    if not os.path.exists(root_results_dir+"n_xp.json"):    
        save_dict({"xp":0}, root_results_dir+"n_xp.json")
    if n_xp is None:
        n_xp = load_dict(root_results_dir+"n_xp.json")['xp']+1
    save_dict({"xp":n_xp}, root_results_dir+"n_xp.json")
    date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    if params['k_fold']>1:
        k_fold_dir = f"{root_results_dir}/{n_xp}_kfold_{params['k_fold']}_supervised_{params['supervised']}_{params['dataset_file_prefix']}"
        if not params['supervised']:
            k_fold_dir + f"kl_{params['beta_kl']}"
        if not os.path.exists(k_fold_dir):
            os.makedirs(k_fold_dir)    
        res_dir = f"{k_fold_dir}/{n_xp}_kfold_{params['k_fold']}_n_{params['n_fold']}_d{date}_supervised_{params['supervised']}_{params['dataset_file_prefix']}"
        same_fold_dir = check_fold_res_dir(k_fold_dir, n_xp, params)
        if len(same_fold_dir)>0:
            if overwrite_xp:
                warnings.warn("WARNING: Overwriting existing fold experiment in 5s")
                sleep(5)
                shutil.rmtree(k_fold_dir + "/"+ same_fold_dir)
            else:
                raise ValueError(f"The same experiment (fold) has already been carried out at {same_fold_dir}.\n Please change the number of fold or allow overwrite")
    else:
        res_dir = f"{root_results_dir}/{n_xp}_d{date}_supervised_{params['supervised']}_{params['dataset_file_prefix']}"
    os.makedirs(res_dir)    
    return res_dir



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
       
    return parser

def training_loop(phenoVAE, optimizer, n_epoch, train_loader, valid_loader, res_dir, n_samples=20):
    all_train_loss_df = pd.DataFrame()
    all_valid_loss_df = pd.DataFrame()
    best_val_loss = torch.inf
    for epoch in trange(n_epoch, desc='phenoVAE training', leave=True):
        try:
            train_loss_dict = phenoVAE.fit(train_loader, optimizer, n_samples=n_samples)
        except:
            print(f"Error during Training at epoch {epoch} !")
            break
        try:
            valid_loss_dict = phenoVAE.validate(train_loader, n_samples=n_samples)
        except:
            print(f"Error during Validation at epoch {epoch} !")
        train_loss_dict['epoch']=epoch
        valid_loss_dict['epoch']=epoch
        all_train_loss_df = pd.concat([all_train_loss_df, 
                   pd.DataFrame(train_loss_dict, index=[0])],ignore_index=True)
        all_valid_loss_df = pd.concat([all_valid_loss_df, 
                   pd.DataFrame(valid_loss_dict, index=[0])],ignore_index=True)
        if valid_loss_dict['loss_sum'] < best_val_loss:
            best_val_loss = valid_loss_dict['loss_sum'] 
            phenoVAE.save_ae(epoch, optimizer, best_val_loss, res_dir + "/prosailvae_weigths.tar")
    return all_train_loss_df, all_valid_loss_df


if __name__ == "__main__":
    parser = get_prosailvae_train_parser().parse_args()
    root_dir = os.path.join(os.path.dirname(prosailvae.__file__),os.pardir)
    
    config_dir = os.path.join(root_dir,"config/")
    results_dir = os.path.join(root_dir,"results/")
    if len(parser.data_dir)==0:
        data_dir = os.path.join(root_dir,"data/")
    else:
        data_dir = parser.data_dir

    assert parser.n_fold < parser.n_xp 
    params = load_dict(config_dir + parser.config_file)
    if params["supervised"]:
        params["dataset_file_prefix"]='sim_'

    params["n_fold"] = parser.n_fold if params["k_fold"] > 1 else None
    
    # load_train_valid_ids(k=params["k_fold"],
    #                   n=params["n_fold"], 
    #                   file_prefix=params["dataset_file_prefix"])
    
    train_loader, valid_loader = get_simloader(valid_ratio=params["valid_ratio"], 
                                              file_prefix=params["dataset_file_prefix"], 
                                              sample_ids=None,
                                              batch_size=params["batch_size"],
                                                data_dir=data_dir)
    if len(parser.root_results_dir)==0:
        root_results_dir = os.path.join(os.path.join(os.path.dirname(prosailvae.__file__),os.pardir),"results/")
    else:
        root_results_dir = parser.root_results_dir
    res_dir = get_res_dir_path(root_results_dir, params, 
                               parser.n_xp, parser.overwrite_xp)
    save_dict(params, res_dir+"/config.json")

    vae_params={"input_size":10,  
                "hidden_layers_size":params["hidden_layers_size"], 
                "encoder_last_activation":params["encoder_last_activation"],
                "supervised":params["supervised"],  
                "beta_kl":params["beta_kl"]}
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    norm_mean, norm_std = get_norm_coefs(data_dir, params["dataset_file_prefix"])
    rsr_dir = parser.rsr_dir
    prosail_VAE = get_prosail_VAE(rsr_dir, vae_params=vae_params, device=device,
                                  refl_norm_mean=norm_mean, refl_norm_std=norm_std)
    
    optimizer = optim.Adam(prosail_VAE.parameters(), lr=params["lr"])
    # prosail_VAE.load_ae("/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/results/" + "/prosailvae_weigths.tar", optimizer=optimizer)

    # Training
    all_train_loss_df, all_valid_loss_df = training_loop(prosail_VAE, 
                                                            optimizer, 
                                                            params['epochs'],
                                                            train_loader, 
                                                            valid_loader,
                                                            res_dir=res_dir,
                                                            n_samples=params["n_samples"]) 
    # Saving Loss
    loss_dir = res_dir + "/loss/"
    os.makedirs(loss_dir)
    all_train_loss_df.to_csv(loss_dir + "train_loss.csv")
    all_valid_loss_df.to_csv(loss_dir + "valid_loss.csv")
    loss_curve(all_train_loss_df, save_file=loss_dir+"train_loss.svg")
    loss_curve(all_valid_loss_df, save_file=loss_dir+"valid_loss.svg")
    
    # Computing metrics
    loader = get_simloader(file_prefix="test_", data_dir=data_dir)

    alpha_pi = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 
                0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    alpha_pi.reverse()
    prosail_VAE.eval()
    
    (mae, mpiw, picp, mare, 
    sim_dist, tgt_dist, rec_dist) = get_metrics(prosail_VAE, loader, 
                              n_pdf_sample_points=3001,
                              alpha_conf=alpha_pi)
    save_metrics(res_dir, mae, mpiw, picp, alpha_pi)
    maer = pd.read_csv(res_dir+"/metrics/maer.csv").drop(columns=["Unnamed: 0"])
    mpiwr = pd.read_csv(res_dir+"/metrics/mpiwr.csv").drop(columns=["Unnamed: 0"])
    
    # Plotting results
    metrics_dir = res_dir + "/metrics_plot/"
    os.makedirs(metrics_dir)
    plot_metrics(metrics_dir, alpha_pi, maer, mpiwr, picp, mare)
    rec_dir = res_dir + "/reconstruction/"
    os.makedirs(rec_dir)
    plot_rec_and_latent(prosail_VAE, loader, rec_dir, n_plots=20)
    
    plot_param_dist(metrics_dir, sim_dist, tgt_dist)
    plot_pred_vs_tgt(metrics_dir, sim_dist, tgt_dist)
    ssimulator = prosail_VAE.decoder.ssimulator
    refl_dist = loader.dataset[:][0]
    plot_refl_dist(rec_dist, refl_dist, res_dir, normalized=False, ssimulator=prosail_VAE.decoder.ssimulator)
    
    normed_rec_dist =  (rec_dist - ssimulator.norm_mean) / ssimulator.norm_std 
    normed_refl_dist =  (refl_dist - ssimulator.norm_mean) / ssimulator.norm_std 
    plot_refl_dist(normed_rec_dist, normed_refl_dist, metrics_dir, normalized=True, ssimulator=prosail_VAE.decoder.ssimulator)
    
    pair_plot(normed_rec_dist, tensor_2=None, features = BANDS, 
              res_dir=metrics_dir, filename='normed_rec_pair_plot.svg')
    pair_plot(normed_refl_dist, tensor_2=None, features = BANDS, 
              res_dir=metrics_dir, filename='normed_s2bands_pair_plot.svg')
    
    pair_plot(sim_dist.squeeze(), tensor_2=None, features = PROSAILVARS, 
              res_dir=metrics_dir, filename='sim_prosail_pair_plot.svg')
    pair_plot(tgt_dist.squeeze(), tensor_2=None, features = PROSAILVARS, 
              res_dir=metrics_dir, filename='ref_prosail_pair_plot.svg')