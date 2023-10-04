import os
import shutil
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from prosailvae.prosail_var_dists import VariableDistribution
import argparse
from dataset.generate_dataset import save_dataset
import socket
from bvnet_regression.bvnet_utils import initialize_bvnet, get_bvnet_dataloader
from prosailvae.dist_utils import kl_tntn
from validation.validation import (get_validation_global_metrics, get_all_campaign_lai_results_BVNET,
                                   get_belsar_x_frm4veg_lai_results)

def get_parser():
    """
    Creates a new argument parser.
    """
    parser = argparse.ArgumentParser(description='Parser for data generation')

    parser.add_argument("-d", dest="data_dir",
                        help="path to data directory",
                        type=str, default="/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data_snap_simu_dist/")
    
    
    parser.add_argument("-rsr", dest="rsr_dir",
                        help="directory of rsr_file",
                        type=str, default='/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/')
    
    parser.add_argument("-r", dest="res_dir",
                        help="path to results directory",
                        type=str, default="")
    
    parser.add_argument("-e", dest="epochs",
                        help="number of epochs",
                        type=int, default=1000)
    
    parser.add_argument("-n", dest="n_model_train",
                        type=int, default=20)
    
    parser.add_argument("-lr", dest="lr", 
                        type=float, default=0.001)
    
    parser.add_argument("-s", dest="simulate_dataset",
                        type=bool, default=False)
    
    parser.add_argument("-v", dest="validate_on_terrain",
                        type=bool, default=False)
    
    parser.add_argument("-si", dest="sigma",
                        type=float, default=3)
    return parser

def simulate_data_sets(n_eval:int=20000, 
                       n_samples_sub:int=20000, 
                       save_dir:str="",
                       tg_mu: list=[0,4], 
                       tg_sigma:list=[1,4],
                       tg_mu_0:float=2,
                       tg_sigma_0:float=3, 
                       lai_low:float=0,
                       lai_high:float=15,
                       rsr_dir:str=""):

    lai_var_dist = VariableDistribution(low=lai_low,
                                        high=lai_high,
                                        scale=tg_sigma_0,
                                        loc=tg_mu_0,
                                        law="gaussian")
    print("Simulating evaluation data...")
    save_dataset(save_dir, "evaluation_", nb_simus=n_eval, rsr_dir=rsr_dir, bvnet_bands=False, 
                 prosail_var_dist_type="new_v2",lai_var_dist=lai_var_dist)

    for i, mu in enumerate(tg_mu):
        for j, sigma in enumerate(tg_sigma):
            print(f"Simulating training data (mu:{mu}, sigma:{sigma}) ...")
            lai_var_dist = VariableDistribution(low=lai_low,
                                                high=lai_high,
                                                scale=sigma,
                                                loc=mu,
                                                law="gaussian")
            save_dataset(save_dir, f"train_mu_{mu}_sigma_{sigma}_", nb_simus=n_samples_sub, 
                         rsr_dir=rsr_dir, bvnet_bands=False, noise=0.01,
                         prosail_var_dist_type="new_v2", lai_var_dist=lai_var_dist, lai_thresh=None)

def get_dataset_rmse(dataset, model):
    with torch.no_grad():
        lai_pred = model.forward(dataset[:][0].to(model.device)).cpu()
        lai_true = dataset[:][1].cpu()
        rmse = (lai_pred - lai_true).pow(2).mean().sqrt().item()
    return rmse

def main():
    if socket.gethostname()=='CELL200973':
        args=["-d", "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/snap_distribution_data/",
              "-r", "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/results/snap_distribution_validation/",
              "-e", "400",
              "-n", "2",
              "-lr", "0.001",
              '-v', 'True',
              "-s", "True",
              "-si", "3"
              ]
        disable_tqdm=False
        tg_mu = [2]
        tg_sigma = [parser.sigma]
        n_eval = 40000
        n_samples_sub=40000
        parser = get_parser().parse_args(args)
        frm4veg_data_dir = "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/frm4veg_validation"
        frm4veg_2021_data_dir = "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/frm4veg_2021_validation"
        belsar_data_dir = "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/belSAR_validation"
    else:
        disable_tqdm=True
        parser = get_parser().parse_args()
        tg_mu = [0, 1, 2, 3, 4]
        # tg_mu = [4]
        # tg_sigma = [0.5, 1, 2, 3, 4]
        # tg_sigma = [0.5, 1, 2]
        tg_sigma = [3, 4]
        tg_sigma = [parser.sigma]
        n_eval = 40000
        n_samples_sub=40000
        frm4veg_data_dir = "/work/scratch/zerahy/prosailvae/data/frm4veg_validation"
        frm4veg_2021_data_dir = "/work/scratch/zerahy/prosailvae/data/frm4veg_2021_validation"
        belsar_data_dir = "/work/scratch/zerahy/prosailvae/data/belSAR_validation"
    tg_mu_0 = 2
    tg_sigma_0 = 3
    batch_size = 4096
    res_dir = parser.res_dir
    data_dir = parser.data_dir
    lr = parser.lr
    patience = 10
    lai_low = 0
    lai_high = 15
    if not os.path.isdir(res_dir):
        os.makedirs(res_dir)
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    if parser.simulate_dataset:
        simulate_data_sets(n_eval=n_eval, n_samples_sub=n_samples_sub, save_dir=data_dir, 
                           tg_mu=tg_mu, tg_sigma=tg_sigma, tg_mu_0=tg_mu_0, tg_sigma_0=tg_sigma_0, 
                           rsr_dir=parser.rsr_dir, lai_low=lai_low, lai_high=lai_high)
    rmse_grid = np.zeros((len(tg_mu), len(tg_sigma), parser.n_model_train))
    kl_grid = np.zeros((len(tg_mu), len(tg_sigma)))
    valid_loss_grid = np.zeros((len(tg_mu), len(tg_sigma), parser.n_model_train))

    bands = [1, 2, 3, 4, 5, 7, 8, 9]
    vars = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 11, 13]
    prosail_s2_sim = torch.load(os.path.join(data_dir, f"evaluation_prosail_s2_sim_refl.pt"))[:, bands]
    prosail_vars = torch.load(os.path.join(data_dir, f"evaluation_prosail_sim_vars.pt"))[:, vars]
    eval_loader, _, _, _ = get_bvnet_dataloader(variable='lai', valid_ratio=0.0, batch_size=batch_size, s2_r=prosail_s2_sim, 
                                                prosail_vars=prosail_vars, max_samples=50000)

    
    tg_mu_list = []
    tg_sigma_list = []
    kl_list = []
    valid_loss_list = []
    rmse_list = []
    lai_terrain_rmse_list = []

    for i, mu in enumerate(tg_mu):
        # loc_bv = mu
        for j, sigma in enumerate(tg_sigma):
            # scale_bv = sigma
            kl = kl_tntn(torch.as_tensor(tg_mu_0), torch.as_tensor(tg_sigma_0), 
                         torch.as_tensor(mu), torch.as_tensor(sigma), lower=torch.as_tensor(lai_low), 
                         upper=torch.as_tensor(lai_high)).item()
            kl_grid[i, j] = kl
            prosail_s2_sim = torch.load(os.path.join(data_dir, f"train_mu_{mu}_sigma_{sigma}_prosail_s2_sim_refl.pt"))[:, bands]
            prosail_vars = torch.load(os.path.join(data_dir, f"train_mu_{mu}_sigma_{sigma}_prosail_sim_vars.pt"))[:, vars]
            train_loader, valid_loader, loc_bv, scale_bv = get_bvnet_dataloader(variable="lai", valid_ratio=0.05,
                                                              batch_size=batch_size, s2_r=prosail_s2_sim, 
                                                              prosail_vars=prosail_vars)
            for n in range(parser.n_model_train):
                tg_mu_list.append(mu)
                tg_sigma_list.append(sigma)
                kl_list.append(kl)                
                model = initialize_bvnet("lai", train_loader, valid_loader, loc_bv, scale_bv, res_dir, 
                                                n_models=10, n_epochs=10, lr=1e-3)
                optimizer = optim.Adam(model.parameters(), lr=lr)
                lr_scheduler = ReduceLROnPlateau(optimizer=optimizer, patience=patience, threshold=0.001)
                _, all_valid_losses, all_lr = model.train_model(train_loader, valid_loader, optimizer,
                                                                epochs=parser.epochs, lr_scheduler=lr_scheduler,
                                                                disable_tqdm=disable_tqdm, lr_recompute=patience, 
                                                                loc_bv=loc_bv, scale_bv=scale_bv, res_dir=res_dir)
                valid_loss_grid[i,j,n] = min(all_valid_losses)
                valid_loss_list.append(min(all_valid_losses))
                rmse_grid[i,j,n] = get_dataset_rmse(eval_loader.dataset, model)
                rmse_list.append(rmse_grid[i,j,n])
                res_df_filename = os.path.join(res_dir, "snap_distribution_regression_results.csv")
                model_df = pd.DataFrame(data={"mu":[mu], 
                                              "sigma":[sigma], 
                                              "kl":[kl], 
                                              "rmse":[rmse_grid[i,j,n]], 
                                              "loss":[min(all_valid_losses)]})
                if parser.validate_on_terrain:

                    (barrax_results, barrax_2021_results, wytham_results, belsar_results, 
                    all_belsar) = get_all_campaign_lai_results_BVNET(frm4veg_data_dir, frm4veg_2021_data_dir, 
                                                                    belsar_data_dir, res_dir,
                                                                    method="simple_interpolate", get_all_belsar=False, 
                                                                    remove_files=True, lai_bvnet=model)    
                    df_results = get_belsar_x_frm4veg_lai_results(belsar_results, barrax_results, barrax_2021_results, wytham_results,
                                                                  frm4veg_lai="lai", get_reconstruction_error=False)
                    rmse, _, _, _ = get_validation_global_metrics(df_results, decompose_along_columns=["Campaign"], variable="lai")
                    lai_terrain_rmse = rmse['Campaign'][f'lai_rmse_all'].values[0]
                    model_df = pd.concat((model_df, pd.DataFrame({"terrain_rmse":[lai_terrain_rmse]})), axis=1)
                    lai_terrain_rmse_list.append(lai_terrain_rmse)
                if not os.path.isfile(res_df_filename):
                    model_df.to_csv(res_df_filename, header=model_df.columns, index=False)
                else: # else it exists so append without writing the header
                    model_df.to_csv(res_df_filename, mode='a', index=False, header=False)

    all_results_df = pd.DataFrame(data={"mu":tg_mu_list, 
                       "sigma":tg_sigma_list, 
                       "kl":kl_list, 
                       "rmse":rmse_list, 
                       "loss":valid_loss_list})
    if parser.validate_on_terrain:
        all_results_df = pd.concat((model_df, pd.DataFrame({"terrain_rmse":lai_terrain_rmse_list})), axis=1)
    all_results_df.to_csv(os.path.join(res_dir, "snap_distribution_regression_results_all.csv"), index=False)
    np.save(os.path.join(res_dir, "snap_distribution_regression_rmse.npy"), rmse_grid)
    np.save(os.path.join(res_dir, "snap_distribution_regression_kl.npy"), kl_grid)
    np.save(os.path.join(res_dir, "snap_distribution_regression_loss.npy"), valid_loss_grid)

if __name__ == "__main__":
    main()