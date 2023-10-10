import torch
import os
import numpy as np
import pandas as pd
from bvnet_regression.bvnet_utils import initialize_bvnet, get_bvnet_dataloader
import socket
from dataset.bvnet_dataset import load_bvnet_dataset
import prosailvae
import torch.optim as optim

from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from tqdm import tqdm
from validation.validation import (get_all_campaign_CCC_results_BVNET, get_frm4veg_ccc_results, 
                                   get_validation_global_metrics, get_all_campaign_lai_results_BVNET,
                                   get_belsar_x_frm4veg_lai_results)
from prosailvae.ProsailSimus import ProsailSimulator, SensorSimulator
from prosailvae.prosail_var_dists import get_prosail_var_dist

import argparse

def get_parser():
    """
    Creates a new argument parser.
    """
    parser = argparse.ArgumentParser(description='Parser for data generation')

    parser.add_argument("-n", dest="n_models",
                        help="Number of trained bvnets",
                        type=int, default=20)
    
    parser.add_argument("-b", dest="batch_size",
                        help="Number of trained sample per batch",
                        type=int, default=4096)
    
    parser.add_argument("-d", dest="data_dir",
                        help="path to data directory",
                        type=str, default="/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/sim_data_corr_v2")
    
    parser.add_argument("-r", dest="res_dir",
                        help="path to results directory",
                        type=str, default="/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/results")
    
    parser.add_argument('-p', dest="last_prosail",
                        help="toggle last prosail version to convert Jordi's data",
                        type=bool, default=False)
    
    parser.add_argument('-sd', dest="sim_data",
                        help="toggle my simulated data instead of Jordi's",
                        type=bool, default=False)
    
    parser.add_argument('-l', dest="lai_mode",
                        help="toggle lai instead of cab",
                        type=bool, default=False)
    
    parser.add_argument('-fp', dest="file_prefix",
                        help="prefix of simulated_data_set",
                        type=str, default="")
    return parser

def convert_prosail_data_set(nb_simus=2048, noise=0, psimulator=None, ssimulator=None, 
                                        n_samples_per_batch=1024):
    nb_simus = min(nb_simus, 41474)
    _, prosail_vars_pvae_like = load_bvnet_dataset(os.path.join(prosailvae.__path__[0], os.pardir) + "/field_data/lai/", 
                                                        mode="pvae", psoil0=0.3)
    _, prosail_vars_bvnet = load_bvnet_dataset(os.path.join(prosailvae.__path__[0], os.pardir) + "/field_data/lai/", 
                                                      mode="bvnet", psoil0=0.3)
    prosail_vars_pvae_like = prosail_vars_pvae_like[:nb_simus,:]
    prosail_vars_bvnet = prosail_vars_bvnet[:nb_simus,:]
    s2_a_bvnet = prosail_vars_bvnet[:,-3:]
    
    n_full_batch = nb_simus // n_samples_per_batch
    last_batch = nb_simus - nb_simus // n_samples_per_batch * n_samples_per_batch

    prosail_vars = np.zeros((nb_simus, 14))
    prosail_s2_sim = np.zeros((nb_simus, ssimulator.rsr.size(1)))
    for i in range(n_full_batch):
        prosail_vars[i*n_samples_per_batch:(i+1)*n_samples_per_batch,
                     :] = prosail_vars_pvae_like[i*n_samples_per_batch : (i+1) * n_samples_per_batch,:]
        prosail_r = psimulator(torch.from_numpy(prosail_vars[i*n_samples_per_batch : (i+1) * n_samples_per_batch,
                                                             :]).view(n_samples_per_batch,-1).float())
        sim_s2_r = ssimulator(prosail_r).numpy()
        if noise>0:
            sigma = np.random.rand(n_samples_per_batch,1) * noise * np.ones_like(sim_s2_r)
            add_noise = np.random.normal(loc = np.zeros_like(sim_s2_r), scale=sigma, size=sim_s2_r.shape)
            sim_s2_r += add_noise
        prosail_s2_sim[i*n_samples_per_batch : (i+1) * n_samples_per_batch,:] = sim_s2_r
    if last_batch > 0:
        prosail_vars[n_full_batch*n_samples_per_batch:,:] = prosail_vars_pvae_like[n_full_batch*n_samples_per_batch:,:]
        sim_s2_r = ssimulator(psimulator(torch.from_numpy(prosail_vars[n_full_batch*n_samples_per_batch:,:]).view(last_batch,-1).float())).numpy()
        if noise>0:
            sigma = np.random.rand(last_batch,1) * noise * np.ones_like(sim_s2_r)
            add_noise = np.random.normal(loc = np.zeros_like(sim_s2_r), scale=sigma, size=sim_s2_r.shape)
            sim_s2_r += add_noise
        prosail_s2_sim[n_full_batch*n_samples_per_batch:,:] = sim_s2_r
    prosail_vars[:,-3:] = s2_a_bvnet # permuting angles for bvnet
    return prosail_vars, prosail_s2_sim

def main():
    if socket.gethostname()=='CELL200973':
        frm4veg_data_dir = "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/frm4veg_validation"
        frm4veg_2021_data_dir = "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/frm4veg_2021_validation"
        belsar_data_dir = "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/belSAR_validation"
        res_dir = "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/results/bvnet_lai_prospect_d/" 
        rsr_dir = "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data"

        epochs=500
        n_models=2
        args = [
            "-sd", "True",
            # "-p", "True",
            "-d", "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/sim_data_corr_v2_test_prospect_vD",
            "-l", "True",
            "-fp", "PROSPECTD_"
                ]
        parser = get_parser().parse_args(args)
        # file_prefix = "evaluation_"
        file_prefix = parser.file_prefix
        
    else:
        frm4veg_data_dir = "/work/scratch/zerahy/prosailvae/data/frm4veg_validation"
        frm4veg_2021_data_dir = "/work/scratch/zerahy/prosailvae/data/frm4veg_2021_validation"
        belsar_data_dir = "/work/scratch/zerahy/prosailvae/data/belSAR_validation"
        parser = get_parser().parse_args()
        res_dir = parser.res_dir
        rsr_dir = "/work/scratch/zerahy/prosailvae/data/"
        epochs=500
        n_models=parser.n_models
        file_prefix = parser.file_prefix
        # data_dir = "/work/scratch/zerahy/prosailvae/data/1e5_simulated_full_bands_new_dist_old_corr/"
    data_dir = parser.data_dir
    belsar_pred_dir = parser.res_dir
    if not os.path.isdir(res_dir):
        os.makedirs(res_dir)
    lr = 1e-3
    patience = 20
    # epochs=2000
    disable_tqdm=False
    prosail_vars = None
    prosail_s2_sim = None
    data_set_txt = "weiss"
    variable = "lai" if parser.lai_mode else "ccc"
    if parser.last_prosail:
        psimulator = ProsailSimulator()
        bands = [2, 3, 4, 5, 6, 8, 11, 12]
        ssimulator = SensorSimulator(rsr_dir + "/sentinel2.rsr", bands=bands)
        prosail_vars, prosail_s2_sim = convert_prosail_data_set(nb_simus=41000,
                                                                noise=0,
                                                                psimulator=psimulator,
                                                                ssimulator=ssimulator,
                                                                n_samples_per_batch=1024)
        data_set_txt = "projected"
        print("Projecting samples with PROSPECT-D + 4SAIL")
    elif parser.sim_data:
        bands = [1, 2, 3, 4, 5, 7, 8, 9]
        prosail_s2_sim = torch.load(os.path.join(data_dir, f"{file_prefix}prosail_s2_sim_refl.pt"))[:, bands]
        vars = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 11, 13]
        prosail_vars = torch.load(os.path.join(data_dir, f"{file_prefix}prosail_sim_vars.pt"))[:, vars]
        data_set_txt = "simulated"
        print(f"Using simulated data from {data_dir}/{file_prefix}prosail_s2_sim_refl.pt")
    else:
        print(f"Using Jordi's Data")
    model_dict = {}
    plot_loss = True
    # n_models=10
    batch_size = parser.batch_size
    results_dict = {}
    variable = "ccc" if not parser.lai_mode else "lai"

    results_dict[variable] = []
    for i in range(n_models):
        train_loader, valid_loader, loc_bv, scale_bv = get_bvnet_dataloader(variable=variable, valid_ratio=0.1, 
                                                                            batch_size=batch_size, 
                                                                            prosail_vars=prosail_vars, s2_r=prosail_s2_sim)
        model = initialize_bvnet(variable, train_loader, valid_loader, loc_bv, scale_bv, res_dir, 
                                    n_models=10, n_epochs=20, lr=1e-3)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        lr_scheduler = ReduceLROnPlateau(optimizer=optimizer, patience=patience, threshold=0.001)
        _, all_valid_losses, all_lr = model.train_model(train_loader, valid_loader, optimizer,
                                                        epochs=epochs, lr_scheduler=lr_scheduler,
                                                        disable_tqdm=disable_tqdm, lr_recompute=patience, 
                                                        loc_bv=loc_bv, scale_bv=scale_bv, res_dir=res_dir)
        if plot_loss:
            fig, axs = plt.subplots(2,1, sharex=True)
            axs[0].scatter(np.arange(len(all_valid_losses)), all_valid_losses)
            axs[1].scatter(np.arange(len(all_valid_losses)), all_lr)
            axs[0].set_yscale('log')
            axs[1].set_yscale('log')
            axs[1].set_xlabel("epoch")
            axs[0].set_ylabel("Loss (MSE)")
            axs[1].set_ylabel("LR")
        if variable == "ccc" or variable == "cab":
            barrax_results, barrax_2021_results, wytham_results = get_all_campaign_CCC_results_BVNET(frm4veg_data_dir, 
                                                                                                    frm4veg_2021_data_dir,
                                                                                                    ccc_bvnet=model, 
                                                                                                    cab_mode=variable=="cab")
            df_results = get_frm4veg_ccc_results(barrax_results, barrax_2021_results, wytham_results, 
                                                 frm4veg_ccc="ccc", get_reconstruction_error=False)
            rmse, _, _, _ = get_validation_global_metrics(df_results, decompose_along_columns=["Campaign"], variable="ccc")
            results_dict[variable].append(rmse['Campaign'][f'ccc_rmse_all'].values[0])
            res_df_filename = os.path.join(res_dir, f'bvnet_{variable}_validation_rmse.csv')
            if not os.path.isfile(res_df_filename):
                pd.DataFrame({variable:[rmse['Campaign'][f'ccc_rmse_all'].values[0]]}).to_csv(res_df_filename, 
                                                                                                header=[variable], 
                                                                                                index=False)
            else: # else it exists so append without writing the header
                pd.DataFrame({variable:[rmse['Campaign'][f'ccc_rmse_all'].values[0]]}).to_csv(res_df_filename, mode='a', 
                                                                                              index=False, header=False) 
        else:
            (barrax_results, barrax_2021_results, wytham_results, belsar_results, 
                all_belsar) = get_all_campaign_lai_results_BVNET(frm4veg_data_dir, frm4veg_2021_data_dir, 
                                                                belsar_data_dir, belsar_pred_dir,
                                                                method="simple_interpolate", get_all_belsar=False, 
                                                                remove_files=True, lai_bvnet=model)    
            df_results = get_belsar_x_frm4veg_lai_results(belsar_results, barrax_results, barrax_2021_results, 
                                                            wytham_results,
                                                            frm4veg_lai="lai", get_reconstruction_error=False)
            rmse, _, _, _ = get_validation_global_metrics(df_results, decompose_along_columns=["Campaign"], variable="lai")
            results_dict[variable].append(rmse['Campaign'][f'lai_rmse_all'].values[0])
            res_df_filename = os.path.join(res_dir, f'bvnet_{data_set_txt}_{variable}_validation_rmse.csv')
            if not os.path.isfile(res_df_filename):
                pd.DataFrame({variable:[rmse['Campaign'][f'lai_rmse_all'].values[0]]}).to_csv(res_df_filename, 
                                                                                              header=[variable], 
                                                                                              index=False)
            else: # else it exists so append without writing the header
                pd.DataFrame({variable:[rmse['Campaign'][f'lai_rmse_all'].values[0]]}).to_csv(res_df_filename, mode='a', index=False, header=False)
    pd.DataFrame(results_dict).to_csv(os.path.join(res_dir, f'bvnet_{variable}_validation_rmse_all.csv'))

if __name__ =="__main__":
    main()