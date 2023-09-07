import torch
import os
import numpy as np
import pandas as pd
if __name__ == "__main__":
    from snap_nn import SnapNN
else:
    from snap_regression.snap_nn import SnapNN
import socket
from dataset.weiss_utils import load_weiss_dataset
import prosailvae
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from tqdm import tqdm
from validation.validation import get_all_campaign_CCC_results_SNAP, get_frm4veg_ccc_results, get_validation_global_metrics
from prosailvae.ProsailSimus import ProsailSimulator, SensorSimulator
from prosailvae.prosail_var_dists import get_prosail_var_dist

import argparse

def get_parser():
    """
    Creates a new argument parser.
    """
    parser = argparse.ArgumentParser(description='Parser for data generation')
    
    parser.add_argument("-r", dest="res_dir",
                        help="path to results directory",
                        type=str, default="")
    parser.add_argument('-p', dest="last_prosail",
                        help="toggle last prosail version",
                        type=bool, default=False)
    return parser

def convert_prosail_data_set_from_weiss(nb_simus=2048, noise=0, psimulator=None, ssimulator=None, 
                                        n_samples_per_batch=1024):
    _, prosail_vars_weiss = load_weiss_dataset(os.path.join(prosailvae.__path__[0], os.pardir) + "/field_data/lai/", 
                                               mode="pvae")
    _, prosail_vars_snap = load_weiss_dataset(os.path.join(prosailvae.__path__[0], os.pardir) + "/field_data/lai/", 
                                               mode="snap")
    s2_a_snap = prosail_vars_snap[:,-3:]
    # s2_a_weiss = prosail_vars_weiss[:,-3:]
    # prosail_vars_weiss = prosail_vars_weiss[:,:-3]
    nb_simus = min(nb_simus, 41474)
    n_full_batch = nb_simus // n_samples_per_batch
    last_batch = nb_simus - nb_simus // n_samples_per_batch * n_samples_per_batch

    prosail_vars = np.zeros((nb_simus, 14))
    prosail_s2_sim = np.zeros((nb_simus, ssimulator.rsr.size(1)))
    for i in range(n_full_batch):
        prosail_vars[i*n_samples_per_batch : (i+1) * n_samples_per_batch,:] = prosail_vars_weiss[i*n_samples_per_batch : (i+1) * n_samples_per_batch,:]
        prosail_r = psimulator(torch.from_numpy(prosail_vars[i*n_samples_per_batch : (i+1) * n_samples_per_batch,:]).view(n_samples_per_batch,-1).float())
        sim_s2_r = ssimulator(prosail_r).numpy()
        if noise>0:
            sigma = np.random.rand(n_samples_per_batch,1) * noise * np.ones_like(sim_s2_r)
            add_noise = np.random.normal(loc = np.zeros_like(sim_s2_r), scale=sigma, size=sim_s2_r.shape)
            sim_s2_r += add_noise
        prosail_s2_sim[i*n_samples_per_batch : (i+1) * n_samples_per_batch,:] = sim_s2_r
    if last_batch > 0:
        prosail_vars[n_full_batch*n_samples_per_batch:,:] = prosail_vars_weiss[n_full_batch*n_samples_per_batch:,:]
        sim_s2_r = ssimulator(psimulator(torch.from_numpy(prosail_vars[n_full_batch*n_samples_per_batch:,:]).view(last_batch,-1).float())).numpy()
        if noise>0:
            sigma = np.random.rand(last_batch,1) * noise * np.ones_like(sim_s2_r)
            add_noise = np.random.normal(loc = np.zeros_like(sim_s2_r), scale=sigma, size=sim_s2_r.shape)
            sim_s2_r += add_noise
        prosail_s2_sim[n_full_batch*n_samples_per_batch:,:] = sim_s2_r
    prosail_vars[:,-3:] = s2_a_snap # permuting angles
    return prosail_vars, prosail_s2_sim


def get_weiss_dataloader(variable='lai', valid_ratio=0.05, batch_size=1024, s2_r=None, prosail_vars=None):
    if prosail_vars is None or s2_r is None:
        s2_r, prosail_vars = load_weiss_dataset(os.path.join(prosailvae.__path__[0], os.pardir) + "/field_data/lai/", mode="snap")
    
    s2_a = prosail_vars[:,-3:]
    bv = {"lai":prosail_vars[:,6],
          "cab":prosail_vars[:,1],
          "cw": prosail_vars[:,4],
          "ccc": prosail_vars[:,6] * prosail_vars[:,1],
          "cwc": prosail_vars[:,6] * prosail_vars[:,4]}
    bv = bv[variable]
    loc_bv = bv.mean(0)
    scale_bv = bv.std(0)
    data_weiss = torch.from_numpy(np.concatenate((s2_r, np.cos(np.deg2rad(s2_a)), bv.reshape(-1, 1)), 1))
    seed = 4567895683301
    g_cpu = torch.Generator()
    g_cpu.manual_seed(seed)
    idx = torch.randperm(len(bv), generator=g_cpu)
    
    g_cpu = torch.Generator()
    g_cpu.manual_seed(seed)

    n_valid = int(valid_ratio * data_weiss.size(0))
    idx = torch.randperm(data_weiss.size(0), generator=g_cpu)
    data_valid = data_weiss[idx[:n_valid],:].float()
    data_train = data_weiss[idx[n_valid:],:].float()
    train_dataset = TensorDataset(data_train[:,:-1], data_train[:,-1].unsqueeze(1))
    valid_dataset = TensorDataset(data_valid[:,:-1], data_valid[:,-1].unsqueeze(1))

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                              num_workers=0, shuffle=True)
    if n_valid > 0:
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size,
                                  num_workers=0, shuffle=True)
    else:
        valid_loader = None
    return train_loader, valid_loader, loc_bv, scale_bv

def main():
    if socket.gethostname()=='CELL200973':
        frm4veg_data_dir = "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/frm4veg_validation"
        frm4veg_2021_data_dir = "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/frm4veg_2021_validation"
        res_dir = "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/results/snap_ccc/" 
        rsr_dir = "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data"
    else:
        frm4veg_data_dir = "/work/scratch/zerahy/prosailvae/data/frm4veg_validation"
        frm4veg_2021_data_dir = "/work/scratch/zerahy/prosailvae/data/frm4veg_2021_validation"
        parser = get_parser().parse_args()
        res_dir = parser.res_dir
        rsr_dir = "/work/scratch/zerahy/prosailvae/data/"
    if not os.path.isdir(res_dir):
        os.makedirs(res_dir)
    lr = 1e-3
    patience = 20
    epochs=2000
    disable_tqdm=False
    prosail_vars = None
    prosail_s2_sim = None
    if parser.last_prosail:
        psimulator = ProsailSimulator()
        bands = [2, 3, 4, 5, 6, 8, 11, 12]
        ssimulator = SensorSimulator(rsr_dir + "/sentinel2.rsr", bands=bands)
        prosail_vars, prosail_s2_sim = convert_prosail_data_set_from_weiss(nb_simus=43000,
                                                                            noise=0,
                                                                            psimulator=psimulator,
                                                                            ssimulator=ssimulator,
                                                                            n_samples_per_batch=1024)
    model_dict = {}
    plot_loss = False
    n_models=10
    batch_size=1024
    results_dict = {}
    for variable in ["cab", "ccc"]:
        results_dict[variable] = []
        for i in range(n_models):
            train_loader, valid_loader, loc_bv, scale_bv = get_weiss_dataloader(variable=variable, valid_ratio=0.05, 
                                                                                batch_size=batch_size, 
                                                                                prosail_vars=prosail_vars, s2_r=prosail_s2_sim)
            model = SnapNN(ver="3A", variable=variable, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            # model_dict[variable] = model
            optimizer = optim.Adam(model.parameters(), lr=lr)
            lr_scheduler = ReduceLROnPlateau(optimizer=optimizer, patience=patience,
                                             threshold=0.001)
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
                
            barrax_results, barrax_2021_results, wytham_results = get_all_campaign_CCC_results_SNAP(frm4veg_data_dir, 
                                                                                                    frm4veg_2021_data_dir,
                                                                                                    ccc_snap=model, 
                                                                                                    cab_mode=variable=="cab")
            df_results = get_frm4veg_ccc_results(barrax_results, barrax_2021_results, wytham_results, frm4veg_ccc="ccc",
                                            get_reconstruction_error=False)
            rmse, _, _, _ = get_validation_global_metrics(df_results, decompose_along_columns=["Campaign"], variable="ccc")
            results_dict[variable].append(rmse['Campaign'][f'ccc_rmse_all'].values[0])
    pd.DataFrame(results_dict).to_csv(os.path.join(res_dir, f'snap_{variable}_validation_rmse.csv'))

if __name__ =="__main__":
    main()