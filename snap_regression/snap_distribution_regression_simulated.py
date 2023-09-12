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
from dataset.loaders import get_simloader
from snap_regression.snap_nn import SnapNN
from prosailvae.dist_utils import kl_tntn

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
    return parser

def simulate_data_sets(n_eval:int=20000, n_samples_sub:int=20000, 
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
                                        scale=tg_mu_0,
                                        loc=tg_sigma_0,
                                        law="gaussian")
    print("Simulating evaluation data...")
    save_dataset(save_dir, "evaluation_", nb_simus=n_eval, rsr_dir=rsr_dir,weiss_mode=True, 
                 prosail_var_dist_type="new",lai_var_dist=lai_var_dist)

    for i, mu in enumerate(tg_mu):
        for j, sigma in enumerate(tg_sigma):
            print(f"Simulating training data (mu:{mu}, sigma:{sigma}) ...")
            lai_var_dist = VariableDistribution(low=lai_low,
                                                high=lai_high,
                                                scale=sigma,
                                                loc=mu,
                                                law="gaussian")
            save_dataset(save_dir, "train_mu_{mu}_sigma_{sigma}_", nb_simus=n_samples_sub, 
                         rsr_dir=rsr_dir, weiss_mode=True, 
                         prosail_var_dist_type="new", lai_var_dist=lai_var_dist)

def get_weiss_simloader(data_dir, file_prefix, valid_ratio=0.05, batch_size=1024, shuffle=True):
    prosail_s2_sim = torch.load(data_dir + f"/{file_prefix}prosail_s2_sim_refl.pt")
    vars = [6, 12, 11, 13]
    prosail_vars = torch.load(data_dir + f"/{file_prefix}prosail_sim_vars.pt")[:,vars]
    s2_a = prosail_vars[:,1:]
    lai = prosail_vars[:,:1].float()
    snap_input = torch.cat((prosail_s2_sim, np.cos(np.deg2rad(s2_a))), 1).float()
    if valid_ratio == 0.0:
        dataset = TensorDataset(snap_input, lai)
        loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=0, shuffle=shuffle)
        return loader
    n_valid = int(valid_ratio * lai.size(0))
    seed = 4567895683301
    g_cpu = torch.Generator()
    g_cpu.manual_seed(seed)
    idx = torch.randperm(len(lai), generator=g_cpu)
    
    g_cpu = torch.Generator()
    g_cpu.manual_seed(seed)

    idx = torch.randperm(lai.size(0), generator=g_cpu)

    data_valid = snap_input[idx[:n_valid],:]
    data_train = snap_input[idx[n_valid:],:]
    lai_valid = lai[idx[:n_valid],:]
    lai_train = lai[idx[n_valid:],:]
    train_dataset = TensorDataset(data_train, lai_train)
    valid_dataset = TensorDataset(data_valid, lai_valid)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                              num_workers=0, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size,
                                num_workers=0, shuffle=True)
    return train_loader, valid_loader

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
              "-e", "3",
              "-n", "2",
              "-lr", "0.001",
              ]
        disable_tqdm=False
        tg_mu = [1, 2]
        tg_sigma = [0.5, 3]
        n_eval = 1000
        n_samples_sub=1000
        parser = get_parser().parse_args(args)

    else:
        disable_tqdm=True
        parser = get_parser().parse_args()
        tg_mu = [0, 1, 2, 3, 4]
        tg_sigma = [0.5, 1, 2, 3]
        n_eval = 20000
        n_samples_sub=20000
    tg_mu_0 = 2
    tg_sigma_0 = 3
    batch_size = 1024
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
    eval_loader = get_weiss_simloader(valid_ratio=0.0, file_prefix="evaluation_",
                                      batch_size=batch_size, data_dir=data_dir)
    
    tg_mu_list = []
    tg_sigma_list = []
    kl_list = []
    valid_loss_list = []
    rmse_list = []
    for i, mu in enumerate(tg_mu):
        loc_bv = mu
        for j, sigma in enumerate(tg_sigma):
            scale_bv = sigma
            kl = kl_tntn(torch.as_tensor(tg_mu_0), torch.as_tensor(tg_sigma_0), 
                         torch.as_tensor(mu), torch.as_tensor(sigma), lower=torch.as_tensor(lai_low), 
                         upper=torch.as_tensor(lai_high)).item()
            kl_grid[i, j] = kl
            
            train_loader, valid_loader = get_weiss_simloader(valid_ratio=0.05,
                                                            file_prefix="train_mu_{mu}_sigma_{sigma}_",
                                                            batch_size=batch_size,
                                                            data_dir=data_dir)
            for n in range(parser.n_model_train):
                tg_mu_list.append(mu)
                tg_sigma_list.append(sigma)
                kl_list.append(kl)                
                model = SnapNN(ver="3A", variable="lai",
                               device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
                optimizer = optim.Adam(model.parameters(), lr=lr)
                lr_scheduler = ReduceLROnPlateau(optimizer=optimizer, patience=patience,
                                                threshold=0.001)
                _, all_valid_losses, _ = model.train_model(train_loader, valid_loader, optimizer,
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
                if not os.path.isfile(res_df_filename):
                    model_df.to_csv(res_df_filename, header=model_df.columns, index=False)
                else: # else it exists so append without writing the header
                    model_df.to_csv(res_df_filename, mode='a', index=False, header=False)
    pd.DataFrame(data={"mu":tg_mu_list, 
                       "sigma":tg_sigma_list, 
                       "kl":kl_list, 
                       "rmse":rmse_list, 
                       "loss":valid_loss_list}).to_csv(os.path.join(res_dir, "snap_distribution_regression_results_all.csv"), 
                                                       index=False)
    np.save(os.path.join(res_dir, "snap_distribution_regression_rmse.npy"), rmse_grid)
    np.save(os.path.join(res_dir, "snap_distribution_regression_kl.npy"), kl_grid)
    np.save(os.path.join(res_dir, "snap_distribution_regression_loss.npy"), valid_loss_grid)

if __name__ == "__main__":
    main()