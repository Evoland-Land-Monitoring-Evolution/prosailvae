import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from prosailvae.ProsailSimus import PROSAILVARS

def preprocess_data(res_root_dir):
    
    configs = np.load(res_root_dir + "configs_var_params.npy")
    lat_nll = torch.load(res_root_dir + "lat_nll.pt")
    test_loss = pd.read_csv(res_root_dir + "all_losses.csv", index_col=[0])
    test_loss.replace(100000.0,np.nan, inplace=True)
    test_loss_all_nan = test_loss.isnull().all(1).values
    row_nan_test = np.arange(lat_nll.size(0))[test_loss_all_nan]

    lat_nll[lat_nll == 100000] = torch.nan
    config_filter = np.where(configs[:,:,:,0]<lat_nll.size(0))
    configs = configs[config_filter]

    for i in reversed(range(len(row_nan_test))):
        config_filter = np.where(configs[:,0] != row_nan_test[i])
        configs = configs[config_filter]
        lat_nll = lat_nll[config_filter[0],:,:]
    test_loss.drop(index=row_nan_test.tolist(), inplace=True)

    return configs, test_loss.values, lat_nll

def lat_nll_vs_test_nll(test_loss, lat_nll, save_dir=None):
    valid_test_loss = []
    valid_lat_nll = []
    for i in range(test_loss.shape[0]):
        for j in range(test_loss.shape[1]):
            test_loss_ij = test_loss[i,j]
            lat_nll_ij = lat_nll[i,j,:]
            if not np.isnan(test_loss_ij): 
                valid_test_loss.append(test_loss_ij)
                valid_lat_nll.append(lat_nll_ij)
    valid_lat_nll = torch.vstack(valid_lat_nll)
    for i in range(len(PROSAILVARS)):
        fig, ax = plt.subplots(1,1,dpi=150, tight_layout=True)
        ax.scatter(valid_test_loss, valid_lat_nll[:,i],marker='+')
        ax.set_xlabel("Test loss (NLL)")
        ax.set_ylabel(f"{PROSAILVARS[i]} loss (NLL)")
        ax.set_xscale("symlog",linthresh=1e-3)
        ax.set_yscale("symlog",linthresh=1e-2)
        if save_dir is not None:
            fig.savefig(save_dir+f'/test_loss_vs_{PROSAILVARS[i]}_NLL.svg')
    pass


def test_loss_vs_N(test_loss, configs, save_dir=None):
    best_fold_loss = np.nanmin(test_loss,1)
    fig, ax = plt.subplots(1,1,dpi=150, tight_layout=True)
    ax.scatter(configs[:,3], best_fold_loss, marker='+')
    ax.set_xlabel("RNN Length - Number of RNN blocks (N)")
    ax.set_ylabel("Test loss (NLL)")
    if save_dir is not None:
        fig.savefig(save_dir+'/test_loss_vs_N.svg')
    return

def test_loss_vs_L(test_loss, configs, save_dir=None):
    best_fold_loss = np.nanmin(test_loss,1)
    fig, ax = plt.subplots(1,1,dpi=150, tight_layout=True)
    ax.scatter(configs[:,1], best_fold_loss, marker='+')
    ax.set_xlabel("RNN Width - Size of layers (L)")
    ax.set_ylabel("Test loss (NLL)")
    if save_dir is not None:
        fig.savefig(save_dir+'/test_loss_vs_L.svg')
    return

def test_loss_vs_D(test_loss, configs, save_dir=None):
    best_fold_loss = np.nanmin(test_loss,1)
    fig, ax = plt.subplots(1,1,dpi=150, tight_layout=True)
    ax.scatter(configs[:,2], best_fold_loss, marker='+')
    ax.set_xlabel("RNN Depth - Depth of RNN blocks (D)")
    ax.set_ylabel("Test loss (NLL)")
    if save_dir is not None:
        fig.savefig(save_dir+'/test_loss_vs_D.svg')
    return

def get_bests(test_loss, configs, n=5):
    best_fold_loss = np.nanmin(test_loss,1)
    ind_n_best = np.argpartition(best_fold_loss, n)[:n]
    sorted_n_best = ind_n_best[np.argsort(best_fold_loss[ind_n_best])]
    top_n_loss = best_fold_loss[sorted_n_best]
    top_n_configs = configs[sorted_n_best, :]
    return top_n_loss, top_n_configs

def main():
    res_root_dir = "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/results/rnn_hyper_2/"
    configs, test_loss, lat_nll = preprocess_data(res_root_dir)
    plot_dir = res_root_dir + "/plots/"
    if not os.path.isdir(plot_dir):
        os.makedirs(plot_dir)
    top_n_loss, top_n_configs = get_bests(test_loss, configs, n=5)
    test_loss_vs_D(test_loss, configs, save_dir=plot_dir)
    test_loss_vs_N(test_loss, configs, save_dir=plot_dir)
    test_loss_vs_L(test_loss, configs, save_dir=plot_dir)
    lat_nll_vs_test_nll(test_loss, lat_nll, save_dir=plot_dir)
    pass

if __name__=="__main__":
    main()