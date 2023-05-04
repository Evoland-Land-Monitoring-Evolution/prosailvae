import os
import shutil
from typing import Callable, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import r2_score
import argparse
import socket
import prosailvae
from prosailvae.dist_utils import (sample_truncated_gaussian, kl_tntn, truncated_gaussian_pdf, 
                                   numerical_kl_from_pdf, truncated_gaussian_cdf)
from snap_nn import SnapNN, test_snap_nn
from utils.image_utils import rgb_render



def get_parser():
    """
    Creates a new argument parser.
    """
    parser = argparse.ArgumentParser(description='Parser for data generation')

    parser.add_argument("-d", dest="data_dir",
                        help="path to data direcotry",
                        type=str, default="/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/")
    
    parser.add_argument("-r", dest="results_dir",
                        help="path to results directory",
                        type=str, default="")
    
    parser.add_argument("-e", dest="epochs",
                        help="number of epochs",
                        type=int, default=1000)
    
    parser.add_argument("-n", dest="n_model_train", 
                        type=int, default=20)
    
    parser.add_argument("-i", dest="init_models", 
                        type=bool, default=False)
    
    parser.add_argument("-lr", dest="lr", 
                        type=float, default=0.001)
    
    parser.add_argument("-f", dest="fold_xp", 
                        type=bool, default=False)
    return parser

def load_refl_angles(path_to_data_dir: str):
    """
    Loads simulated s2 reflectance angles and LAI from weiss dataset.
    """
    path_to_file = path_to_data_dir + "/InputNoNoise_2.csv"
    assert os.path.isfile(path_to_file)
    df_validation_data = pd.read_csv(path_to_file, sep=" ", engine="python")
    s2_r = df_validation_data[['B3', 'B4', 'B5', 'B6', 'B7', 'B8A', 'B11', 'B12']].values
    tts = np.rad2deg(np.arccos(df_validation_data['cos(thetas)'].values))
    tto = np.rad2deg(np.arccos(df_validation_data['cos(thetav)'].values))
    psi = np.rad2deg(np.arccos(df_validation_data['cos(phiv-phis)'].values))
    lai = df_validation_data['lai_true'].values
    return s2_r, tto, tts, psi, lai # Warning, inverted tto and tts w.r.t my prosil version

def load_weiss_dataset(path_to_data_dir: str):
    """
    Loads simulated s2 reflectance angles and LAI from weiss dataset as aggregated numpy arrays.
    """
    s2_r, tto, tts, psi, lai = load_refl_angles(path_to_data_dir)
    s2_a = np.stack((tto, tts, psi), 1)
    return s2_r, s2_a, lai

def sample_from_dist(input_samples: np.ndarray, n: int = 100, 
                     kernel_p: Callable | None = None, kernel_q: Callable | None = None):
    """
    Draw samples from a dataset with an original distribution, with a specified target distribution
    """
    if kernel_p is None:
        kernel_p = stats.gaussian_kde(input_samples)
    if kernel_q is None:
        input_samples_max = input_samples.max()
        kernel_q = lambda _ : 1 / input_samples_max
    sampled_idx = []
    q_x = 1 / input_samples.max()
    for i in range(n):
        sample = None
        while sample is None:
            i = np.random.randint(low=0,high=len(input_samples))
            input_samples_i = input_samples[i]
            q_x = kernel_q(input_samples_i)
            p_x = kernel_p(input_samples_i)
            if q_x > p_x:
                sample = i
            else:
                rand = np.random.rand()
                if rand < q_x / p_x:
                    sample = i
        sampled_idx.append(sample)
    return sampled_idx

def swap_sampling(samples: torch.Tensor, tgt_dist_samples: torch.Tensor, allow_doubles:bool=True):
    """
    Draw samples of an input dataset with a specified distribution, by finding the closest match to 
    target distribution.
    """
    samples_idx = []
    for tgt_sample in tgt_dist_samples:
        idx = torch.argmin((samples - tgt_sample).abs()).item()
        samples_idx.append(idx)
    samples_idx = torch.tensor(samples_idx)
    if not allow_doubles:
        samples_idx = torch.unique(samples_idx)
    return samples_idx

def swap_sampling_truncated_gaussians(samples:torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor,
                                      n_samples:int=1, allow_doubles:bool=True):
    """
    Draw samples of an input dataset with a specified distribution, by finding the closest match to 
    target distribution.
    """
    samples_idx = []
    for _ in range(n_samples):
        idx = 0
        while idx in samples_idx:
            tgt_sample = sample_truncated_gaussian(mu.reshape(1,1),
                                                   sigma.reshape(1,1),
                                                   n_samples=1,
                                                   lower=torch.tensor(0),
                                                                upper=torch.tensor(14))
            idx = torch.argmin((samples - tgt_sample).abs()).item()
            if allow_doubles:
                break
        samples_idx.append(idx)
    return samples_idx


def prepare_datasets(n_eval:int=5000, n_samples_sub:int=5000, save_dir:str="",
                     reduce_to_common_samples_nb:bool=True,
                     tg_mu: torch.Tensor=torch.tensor([0,4]), tg_sigma:torch.Tensor=torch.tensor([1,4]),
                     plot_dist:bool=False, s2_tensor_image_path:str = ""):
    """
    Prepare dataset for nn regression from weiss reflectances / lai dataset.
    """
    max_im_size = 256
    bands = torch.tensor([1,2,4,5,6,7,8,9])
    image_tensor = torch.load(s2_tensor_image_path)
    max_im_size = min(max_im_size, image_tensor.size(1), image_tensor.size(2))
    image_tensor = image_tensor[:,:max_im_size,:max_im_size]
    angles = torch.zeros(3, image_tensor.size(1),image_tensor.size(2))
    angles[1,...] = image_tensor[11,...]
    angles[0,...] = image_tensor[13,...] # inverted from my networks
    angles[2,...] = image_tensor[12,...] - image_tensor[14, ...]
    s2_r = image_tensor[bands,...].reshape(len(bands), -1).transpose(1,0)
    s2_a = angles.reshape(3, -1).transpose(1,0)
    with torch.no_grad():
        snap_nn = SnapNN(device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'), ver="3A")
        snap_nn.set_weiss_weights()
        snap_lai = snap_nn.forward(torch.cat((s2_r, np.cos(np.deg2rad(s2_a))), 1).to(snap_nn.device)).cpu()
        fig, ax = plt.subplots(dpi=150, figsize=(6,6))
        ax.imshow(rgb_render(image_tensor)[0])
        fig.savefig(save_dir + "/s2_data_image.png")
        fig, ax = plt.subplots(dpi=150, figsize=(6,6))
        ax.hist(snap_lai.squeeze().cpu().numpy(), density=True, bins=200)
        fig.savefig(save_dir + "/s2_data_laid_hist.png")

    data_s2 = torch.cat((s2_r, np.cos(np.deg2rad(s2_a)), snap_lai), 1)
    s2_r, s2_a, lai = load_weiss_dataset(os.path.join(prosailvae.__path__[0], os.pardir) + "/field_data/lai/")
    data_weiss = torch.from_numpy(np.concatenate((s2_r, np.cos(np.deg2rad(s2_a)), lai.reshape(-1,1)), 1))
    seed = 4567895683301
    g_cpu = torch.Generator()
    g_cpu.manual_seed(seed)
    idx = torch.randperm(len(lai), generator=g_cpu)
    data_eval_weiss = data_weiss[idx[:n_eval],:]
    data_eval_list = [data_eval_weiss, data_s2]
    if plot_dist:
        fig, ax = plt.subplots()
        ax.hist(data_eval_weiss[:,-1], bins=200, range=[0,14], density=True, label="evaluation samples")
        ax.plot(torch.arange(0,14,0.1), 
                truncated_gaussian_pdf(torch.arange(0,14,0.1), torch.tensor(2), torch.tensor(3),
                                       lower=torch.tensor(0), upper=torch.tensor(14)),
                'r', label='Original distribution')
        ax.set_xlabel("Eval data LAI distribution")
        ax.legend()
        fig.savefig(save_dir + "/eval_data_lai.png")
        plt.close('all')
    torch.save(data_eval_weiss, save_dir + "/eval_data_weiss.pth")
    data_train = data_weiss[idx[n_eval:],:]
    mu_ref = torch.tensor(2)
    sigma_ref = torch.tensor(3)
    list_params = []
    list_kl = []
    # list_kl_true = []
    list_idx_samples = []
    min_sample_nb = n_samples_sub
    # hist_data, bin_edges = np.histogram(data_train[:,-1], bins=200, range=[0, 14], density=True)
    for i, mu in enumerate(tg_mu):
        list_idx_samples_i = []
        for j, sigma in enumerate(tg_sigma):
            kl = kl_tntn(mu_ref, sigma_ref, mu, sigma, lower=torch.tensor(0.0),
                         upper=torch.tensor(14.0)).item()
            list_kl.append(kl)
            list_params.append([mu, sigma])
            # tgt_dist_samples = sample_truncated_gaussian(mu.reshape(1,1), sigma.reshape(1,1),
            #                                              n_samples=n_samples_sub*2,
            #                                              lower = torch.tensor(0),
            #                                              upper=torch.tensor(14))
            # idx_samples = swap_sampling(data_train[:,-1], tgt_dist_samples.squeeze(),
            #                             allow_doubles=False)
            idx_samples = swap_sampling_truncated_gaussians(data_train[:,-1], mu.reshape(1,1),
                                                            sigma.reshape(1,1),
                                                            n_samples=n_samples_sub,
                                                            allow_doubles=False)
            h, p = np.histogram(data_train[idx_samples,-1].squeeze(), bins=200,
                                range=[0,14], density=True)
            cum_hist = np.cumsum(h) * np.diff(p)[0]
            cdf_hist = truncated_gaussian_cdf(torch.from_numpy(p[:-1]), mu, sigma,
                                                lower=torch.tensor(0), upper=torch.tensor(14)).numpy()
            mse_at_hist = np.mean((cum_hist - cdf_hist)**2)
            print(f"number of samples for mu = {mu.item()} sigma = {sigma.item()} (kl = {kl:.2e}, "
                  f"sampling cdf mse: {mse_at_hist:.2e}) : {len(idx_samples)}")
            list_idx_samples_i.append(idx_samples)
            min_sample_nb = min(min_sample_nb, len(idx_samples))
            if plot_dist:
                fig, ax = plt.subplots(1, dpi=150)
                ax.plot(torch.arange(0,14,0.1),
                        truncated_gaussian_pdf(torch.arange(0,14,0.1), mu_ref, sigma_ref,
                                               lower=torch.tensor(0), upper=torch.tensor(14)),
                        'k', label='original distribution')
                ax.hist(data_train[idx_samples,-1].squeeze(), bins=200, range=[0,14], 
                        alpha=0.5, density=True, label='samples')
                ax.plot(torch.arange(0,14,0.1),
                        truncated_gaussian_pdf(torch.arange(0,14,0.1), mu, sigma,
                                               lower=torch.tensor(0), upper=torch.tensor(14)),
                        'r', label='sampling distribution (kl={:.2f})'.format(kl))
                ax.set_xlabel("LAI")
                ax.set_xlim([0,14])
                ax.legend()
                fig.savefig(save_dir + f"/samples_lai_mu_{mu.item()}_sigma_{sigma.item()}.png")

                fig, ax = plt.subplots(1, dpi=150)
                ax.plot(torch.arange(0,14,0.1),
                        truncated_gaussian_cdf(torch.arange(0,14,0.1), mu_ref, sigma_ref,
                                               lower=torch.tensor(0), upper=torch.tensor(14)),
                        'k', label='original distribution')
                ax.hist(data_train[idx_samples,-1].squeeze(), bins=200, 
                        range=[0,14], cumulative=True,histtype='step',
                        alpha=0.5, density=True, label='samples')
                ax.plot(torch.arange(0,14,0.1),
                        truncated_gaussian_cdf(torch.arange(0,14,0.1), mu, sigma,
                                               lower=torch.tensor(0), upper=torch.tensor(14)),
                        'r', label='sampling distribution (kl={:.2f})'.format(kl))
                ax.set_xlabel("LAI")
                ax.set_xlim([0,14])
                ax.legend()
                fig.savefig(save_dir + f"/cdf_samples_lai_mu_{mu.item()}_sigma_{sigma.item()}.png")
                plt.close('all')
        list_idx_samples.append(list_idx_samples_i)
    tg_data_list = []
    for i, mu in enumerate(tg_mu):
        for j, sigma in enumerate(tg_sigma):
            idx_samples = list_idx_samples[i][j]
            if reduce_to_common_samples_nb:
                idx_samples = idx_samples[:min_sample_nb]
            # hist_sub_data, _ = np.histogram(data_train[idx_samples,-1], bins=200, range=[0,14], density=True)
            # list_kl_true.append(numerical_kl_from_pdf(torch.from_numpy(hist_data).float(),
            #                                           torch.from_numpy(hist_sub_data).float(),
            #                                           dx=bin_edges[0]))
            torch.save(data_train[idx_samples, :], save_dir + f"/tg_lai_mu_{mu.item()}_sigma_{sigma.item()}_data.pth")
            tg_data_list.append(data_train[idx_samples, :])
    n_folds = len(data_train) // min_sample_nb
    idx_train = torch.randperm(len(data_train), generator=g_cpu)
    fold_data_list = []
    for i in range(n_folds):
        data_i = data_train[idx_train[i*min_sample_nb:(i+1)*min_sample_nb],:]
        torch.save(data_i, save_dir + f"/fold_{i}_data.pth")
        fold_data_list.append(data_i)
    return data_eval_list, fold_data_list, tg_data_list, list_kl, list_params #, list_kl_true




def get_loaders(data:torch.Tensor, seed:int=86294692001, valid_ratio:float=0.1, batch_size:int=256):
    """
    Get train and valid loader from input data
    """
    # data = torch.load(filepath)
    g_cpu = torch.Generator()
    g_cpu.manual_seed(seed)
    n_valid = int(valid_ratio * data.size(0))
    idx = torch.randperm(data.size(0), generator=g_cpu)
    data_valid = data[idx[:n_valid],:].float()
    data_train = data[idx[n_valid:],:].float()
    train_dataset = TensorDataset(data_train[:,:-1], data_train[:,-1].unsqueeze(1))
    valid_dataset = TensorDataset(data_valid[:,:-1], data_valid[:,-1].unsqueeze(1))

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                              num_workers=0, shuffle=True)
    if n_valid > 0:
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size,
                                  num_workers=0, shuffle=True)
    else:
        valid_loader = None
    return train_loader, valid_loader

def get_model_metrics(test_data, model, all_valid_losses=[]):
    """
    """
    with torch.no_grad():
        if len(all_valid_losses) == 0 :
            all_valid_losses = [100000]
        lai_pred = model.forward(test_data[0].to(model.device)).cpu()
        lai_true = test_data[1].cpu()
        rmse = (lai_pred - lai_true).pow(2).mean().sqrt().item()
        r2 = r2_score(lai_true.squeeze().numpy(), lai_pred.squeeze().numpy())
        mae = (lai_pred - lai_true).abs().mean().item()
        reg_m, reg_b = np.polyfit(lai_true.squeeze().numpy(), lai_pred.squeeze().numpy(), 1)
        best_valid_loss = min(all_valid_losses)

    return torch.tensor([rmse, r2, mae, reg_m, reg_b, best_valid_loss])

def get_n_model_metrics(train_loader, valid_loader, test_loader_list:List|None=None,
                        n_models:int=10, epochs:int=500, lr:float=0.001,
                        disable_tqdm:bool=False,  patience:int=10, init_models:bool=False, ver:str="2.1"):
    """
    Trains several models on given train and validation dataloaders and assesses their regression metrics
    on provided test dataloader
    """
    if test_loader_list is None:
        raise ValueError("Please input a list of test loaders")
    metrics_names=["rmse", "r2", "mae", "reg_m", "reg_b", "best_valid_loss"]
    metrics = torch.zeros((n_models, len(test_loader_list), len(metrics_names)))
    for i in range(n_models):
        snap_nn = SnapNN(device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'), ver=ver)
        if init_models:
            snap_nn.set_weiss_weights()
        optimizer = optim.Adam(snap_nn.parameters(), lr=lr)
        lr_scheduler = ReduceLROnPlateau(optimizer=optimizer, patience=patience,
                                         threshold=0.001)
        _, all_valid_losses, _ = snap_nn.train_model(train_loader, valid_loader, optimizer,
                                                     epochs=epochs, lr_scheduler=lr_scheduler,
                                                     disable_tqdm=disable_tqdm)
        for j, test_loader in enumerate(test_loader_list):
            with torch.no_grad():
                test_data = test_loader.dataset[:]
                metrics[i,j,:] = get_model_metrics(test_data, snap_nn, all_valid_losses=all_valid_losses)
    return metrics

def get_pixel_log_likelihood_with_weiss(s2_r):
    from sklearn.mixture import GaussianMixture
    s2_r_ref, _, _ = load_weiss_dataset(os.path.join(prosailvae.__path__[0], os.pardir) + "/field_data/lai/")
    gm = GaussianMixture(n_components=8, random_state=0).fit(s2_r_ref)
    return gm.score_samples(s2_r)


def main():
    test_snap_nn(ver="2.1")
    test_snap_nn(ver="3A")
    test_snap_nn(ver="3B")
    
    if socket.gethostname()=='CELL200973':
        args=["-d", "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/snap_validation_data/",
              "-r", "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/results/snap_validation/",
              "-e", "3",
              "-n", "2",
              "-i", 't',
              "-lr", "0.001", 
              "-f", "t"]
        disable_tqdm=False
        # tg_mu = torch.tensor([0,1])
        # tg_sigma = torch.tensor([0.5,1])
        tg_mu = torch.tensor([0,1,2,3,4,5])
        tg_sigma = torch.tensor([0.5,1,2,3,4])
        parser = get_parser().parse_args(args)
        s2_tensor_image_path = "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/torch_files/T31TCJ/after_SENTINEL2B_20171127-105827-648_L2A_T31TCJ_C_V2-2_roi_0.pth"  
        ver="3B"
    else:
        parser = get_parser().parse_args()

        tg_mu = torch.tensor([0, 1, 2, 3, 4])
        tg_sigma = torch.tensor([0.5, 1, 2, 3])
        TENSOR_DIR="/work/CESBIO/projects/MAESTRIA/prosail_validation/validation_sites/torchfiles/T31TCJ/"
        image_filename = "/before_SENTINEL2A_20180620-105211-086_L2A_T31TCJ_C_V2-2_roi_0.pth"
        ver="3A"
        s2_tensor_image_path = TENSOR_DIR + image_filename
        disable_tqdm=True
    init_models = parser.init_models
    prepare_data = True
    epochs = parser.epochs
    n_models = parser.n_model_train
    compute_metrics = True
    save_dir = parser.data_dir
    res_dir = parser.results_dir
    lr = parser.lr
    if not os.path.isdir(res_dir):
        os.makedirs(res_dir)
    if prepare_data:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        else:
            for filename in os.listdir(save_dir):
                file_path = os.path.join(save_dir, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as exc:
                    print('Failed to delete %s. Reason: %s' % (file_path, exc))
        (data_eval_list, fold_data_list, tg_data_list,
         list_kl, list_params) = prepare_datasets(n_eval=5000, n_samples_sub=5000,
                                                  save_dir=save_dir,  tg_mu = tg_mu,
                                                  tg_sigma = tg_sigma,
                                                  plot_dist=True,
                                                  s2_tensor_image_path=s2_tensor_image_path)
    fold_xp = parser.fold_xp
    if fold_xp:
        metrics_names = ["rmse", "r2", "mae", "reg_m", "reg_b", "best_valid_loss"]
        eval_data_name = ["simulated_data", "snap_s2"]
        snap_ref = SnapNN(device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'), ver=ver)
        snap_ref.set_weiss_weights()
        test_loader_list = []
        all_metrics_ref = []
        for k, data_eval in enumerate(data_eval_list):
            test_loader, _ = get_loaders(data_eval, seed=86294692001, valid_ratio=0, batch_size=256)
            test_loader_list.append(test_loader)
            metrics_ref = []
            for i, fold_data in enumerate(fold_data_list):
                _, valid_loader = get_loaders(fold_data, seed=86294692001, valid_ratio=0.1,
                                              batch_size=256)
                snap_valid_loss = snap_ref.validate(valid_loader)
                metrics_ref.append(get_model_metrics(test_loader.dataset[:],
                                                     model=snap_ref,
                                                     all_valid_losses=[snap_valid_loss]))
            metrics_ref = torch.stack(metrics_ref, dim=0)
            all_metrics_ref.append(metrics_ref)
            loader_pixels_ll = get_pixel_log_likelihood_with_weiss(test_loader.dataset[:][0][:,:8].numpy())
            with torch.no_grad():
                absolute_errors = (snap_ref.forward((test_loader.dataset[:][0]
                                                    - test_loader.dataset[:][1]).to(snap_ref.device))).abs().squeeze().cpu().numpy()
            fig, ax = plt.subplots()
            ax.scatter(loader_pixels_ll, absolute_errors, s=0.5)
            ax.set_xlabel("Reflectances log-likelihood from the simulated dataset distribution")
            ax.set_ylabel("LAI Absolute error")
            fig.savefig(res_dir + f"/scatter_error_vs_loglikelihood_{eval_data_name[k]}.png")
            fig, ax = plt.subplots()
            ax.hist2d(loader_pixels_ll, absolute_errors, bins=100)
            ax.set_xlabel("Reflectances log-likelihood from the simulated dataset distribution")
            ax.set_ylabel("LAI Absolute error")
            fig.savefig(res_dir + f"/hist_error_vs_loglikelihood_{eval_data_name[k]}.png")
        metrics_ref = torch.stack(all_metrics_ref, dim=0).transpose(1,0)

        if compute_metrics:
            mean_metrics = []
            all_metrics = []
            for i, fold_data in enumerate(fold_data_list):
                train_loader, valid_loader = get_loaders(fold_data, seed=86294692001,
                                                         valid_ratio=0.1, batch_size=256)
                metrics = get_n_model_metrics(train_loader, valid_loader,
                                              test_loader_list=test_loader_list,
                                              n_models=n_models, epochs=epochs, lr=lr,
                                              disable_tqdm=disable_tqdm, patience=20,
                                              init_models=init_models, ver=ver)
                mean_metrics.append(metrics.mean(0).unsqueeze(0))
                all_metrics.append(metrics.unsqueeze(0))
            mean_metrics = torch.cat(mean_metrics, 0)
            all_metrics = torch.cat(all_metrics, 0)
            torch.save(all_metrics, res_dir + "/all_metrics.pth")
        else:
            all_metrics = torch.load(res_dir + "/all_metrics.pth")
            mean_metrics = all_metrics.mean(1)
        median_metrics = torch.quantile(all_metrics,0.5,dim=1)
        if not os.path.isdir(res_dir):
            os.makedirs(res_dir)
        for j in range(mean_metrics.size(1)): # dataset
            for k in range(mean_metrics.size(2)): # Métrique
                fig, ax = plt.subplots(1, dpi=150, tight_layout=True)
                ax.hist(all_metrics[:,:,j,k].reshape(-1), bins=100, density=True, label='trained models')
                ax.set_xlabel(f"{metrics_names[k]} histogram")
                if metrics_ref[:,j,k].min() == metrics_ref[:,j,k].max():
                    ax.axvline(metrics_ref[:,j,k].min(), c='k', label=f'SNAP {metrics_names[k]}')
                else:
                    ax.hist(metrics_ref[:,j,k], label=f'SNAP {metrics_names[k]}', bins=50, color='k')
                ax.legend()
                fig.savefig(res_dir + f"/{eval_data_name[j]}_{metrics_names[k]}_histogram.png")

                fig, ax = plt.subplots(1, dpi=150, tight_layout=True)
                ax.scatter(np.arange(len(fold_data_list)), mean_metrics[:,j,k])
                if metrics_ref[:,j,k].min() == metrics_ref[:,j,k].max():
                    ax.axhline(metrics_ref[:,j,k].min(), c='k', label=f'SNAP {metrics_names[k]}')
                else:
                    ax.scatter(np.arange(len(fold_data_list)), metrics_ref[:,j,k],
                               label=f'SNAP {metrics_names[k]}', c='k')
                ax.set_xlabel("Fold number")
                ax.set_ylabel(metrics_names[k])
                fig.savefig(res_dir + f"/means_{eval_data_name[j]}_{metrics_names[k]}_vs_fold.png")
                        
                fig, ax = plt.subplots(1, dpi=150, tight_layout=True)
                ax.scatter(np.arange(len(fold_data_list)), median_metrics[:,j,k])
                if metrics_ref[:,j,k].min() == metrics_ref[:,j,k].max():
                    ax.axhline(metrics_ref[:,j,k].min(), c='k', label=f'SNAP {metrics_names[k]}')
                else:
                    ax.scatter(np.arange(len(fold_data_list)), metrics_ref[:,j,k],
                               label=f'SNAP {metrics_names[k]}', c='k')
                ax.set_xlabel("Fold number")
                ax.set_ylabel(metrics_names[k])
                fig.savefig(res_dir + f"/median_{eval_data_name[j]}_{metrics_names[k]}_vs_fold.png")
                
                fig, ax = plt.subplots(1, dpi=150, tight_layout=True)
                ax.boxplot(all_metrics[:,:,j,k], positions=np.arange(len(fold_data_list)), widths=0.1)
                if metrics_ref[:,j,k].min() == metrics_ref[:,j,k].max():
                    ax.axhline(metrics_ref[:,j,k].min(), c='k', label=f'SNAP {metrics_names[k]}')
                else:
                    ax.scatter(np.arange(len(fold_data_list)), metrics_ref[:,j,k],
                               label=f'SNAP {metrics_names[k]}', c='k')
                ax.set_xticks(np.arange(0,11), np.arange(0,11))
                ax.set_xlabel("Fold number")
                ax.set_ylabel(metrics_names[k])
                fig.savefig(res_dir + f"/boxplot_{eval_data_name[j]}_{metrics_names[k]}_vs_fold.png")

                fig, ax = plt.subplots(1, dpi=150, tight_layout=True)
                ax.scatter(torch.arange(len(fold_data_list)).repeat(all_metrics.size(1),1).transpose(1,0).reshape(-1),
                           all_metrics[:,:,j,k].reshape(-1), s=0.5)
                if metrics_ref[:,j,k].min() == metrics_ref[:,j,k].max():
                    ax.axhline(metrics_ref[:,j,k].min(), c='k', label=f'SNAP {metrics_names[k]}')
                else:
                    ax.scatter(np.arange(len(fold_data_list)), metrics_ref[:,j,k],
                               label=f'SNAP {metrics_names[k]}', c='k')
                ax.set_xlabel("Fold Number")
                ax.set_ylabel(metrics_names[k])
                fig.savefig(res_dir + f"/scatter_{eval_data_name[j]}_{metrics_names[k]}_vs_fold.png")
                plt.close('all')

    else:
        metrics_names = ["rmse", "r2", "mae", "reg_m", "reg_b", "best_valid_loss"]
        eval_data_name = ["simulated_data", "snap_s2"]
        snap_ref = SnapNN(device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        snap_ref.set_weiss_weights()
        test_loader_list = []
        all_metrics_ref = []
        for _, data_eval in enumerate(data_eval_list):
            test_loader, _ = get_loaders(data_eval, seed=86294692001,
                                         valid_ratio=0, batch_size=256)
            test_loader_list.append(test_loader)
            metrics_ref = []
            for i, tg_data in enumerate(tg_data_list):
                _, valid_loader = get_loaders(tg_data, seed=86294692001,
                                              valid_ratio=0.1, batch_size=256)
                snap_valid_loss = snap_ref.validate(valid_loader)
                metrics_ref.append(get_model_metrics(test_loader.dataset[:], model=snap_ref,
                                                     all_valid_losses=[snap_valid_loss]))
            metrics_ref = torch.stack(metrics_ref, dim=0)
            all_metrics_ref.append(metrics_ref)
        metrics_ref = torch.stack(all_metrics_ref, dim=0).transpose(1,0)
        if compute_metrics:
            kl = torch.zeros(len(tg_data_list))
            mean_metrics = []
            all_metrics = []
            for i, tg_data in enumerate(tg_data_list):
                kl[i] = list_kl[i]
                train_loader, valid_loader = get_loaders(tg_data, seed=86294692001, valid_ratio=0.1,
                                                         batch_size=256)
                metrics = get_n_model_metrics(train_loader, valid_loader,
                                              test_loader_list=test_loader_list,
                                              n_models=n_models, epochs=epochs,
                                              lr=lr,disable_tqdm=disable_tqdm, patience=20,
                                              init_models=init_models)
                mean_metrics.append(metrics.mean(0).unsqueeze(0))
                all_metrics.append(metrics.unsqueeze(0))
            mean_metrics = torch.cat(mean_metrics, 0)
            all_metrics = torch.cat(all_metrics, 0)
            torch.save(all_metrics, res_dir + "/all_metrics.pth")
            torch.save(kl, res_dir + "/kl.pth")
        else:
            all_metrics = torch.load(res_dir + "/all_metrics.pth")
            mean_metrics = all_metrics.mean(1)

            kl = torch.load(res_dir + "/kl.pth")
        median_metrics = torch.quantile(all_metrics,0.5,dim=1)
        kl_res_dir = res_dir + "/kl/"
        if not os.path.isdir(kl_res_dir):
            os.makedirs(kl_res_dir)

        for j in range(mean_metrics.size(1)):
            fig, ax = plt.subplots(1, dpi=150, tight_layout=True)
            sc = ax.scatter(all_metrics[:,:,j,5], all_metrics[:,:,j,1],
                            c=kl.repeat(all_metrics.size(1),1).transpose(1,0).reshape(-1), s=1)
            plt.colorbar(sc, ax=ax, 
                         label='kl divergence between evaluation dataset and training dataset')
            ax.set_xlabel("Validation loss")
            ax.set_ylabel(f"r2 score on evaluation dataset {eval_data_name[j]}")
            plt.legend()
            fig.savefig(res_dir + f"/{eval_data_name[j]}_valid_loss_vs_r2.png")

            for k in range(mean_metrics.size(2)):
                fig, ax = plt.subplots(1, dpi=150, tight_layout=True)
                ax.hist(all_metrics[:,:,j,k].reshape(-1), bins=100, density=True, label='trained models')
                ax.set_xlabel(f"{metrics_names[k]} histogram")
                if metrics_ref[:,j,k].min() == metrics_ref[:,j,k].max():
                    ax.axvline(metrics_ref[:,j,k].min(), c='k', label=f'SNAP {metrics_names[k]}')
                else:
                    ax.hist(metrics_ref[:,j,k], label=f'SNAP {metrics_names[k]}', bins=50, color='k')
                ax.legend()
                fig.savefig(res_dir + f"/{eval_data_name[j]}_{metrics_names[k]}_histogram.png")

                fig, ax = plt.subplots(1, dpi=150, tight_layout=True)
                ax.scatter(kl, mean_metrics[:,j,k])
                if metrics_ref[:,j,k].min() == metrics_ref[:,j,k].max():
                    ax.axhline(metrics_ref[:,j,k].min(), c='k', label=f'SNAP {metrics_names[k]}')
                else:
                    ax.scatter(kl, metrics_ref[:,j,k], label=f'SNAP {metrics_names[k]}', c='k')
                ax.set_xlabel("KL distance between distribution of lai in training and evaluation dataset")
                ax.set_ylabel(metrics_names[k])
                ax.set_xscale('symlog')
                fig.savefig(kl_res_dir + f"/means_{eval_data_name[j]}_{metrics_names[k]}_vs_kl.png")
                        
                fig, ax = plt.subplots(1, dpi=150, tight_layout=True)
                ax.scatter(kl, median_metrics[:,j,k])
                if metrics_ref[:,j,k].min() == metrics_ref[:,j,k].max():
                    ax.axhline(metrics_ref[:,j,k].min(), c='k', label=f'SNAP {metrics_names[k]}')
                else:
                    ax.scatter(kl, metrics_ref[:,j,k], label=f'SNAP {metrics_names[k]}', c='k')
                ax.set_xlabel("KL distance between distribution of lai in training and evaluation dataset")
                ax.set_ylabel(metrics_names[k])
                ax.set_xscale('symlog')
                fig.savefig(kl_res_dir + f"/median_{eval_data_name[j]}_{metrics_names[k]}_vs_kl.png")
                
                fig, ax = plt.subplots(1, dpi=150, tight_layout=True)
                ax.boxplot(all_metrics[:,:,j,k], positions=kl, widths=0.1)
                if metrics_ref[:,j,k].min() == metrics_ref[:,j,k].max():
                    ax.axhline(metrics_ref[:,j,k].min(), c='k', label=f'SNAP {metrics_names[k]}')
                else:
                    ax.scatter(kl, metrics_ref[:,j,k], label=f'SNAP {metrics_names[k]}', c='k')
                ax.set_xticks(np.arange(0,11), np.arange(0,11))
                ax.set_xlabel("KL distance between distribution of lai in training and evaluation dataset")
                ax.set_ylabel(metrics_names[k])
                ax.set_xscale('symlog')
                fig.savefig(kl_res_dir + f"/boxplot_{eval_data_name[j]}_{metrics_names[k]}_vs_kl.png")

                fig, ax = plt.subplots(1, dpi=150, tight_layout=True)
                ax.scatter(kl.repeat(all_metrics.size(1),1).transpose(1,0).reshape(-1),
                           all_metrics[:,:,j,k].reshape(-1), s=0.5)
                if metrics_ref[:,j,k].min() == metrics_ref[:,j,k].max():
                    ax.axhline(metrics_ref[:,j,k].min(), c='k', label=f'SNAP {metrics_names[k]}')
                else:
                    ax.scatter(kl, metrics_ref[:,j,k], label=f'SNAP {metrics_names[k]}', c='k')
                ax.set_xlabel("KL distance between distribution of lai in training and evaluation dataset")
                ax.set_ylabel(metrics_names[k])
                ax.set_xscale('symlog')
                fig.savefig(kl_res_dir + f"/scatter_{eval_data_name[j]}_{metrics_names[k]}_vs_kl.png")
                plt.close('all')

        size_mu = len(tg_mu)
        size_sigma = len(tg_sigma)  
        tg_mu_rep = tg_mu.repeat(size_sigma,1).transpose(0,1).reshape(-1)     
        tg_sigma_rep = tg_sigma.repeat(size_mu,1).reshape(-1)     
        mu_res_dir = res_dir + "/mu/"
        if not os.path.isdir(mu_res_dir):
            os.makedirs(mu_res_dir)
        for j in range(mean_metrics.size(1)):
            for k in range(mean_metrics.size(2)):
                fig, ax = plt.subplots(1, dpi=150, tight_layout=True)
                ax.scatter(tg_mu_rep, mean_metrics[:,j,k])
                if metrics_ref[:,j,k].min() == metrics_ref[:,j,k].max():
                    ax.axhline(metrics_ref[:,j,k].min(), c='k', label=f'SNAP {metrics_names[k]}')
                else:
                    ax.scatter(tg_mu_rep, metrics_ref[:,j,k], label=f'SNAP {metrics_names[k]}', c='k')
                ax.set_xlabel("sampling distribution mu")
                ax.set_ylabel(metrics_names[k])
                ax.set_xscale('symlog')
                fig.savefig(mu_res_dir + f"/means_{eval_data_name[j]}_{metrics_names[k]}_vs_mu.png")
                        
                fig, ax = plt.subplots(1, dpi=150, tight_layout=True)
                ax.scatter(tg_mu_rep, median_metrics[:,j,k])
                if metrics_ref[:,j,k].min() == metrics_ref[:,j,k].max():
                    ax.axhline(metrics_ref[:,j,k].min(), c='k', label=f'SNAP {metrics_names[k]}')
                else:
                    ax.scatter(tg_mu_rep, metrics_ref[:,j,k], label=f'SNAP {metrics_names[k]}', c='k')
                ax.set_xlabel("sampling distribution mu")
                ax.set_ylabel(metrics_names[k])
                ax.set_xscale('symlog')
                fig.savefig(mu_res_dir + f"/median_{eval_data_name[j]}_{metrics_names[k]}_vs_mu.png")
                
                fig, ax = plt.subplots(1, dpi=150, tight_layout=True)
                ax.boxplot(all_metrics[:,:,j,k], positions=tg_mu_rep, widths=0.1)
                if metrics_ref[:,j,k].min() == metrics_ref[:,j,k].max():
                    ax.axhline(metrics_ref[:,j,k].min(), c='k', label=f'SNAP {metrics_names[k]}')
                else:
                    ax.scatter(tg_mu_rep, metrics_ref[:,j,k], label=f'SNAP {metrics_names[k]}', c='k')
                ax.set_xticks(np.arange(0,11), np.arange(0,11))
                ax.set_xlabel("sampling distribution mu")
                ax.set_ylabel(metrics_names[k])
                ax.set_xscale('symlog')
                fig.savefig(mu_res_dir + f"/boxplot_{eval_data_name[j]}_{metrics_names[k]}_vs_mu.png")

                fig, ax = plt.subplots(1, dpi=150, tight_layout=True)
                ax.scatter(tg_mu_rep.repeat(all_metrics.size(1),1).transpose(1,0).reshape(-1), all_metrics[:,:,j,k].reshape(-1), s=0.5)
                if metrics_ref[:,j,k].min() == metrics_ref[:,j,k].max():
                    ax.axhline(metrics_ref[:,j,k].min(), c='k', label=f'SNAP {metrics_names[k]}')
                else:
                    ax.scatter(tg_mu_rep, metrics_ref[:,j,k], label=f'SNAP {metrics_names[k]}', c='k')
                ax.set_xlabel("sampling distribution mu")
                ax.set_ylabel(metrics_names[k])
                ax.set_xscale('symlog')
                fig.savefig(mu_res_dir + f"/scatter_{eval_data_name[j]}_{metrics_names[k]}_vs_mu.png")
                plt.close('all')

        sigma_res_dir = res_dir + "/sigma/"
        if not os.path.isdir(sigma_res_dir):
            os.makedirs(sigma_res_dir)
        for j in range(mean_metrics.size(1)):
            for k in range(mean_metrics.size(2)):
                fig, ax = plt.subplots(1, dpi=150, tight_layout=True)
                ax.scatter(tg_sigma_rep, mean_metrics[:,j,k])
                if metrics_ref[:,j,k].min() == metrics_ref[:,j,k].max():
                    ax.axhline(metrics_ref[:,j,k].min(), c='k', label=f'SNAP {metrics_names[k]}')
                else:
                    ax.scatter(tg_sigma_rep, metrics_ref[:,j,k], label=f'SNAP {metrics_names[k]}', c='k')
                ax.set_xlabel("sampling distribution sigma")
                ax.set_ylabel(metrics_names[k])
                ax.set_xscale('symlog')
                fig.savefig(sigma_res_dir + f"/means_{eval_data_name[j]}_{metrics_names[k]}_vs_sigma.png")
                        
                fig, ax = plt.subplots(1, dpi=150, tight_layout=True)
                ax.scatter(tg_sigma_rep, median_metrics[:,j,k])
                if metrics_ref[:,j,k].min() == metrics_ref[:,j,k].max():
                    ax.axhline(metrics_ref[:,j,k].min(), c='k', label=f'SNAP {metrics_names[k]}')
                else:
                    ax.scatter(tg_sigma_rep, metrics_ref[:,j,k], label=f'SNAP {metrics_names[k]}', c='k')
                ax.set_xlabel("sampling distribution sigma")
                ax.set_ylabel(metrics_names[k])
                ax.set_xscale('symlog')
                fig.savefig(sigma_res_dir + f"/median_{eval_data_name[j]}_{metrics_names[k]}_vs_sigma.png")
                
                fig, ax = plt.subplots(1, dpi=150, tight_layout=True)
                ax.boxplot(all_metrics[:,:,j,k], positions=tg_sigma_rep, widths=0.1)
                if metrics_ref[:,j,k].min() == metrics_ref[:,j,k].max():
                    ax.axhline(metrics_ref[:,j,k].min(), c='k', label=f'SNAP {metrics_names[k]}')
                else:
                    ax.scatter(tg_sigma_rep, metrics_ref[:,j,k], label=f'SNAP {metrics_names[k]}', c='k')
                ax.set_xticks(np.arange(0,11), np.arange(0,11))
                ax.set_xlabel("sampling distribution sigma")
                ax.set_ylabel(metrics_names[k])
                ax.set_xscale('symlog')
                fig.savefig(sigma_res_dir + f"/boxplot_{eval_data_name[j]}_{metrics_names[k]}_vs_sigma.png")

                fig, ax = plt.subplots(1, dpi=150, tight_layout=True)
                ax.scatter(tg_sigma_rep.repeat(all_metrics.size(1),1).transpose(1,0).reshape(-1), 
                           all_metrics[:,:,j,k].reshape(-1), s=0.5)
                if metrics_ref[:,j,k].min() == metrics_ref[:,j,k].max():
                    ax.axhline(metrics_ref[:,j,k].min(), c='k', label=f'SNAP {metrics_names[k]}')
                else:
                    ax.scatter(tg_sigma_rep, metrics_ref[:,j,k], label=f'SNAP {metrics_names[k]}', c='k')
                ax.set_xlabel("sampling distribution sigma")
                ax.set_ylabel(metrics_names[k])
                ax.set_xscale('symlog')
                fig.savefig(sigma_res_dir + f"/scatter_{eval_data_name[j]}_{metrics_names[k]}_vs_sigma.png")
                plt.close('all')
    return

if __name__=="__main__":
    main()
