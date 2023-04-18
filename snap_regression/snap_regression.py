import os
import prosailvae
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import torch
from math import pi
from prosailvae.dist_utils import sample_truncated_gaussian, kl_tntn
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from tqdm import trange
from sklearn.metrics import r2_score
import argparse
import socket


def get_prosailvae_results_parser():
    """
    Creates a new argument parser.
    """
    parser = argparse.ArgumentParser(description='Parser for data generation')

    parser.add_argument("-d", dest="data_dir",
                        help="path to data direcotry",
                        type=str, default="/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/")
    
    parser.add_argument("-r", dest="results_dir",
                        help="path to root results direcotry",
                        type=str, default="")
    return parser

def load_refl_angles(path_to_data_dir):
    path_to_file = path_to_data_dir + "/InputNoNoise_2.csv"
    assert os.path.isfile(path_to_file)
    df_validation_data = pd.read_csv(path_to_file, sep=" ", engine="python")
    s2_r = df_validation_data[['B3', 'B4', 'B5', 'B6', 'B7', 'B8A', 'B11', 'B12']].values
    tts = np.rad2deg(np.arccos(df_validation_data['cos(thetas)'].values))
    tto = np.rad2deg(np.arccos(df_validation_data['cos(thetav)'].values))
    psi = np.rad2deg(np.arccos(df_validation_data['cos(phiv-phis)'].values))
    lai = df_validation_data['lai_true'].values
    return s2_r, tts, tto, psi, lai

def load_weiss_dataset(path_to_data_dir):
    s2_r, tts, tto, psi, lai = load_refl_angles(path_to_data_dir)
    s2_a = np.stack((tts,tto,psi),1)
    return s2_r, s2_a, lai

def sample_from_dist(lai, n=100, kernel_p=None, kernel_q=None):
    if kernel_p is None:
        kernel_p = stats.gaussian_kde(lai)
    if kernel_q is None:
        lai_max = lai.max()
        kernel_q = lambda _ : 1/lai_max
    sampled_idx = []
    q_x = 1 / lai.max()
    for i in range(n):
        sample = None
        while sample is None:
            i = np.random.randint(low=0,high=len(lai))
            lai_i = lai[i]
            q_x = kernel_q(lai_i)
            p_x = kernel_p(lai_i)
            if q_x > p_x:
                sample = i
            else:
                rand = np.random.rand()
                if rand < q_x / p_x:
                    sample = i
        sampled_idx.append(sample)
    return sampled_idx

def swap_sampling(samples, tgt_dist_samples, allow_doubles=True):
    samples_idx = []
    for i in range(len(tgt_dist_samples)):
        tgt_sample_i = tgt_dist_samples[i]
        idx = torch.argmin((samples - tgt_sample_i).abs()).item()
        samples_idx.append(idx)
    samples_idx = torch.tensor(samples_idx)
    if not allow_doubles:
        samples_idx = torch.unique(samples_idx)
    return samples_idx

def normalize(unnormalized, min_sample, max_sample): 
    return 2 * (unnormalized - min_sample) / (max_sample - min_sample) - 1

def denormalize(normalized, min_sample, max_sample): 
    return 0.5 * (normalized + 1) * (max_sample - min_sample) + min_sample

def prepare_datasets(n_eval=5000, n_samples_sub=5000, save_dir="", reduce_to_common_samples_nb=True, tg_mu = torch.tensor([0,4]),
                    tg_sigma = torch.tensor([1,4])):
    s2_r, s2_a, lai = load_weiss_dataset(os.path.join(prosailvae.__path__[0], os.pardir) + "/field_data/lai/")
    data = torch.from_numpy(np.concatenate((s2_r,np.cos(np.deg2rad(s2_a)),lai.reshape(-1,1)), 1))
    seed = 4567895683301
    g_cpu = torch.Generator()
    g_cpu.manual_seed(seed)
    idx = torch.randperm(len(lai), generator=g_cpu)
    data_eval = data[idx[:n_eval],:]
    torch.save(data_eval, save_dir + f"/eval_data.pth")
    data_train = data[idx[n_eval:],:]
    n_folds = len(data_train) // 5000
    idx_train = torch.randperm(len(data_train), generator=g_cpu)
    fold_data_list = []
    for i in range(n_folds):
        data_i = data_train[idx_train[i*n_samples_sub:(i+1)*n_samples_sub],:]
        torch.save(data_i, save_dir + f"/fold_{i}_data.pth")
        fold_data_list.append(data_i)
    mu_ref = torch.tensor(2)
    sigma_ref = torch.tensor(3)
    list_params = []
    list_kl = []
    list_idx_samples = []
    min_sample_nb = n_samples_sub
    for i in range(len(tg_mu)):
        
        list_idx_samples_i = []
        for j in range(len(tg_sigma)):
            kl = kl_tntn(mu_ref, sigma_ref, tg_mu[i], tg_sigma[j], lower=torch.tensor(0.0), upper=torch.tensor(14.0)).item()
            list_kl.append(kl)
            list_params.append([tg_mu[i], tg_sigma[j]])
            tgt_dist_samples = sample_truncated_gaussian(tg_mu[i].reshape(1,1), tg_sigma[j].reshape(1,1), n_samples=n_samples_sub, 
                                                        lower = torch.tensor(0), upper=torch.tensor(14))
            idx_samples = swap_sampling(data_train[:,-1], tgt_dist_samples.squeeze(), allow_doubles=False)
            print(f"number of samples for mu = {tg_mu[i].item()} sigma = {tg_sigma[j].item()} (kl = {kl}) : {len(idx_samples)}")
            list_idx_samples_i.append(idx_samples)
            min_sample_nb = min(min_sample_nb, len(idx_samples))
        
        list_idx_samples.append(list_idx_samples_i)
    tg_data_list = []
    for i in range(len(tg_mu)):
        for j in range(len(tg_sigma)):
            if reduce_to_common_samples_nb:
                idx_samples = idx_samples[:min_sample_nb]
            torch.save(data_train[idx_samples, :], save_dir + f"/tg_lai_mu_{tg_mu[i].item()}_sigma_{tg_sigma[j].item()}_data.pth")
            tg_data_list.append(data_train[idx_samples, :])
    return data_eval, fold_data_list, tg_data_list, list_kl, list_params

def get_norm_factors(ver='2.1'):
    if ver == "2.1":
        input_norm = torch.tensor([ [0,                   0.253061520472],
                                    [0,                   0.290393577911],
                                    [0,                   0.305398915249],
                                    [0.00663797254225,    0.608900395798],
                                    [0.0139727270189,     0.753827384323],
                                    [0.0266901380821,     0.782011770669],
                                    [0.0163880741923,     0.493761397883],
                                    [0,                   0.49302598446 ],
                                    [0.918595400582,      0.999999999991],
                                    [0.342022871159,      0.936206429175],
                                    [-0.999999982118,     0.99999999891 ]])
        input_min = input_norm[:,0]
        input_max = input_norm[:,1]
        lai_min = torch.tensor(0.000319182538301)
        lai_max = torch.tensor(14.4675094548)
    else:
        raise NotImplementedError
    return input_min, input_max, lai_min, lai_max

def get_loaders(data, seed=86294692001, valid_ratio=0.1, batch_size=256):
    # data = torch.load(filepath)
    g_cpu = torch.Generator()
    g_cpu.manual_seed(seed)
    n_valid = int(valid_ratio * data.size(0)) 
    idx = torch.randperm(data.size(0), generator=g_cpu)
    data_valid = data[idx[:n_valid],:].float()
    data_train = data[idx[n_valid:],:].float()
    train_dataset = TensorDataset(data_train[:,:-1], data_train[:,-1].unsqueeze(1))
    valid_dataset = TensorDataset(data_valid[:,:-1], data_valid[:,-1].unsqueeze(1))
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=0, shuffle=True)
    if n_valid > 0:
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, num_workers=0, shuffle=True)
    else:
        valid_loader = None
    return train_loader, valid_loader

class SnapNN(nn.Module):
    def __init__(self, device='cpu', ver="2.1"): 
        super().__init__()
        input_min, input_max, lai_min, lai_max = get_norm_factors(ver=ver)
        input_size = len(input_max) # 8 bands + 3 angles
        hidden_layer_size = 5
        layers = [nn.Linear(in_features=input_size, out_features=hidden_layer_size),   
                  nn.Tanh(), nn.Linear(in_features=hidden_layer_size, out_features=1)]
        self.input_min = input_min.to(device)
        self.input_max = input_max.to(device)
        self.lai_min = lai_min.to(device)
        self.lai_max = lai_max.to(device)
        self.net = nn.Sequential(*layers).to(device)
        self.device = device
        
    def forward(self, x):
        x_norm = normalize(x, self.input_min, self.input_max)
        lai_norm = self.net(x_norm)
        lai = denormalize(lai_norm, self.lai_min, self.lai_max)
        return lai
    
    def train_model(self, train_loader, valid_loader, optimizer, epochs=100, lr_scheduler=None, disable_tqdm=False):
        all_train_losses = []
        all_valid_losses = []
        all_lr = []
        for i in trange(epochs, disable=disable_tqdm):
            train_loss = self.fit(train_loader, optimizer) 
            all_train_losses.append(train_loss.item())
            valid_loss = self.validate(valid_loader)     
            all_valid_losses.append(valid_loss.item())    
            all_lr.append(optimizer.param_groups[0]['lr'])
            if lr_scheduler is not None:
                lr_scheduler.step(valid_loss)
            if all_lr[-1] <= 1e-8:
                break
        return all_train_losses, all_valid_losses, all_lr
    
    def fit(self, loader, optimizer):
        self.train()
        loss_mean = torch.tensor(0.0).to(self.device) 
        for _, batch in enumerate(loader):
            loss = self.get_batch_loss(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                loss_mean += loss / batch[0].size(0)
        return loss_mean
    
    def validate(self, loader):
        self.eval()
        with torch.no_grad():
            loss_mean = torch.tensor(0.0).to(self.device) 
            for _, batch in enumerate(loader):
                loss = self.get_batch_loss(batch)
                loss_mean += loss / batch[0].size(0)
        return loss_mean
    
    def get_batch_loss(self, batch):
        input_data, lai = batch
        lai_pred = self.forward(input_data.to(self.device))
        return (lai_pred - lai.to(self.device)).pow(2).sum()

def get_n_model_metrics(train_loader, valid_loader, test_loader_list=[], n=10, epochs=500, lr=0.001,disable_tqdm=False):
    metrics_names=["rmse", "r2", "mae", "reg_m", "reg_b", "best_valid_loss"]
    metrics = torch.zeros((n, len(test_loader_list), len(metrics_names)))
    for i in range(n):
        snap_nn = SnapNN(device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        optimizer = optim.Adam(snap_nn.parameters(), lr=lr)
        lr_scheduler =  torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=10, threshold=0.001)
        _, all_valid_losses, _ = snap_nn.train_model(train_loader, valid_loader, optimizer, epochs=epochs, 
                                                     lr_scheduler=lr_scheduler, disable_tqdm=disable_tqdm)
        for j in range(len(test_loader_list)):
            with torch.no_grad():
                test_data = test_loader_list[j].dataset[:]
                lai_pred = snap_nn.forward(test_data[0].to(snap_nn.device))
                lai_true = test_data[1]
                rmse = (lai_pred - lai_true).pow(2).mean().sqrt().cpu().item()
                r2 = r2_score(lai_true.squeeze().numpy(), lai_pred.squeeze().numpy())
                mae = (lai_pred - lai_true).abs().mean().cpu().item()
                reg_m, reg_b = np.polyfit(lai_true.squeeze().cpu().numpy(), lai_pred.squeeze().cpu().numpy(), 1)
                best_valid_loss = min(all_valid_losses)
                metrics[i,j,:] = torch.tensor([rmse, r2, mae, reg_m, reg_b, best_valid_loss])
    list_metrics_df = []
    for j in range(len(test_loader_list)):
        metrics_df = pd.DataFrame(data=metrics[:,j,:].numpy(), columns=metrics_names)
        list_metrics_df.append(metrics_df)
    return list_metrics_df, metrics

def main():
    if socket.gethostname()=='CELL200973':
        args=["-d", "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/snap_validation_data/",
              "-r", "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/results/snap_validation/",]
        disable_tqdm=False
        parser = get_prosailvae_results_parser().parse_args(args)    
    else:
        parser = get_prosailvae_results_parser().parse_args()
        disable_tqdm=True
    prepare_data = True
    save_dir = parser.data_dir
    res_dir = parser.results_dir
    if not os.path.isdir(res_dir):
        os.makedirs(res_dir)
    if prepare_data:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        data_eval, fold_data_list, tg_data_list, list_kl, list_params = prepare_datasets(n_eval=5000, n_samples_sub=5000, 
                                                                                         save_dir=save_dir, 
                                                                                         tg_mu = torch.tensor([0,1,2,3,4,5]),
                                                                                         tg_sigma = torch.tensor([0.5,1,2,3,4]))
    epochs = 1000
    n = 20
    lr = 0.001
    # eval_filepath = save_dir + "/eval_data.pth"
    # filepath = save_dir + "/fold_0_data.pth"
    # filepath = save_dir + "/tg_lai_mu_0_sigma_4_data.pth"
    test_loader, _ = get_loaders(data_eval, seed=86294692001, valid_ratio=0, batch_size=256)
    kl = torch.zeros(len(tg_data_list))
    mean_metrics = []
    all_metrics = []
    for i in range(len(tg_data_list)):
        kl[i] = list_kl[i]
        train_loader, valid_loader = get_loaders(tg_data_list[i], seed=86294692001, valid_ratio=0.1, batch_size=256)
        list_metrics_df, metrics = get_n_model_metrics(train_loader, valid_loader, test_loader_list=[test_loader], 
                                              n=n, epochs=epochs, lr=lr,disable_tqdm=disable_tqdm)
        mean_metrics.append(metrics.mean(0).unsqueeze(0))
        all_metrics.append(metrics.unsqueeze(0))
    metrics_names = ["rmse", "r2", "mae", "reg_m", "reg_b", "best_valid_loss"]
    eval_data_name = ["simulated_data"]
    mean_metrics = torch.cat(mean_metrics, 0)
    all_metrics = torch.cat(all_metrics, 0)
    torch.save(all_metrics, res_dir + "/all_metrics.pth")
    torch.save(kl, res_dir + "/kl.pth")
    for j in range(mean_metrics.size(1)):
        for k in range(mean_metrics.size(2)):
            fig, ax = plt.subplots(1, dpi=150, tight_layout=True)
            ax.scatter(kl, mean_metrics[:,j,k])
            ax.set_xlabel("KL distance between distribution of lai in training and evaluation dataset")
            ax.set_ylabel(metrics_names[k])
            fig.savefig(res_dir + f"/means_{eval_data_name[j]}_{metrics_names[k]}_vs_kl.png")
            
            fig, ax = plt.subplots(1, dpi=150, tight_layout=True)
            ax.boxplot(all_metrics[:,:,j,k], positions=kl, widths=0.1)
            ax.set_xticks(np.arange(0,11), np.arange(0,11))
            ax.set_xlabel("KL distance between distribution of lai in training and evaluation dataset")
            ax.set_ylabel(metrics_names[k])
            fig.savefig(res_dir + f"/boxplot_{eval_data_name[j]}_{metrics_names[k]}_vs_kl.png")

            fig, ax = plt.subplots(1, dpi=150, tight_layout=True)
            ax.scatter(kl.repeat(6,1).transpose(1,0).reshape(-1), all_metrics[:,:,j,k].reshape(-1), s=0.5)
            ax.set_xlabel("KL distance between distribution of lai in training and evaluation dataset")
            ax.set_ylabel(metrics_names[k])
            fig.savefig(res_dir + f"/scatter_{eval_data_name[j]}_{metrics_names[k]}_vs_kl.png")
    # snap_nn = SnapNN()
    # lr=0.001
    # epochs = 500
    # optimizer = optim.Adam(snap_nn.parameters(), lr=lr)
    # lr_scheduler =  torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=10, threshold=0.001)
    # all_train_losses, all_valid_losses, all_lr = snap_nn.train_model(train_loader, valid_loader, optimizer, epochs=epochs, lr_scheduler=lr_scheduler)
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(2,1, sharex=True)
    # ax[0].plot(torch.arange(epochs).numpy().tolist(), all_train_losses, label='train loss')
    # ax[0].plot(torch.arange(epochs).numpy().tolist(), all_valid_losses, label= "validation loss")
    # ax[0].legend()
    # ax[1].plot(torch.arange(epochs).numpy().tolist(), all_lr, label='lr')
    # ax[1].legend()
    # ax[0].set_yscale('log')
    # ax[1].set_yscale('log')
    # plt.show()
    # with torch.no_grad():
    #     test_data = test_loader.dataset[:]
    #     lai_pred = snap_nn.forward(test_data[0])
    #     lai_true = test_data[1]
    # plt.figure()
    # plt.scatter(lai_true, lai_pred, s=0.5)
    # plt.plot([0,14], [0,14], 'k--')
    # plt.axis('equal')
    # plt.show()

    # s2_r, s2_a, lai = load_weiss_dataset(os.path.join(prosailvae.__path__[0], os.pardir) + "/field_data/lai/")
    # fig, ax = plt.subplots()

    # ax.hist(lai, bins=100, density=True)
    # ax.plot([0, lai.max()],[1/lai.max(), 1/lai.max()])
    # x = torch.arange(0,14,0.01)
    # mu = torch.tensor(2)
    # sigma = torch.tensor(3)
    # p_x = truncated_gaussian_pdf(x, mu, sigma, eps=1e-9, lower=torch.tensor(0), upper=torch.tensor(14)).numpy()
    # ax.plot(x, p_x.squeeze())
    # n=1000
    # mu2 = mu.reshape(1,1)+1
    # sigma2 = sigma.reshape(1,1) * 100
    # tgt_dist_samples = sample_truncated_gaussian(mu2, sigma2, n_samples=n, lower = torch.tensor(0), upper=torch.tensor(14))
    # idx_samples = swap_sampling(torch.from_numpy(lai), tgt_dist_samples.squeeze(), allow_doubles=False)
    # subsampled_lai = lai[idx_samples]
    # ax.hist(subsampled_lai, bins=100, density=True)
    # ax.plot(x, truncated_gaussian_pdf(x, mu2, sigma2, eps=1e-9, lower =torch.tensor(0), upper=torch.tensor(14)).squeeze())
    # kernel_p = lambda x: truncated_gaussian_pdf(torch.tensor(x), mu, sigma, eps=1e-9, lower =torch.tensor(0), upper=torch.tensor(14)).numpy()
    # kernel_q = lambda x: truncated_gaussian_pdf(torch.tensor(x), mu + 2, sigma/2, eps=1e-9, lower =torch.tensor(0), upper=torch.tensor(14)).numpy()
    # sampled_idx = sample_from_dist(lai, n=100000, kernel_p=kernel_p, kernel_q=kernel_q)
    return

if __name__=="__main__":
    main()