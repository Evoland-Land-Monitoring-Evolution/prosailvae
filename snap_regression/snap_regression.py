import os
from typing import Callable, List
from collections import OrderedDict
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import trange
from sklearn.metrics import r2_score
import argparse
import socket
import prosailvae
from prosailvae.dist_utils import sample_truncated_gaussian, kl_tntn, truncated_gaussian_pdf, numerical_kl_from_pdf



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

def normalize(unnormalized:torch.Tensor, min_sample:torch.Tensor, max_sample:torch.Tensor):
    """
    Normalize with sample min and max of distribution
    """
    return 2 * (unnormalized - min_sample) / (max_sample - min_sample) - 1

def denormalize(normalized:torch.Tensor, min_sample:torch.Tensor, max_sample:torch.Tensor):
    """
    de-normalize with sample min and max of distribution
    """
    return 0.5 * (normalized + 1) * (max_sample - min_sample) + min_sample

def prepare_datasets(n_eval:int=5000, n_samples_sub:int=5000, save_dir:str="", reduce_to_common_samples_nb:bool=True,
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
        snap_nn = SnapNN(device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        snap_nn.set_weiss_weights()
        snap_lai = snap_nn.forward(torch.cat((s2_r, s2_a), 1).to(snap_nn.device)).cpu()

    data_s2 = torch.cat((s2_r, s2_a, snap_lai), 1)
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
            kl = kl_tntn(mu_ref, sigma_ref, mu, sigma, lower=torch.tensor(0.0), upper=torch.tensor(14.0)).item()
            list_kl.append(kl)
            list_params.append([mu, sigma])
            tgt_dist_samples = sample_truncated_gaussian(mu.reshape(1,1), sigma.reshape(1,1), n_samples=n_samples_sub*2,
                                                        lower = torch.tensor(0), upper=torch.tensor(14))
            idx_samples = swap_sampling(data_train[:,-1], tgt_dist_samples.squeeze(), allow_doubles=False)

            print(f"number of samples for mu = {mu.item()} sigma = {sigma.item()} (kl = {kl}) : {len(idx_samples)}")
            list_idx_samples_i.append(idx_samples)
            min_sample_nb = min(min_sample_nb, len(idx_samples))
            if plot_dist:
                fig, ax = plt.subplots(1, dpi=150)
                ax.plot(torch.arange(0,14,0.1), 
                        truncated_gaussian_pdf(torch.arange(0,14,0.1), mu_ref, sigma_ref,
                                               lower=torch.tensor(0), upper=torch.tensor(14)),
                        'k', label='original distribution')
                ax.hist(data_train[idx_samples,-1].squeeze(), bins=200, range=[0,14], alpha=0.5, density=True, label='samples')
                ax.plot(torch.arange(0,14,0.1),
                        truncated_gaussian_pdf(torch.arange(0,14,0.1), mu, sigma,
                                               lower=torch.tensor(0), upper=torch.tensor(14)),
                        'r', label='sampling distribution (kl={:.2f})'.format(kl))
                ax.legend()
                fig.savefig(save_dir + f"/samples_lai_mu_{mu.item()}_sigma_{sigma.item()}.png")
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


@dataclass
class ParametersSnapLAINN:
    """
    Weights and biases of Snap NN's two layers
    """
    layer_1_weight: torch.Tensor = torch.tensor([[-0.0234068789665, 0.921655164636, 0.13557654408, -1.9383314724,
                                                  -3.34249581612, 0.90227764801, 0.205363538259, -0.0406078447217,
                                                  -0.0831964097271, 0.260029270774, 0.284761567219],
                                                 [-0.132555480857, -0.139574837334, -1.0146060169, -1.33089003865,
                                                  0.0317306245033, -1.43358354132, -0.959637898575, 1.13311570655,
                                                  0.216603876542, 0.410652303763, 0.0647601555435],
                                                 [0.0860159777249, 0.616648776881, 0.678003876447, 0.141102398645,
                                                  -0.0966822068835, -1.12883263886, 0.302189102741, 0.4344949373,
                                                  -0.0219036994906, -0.228492476802, -0.0394605375898],
                                                 [-0.10936659367, -0.0710462629727, 0.0645824114783, 2.90632523682,
                                                  -0.673873108979, -3.83805186828, 1.69597934453, 0.0469502960817,
                                                  -0.0497096526884, 0.021829545431, 0.0574838271041],
                                                 [-0.08993941616, 0.175395483106, -0.0818473291726, 2.21989536749,
                                                  1.71387397514, 0.7130691861, 0.138970813499, -0.060771761518,
                                                  0.124263341255, 0.210086140404, -0.1838781387]])
    
    layer_1_bias: torch.Tensor = torch.tensor([4.96238030555, 1.41600844398, 1.07589704721, 
                                               1.53398826466, 3.02411593076])
    layer_2_weight: torch.Tensor = torch.tensor([[-1.50013548973,-0.0962832691215,-0.194935930577,-0.352305895756,0.0751074158475]])
    layer_2_bias: torch.Tensor = torch.tensor([1.09696310708])

@dataclass
class NormSnapNN:
    """
    Min and max snap nn input and output for normalization
    """
    input_min: torch.Tensor = torch.tensor([0.0000,  0.0000,  0.0000,  0.00663797254225,
                                            0.0139727270189,  0.0266901380821,  0.0163880741923,  0.0000,
                                            0.918595400582,  0.342022871159, -1.0000])

    input_max: torch.Tensor = torch.tensor([0.253061520472,  0.290393577911,  0.305398915249,  0.608900395798,
                                            0.753827384323,  0.782011770669,  0.493761397883,  0.49302598446,
                                            1.0000,  0.936206429175,  1.0000])


@dataclass
class DenormSNAPLAI:
    lai_min: torch.Tensor = torch.tensor(0.000319182538301)
    lai_max: torch.Tensor = torch.tensor(14.4675094548)

@dataclass
class DenormSNAPCab:
    cab_min: torch.Tensor = torch.tensor(0.00742669295987)
    cab_max: torch.Tensor = torch.tensor(873.90822211)

@dataclass
class DenormSNAPCw:
    cw_min: torch.Tensor = torch.tensor(3.85066859366e-06)
    cw_max: torch.Tensor = torch.tensor(0.522417054645)

def get_SNAP_norm_factors(ver:str='2.1', variable='lai'):
    """
    Get normalization factor for SNAP NN
    """
    if ver == "2.1":
        snap_norm = NormSnapNN()
        if variable=="lai":
            variable_min = DenormSNAPLAI().lai_min
            variable_max = DenormSNAPLAI().lai_max
        elif variable=="cab":
            variable_min = DenormSNAPCab().cab_min
            variable_max = DenormSNAPCab().cab_max
        elif variable=="cw":
            variable_min = DenormSNAPCw().cw_min
            variable_max = DenormSNAPCw().cw_max
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    return snap_norm.input_min, snap_norm.input_max, variable_min, variable_max


@dataclass
class ParametersSnapCabNN:
    """
    Weights and biases of Snap NN's two layers for Cab
    """
    layer_1_weight: torch.Tensor = torch.tensor([[0.400396555257,0.607936279259,0.13746865078,-2.95586657346,
                                                  -3.18674668773,2.20680075125,-0.31378433614,0.256063547511,
                                                  -0.0716132198051,0.51011350421,0.142813982139],
                                                 [-0.250781102415,0.43908630292,-1.16059093752,-1.86193525027,
                                                  0.981359868452,1.63423083425,-0.872527934646,0.448240475035,
                                                  0.0370780835012,0.0300441896704,0.0059566866194],
                                                 [0.552080132569,-0.502919673167,6.10504192497,-1.29438611914,
                                                  -1.05995638835,-1.39409290242,0.324752732711,-1.75887182283,
                                                  -0.0366636798603,-0.183105291401,-0.0381453121174],
                                                 [0.211591184882,-0.248788896074,0.887151598039,1.14367589557,
                                                  -0.753968830338,-1.18545695308,0.541897860472,-0.252685834608,
                                                  -0.0234149010781,-0.0460225035496,-0.00657028408066],
                                                 [0.254790234231,-0.724968611431,0.731872806027,2.30345382102,
                                                  -0.849907966922,-6.42531550054,2.23884455846,-0.199937574298,
                                                  0.0973033317146,0.334528254938,0.113075306592]])

    layer_1_bias: torch.Tensor = torch.tensor([4.24229967016,-0.259569088226,3.13039262734,
                                               0.774423577182,2.58427664853])
    layer_2_weight: torch.Tensor = torch.tensor([[-0.352760040599,-0.603407399151,0.135099379384,
                                                  -1.73567312385,-0.147546813318]])
    layer_2_bias: torch.Tensor = torch.tensor([0.463426463934])


@dataclass
class ParametersSnapCwNN:
    """
    Weights and biases of Snap NN's two layers for Cw
    """
    layer_1_weight: torch.Tensor = torch.tensor([[0.146378710426,1.18979928187,-0.906235139963,-0.808337508767,
                                                  -0.97333491783,-1.42591277646,-0.00561253629588,-0.634520356267,
                                                  -0.117226059989,-0.0602700912102,0.229407587132],
                                                 [0.283319173374,0.149342023041,1.08480588387,-0.138658791035,
                                                  -0.455759407329,0.420571438078,-1.7372949037,-0.704286287226,
                                                  0.0190953782358,-0.0393971316513,-0.00750241581744],
                                                 [-0.197487427943,-0.105460325978,0.158347670681,2.14912426654,
                                                  -0.970716842916,-4.92725317909,1.42034301781,1.45316917226,
                                                  0.0227257053609,0.269298650421,0.0849047657715],
                                                 [0.141405799763,0.33386260328,0.356218929123,-0.545942267639,
                                                  0.0891043076856,0.919298362929,-1.8520892625,-0.427539590779,
                                                  0.00791385646467,0.0148333201478,-0.00153786769736],
                                                 [-0.186781083395,-0.549163704901,-0.181287638772,0.96864043656,
                                                  -0.470442559117,-1.24859725244,2.67014942338,0.49009062438,
                                                  -0.00144931939526,0.00314829369692,0.0206517883893]])

    layer_1_bias: torch.Tensor = torch.tensor([-2.1064083686,-1.69022094794,3.10117655255,
                                               -1.31231626496,1.01131930348])
    layer_2_weight: torch.Tensor = torch.tensor([[-0.0775555890347,-0.86411786119,-0.199212415374,
                                                  1.98730461219,0.458926743489]])
    layer_2_bias: torch.Tensor = torch.tensor([-0.197591709977])


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

class SnapNN(nn.Module):
    """
    Neural Network with SNAP architecture to predict LAI from S2 reflectances and angles
    """
    def __init__(self, device:str='cpu', ver:str="2.1"): 
        super().__init__()
        input_min, input_max, lai_min, lai_max = get_SNAP_norm_factors(ver=ver)
        input_size = len(input_max) # 8 bands + 3 angles
        hidden_layer_size = 5
        layers = OrderedDict([
                  ('layer_1', nn.Linear(in_features=input_size, out_features=hidden_layer_size)),
                  ('tanh', nn.Tanh()),
                  ('layer_2', nn.Linear(in_features=hidden_layer_size, out_features=1))])
        self.input_min = input_min.to(device)
        self.input_max = input_max.to(device)
        self.lai_min = lai_min.to(device)
        self.lai_max = lai_max.to(device)
        self.net:nn.Sequential = nn.Sequential(layers).to(device)
        self.device = device

    def set_weiss_weights(self, ver:str=2.1):
        """
        Set Neural Network weights and biases to SNAP's original values
        """
        if ver==2.1:
            nn_parameters = ParametersSnapLAINN()
            self.net.layer_1.bias = nn.Parameter(nn_parameters.layer_1_bias.to(self.device))
            self.net.layer_1.weight = nn.Parameter(nn_parameters.layer_1_weight.to(self.device))
            self.net.layer_2.bias = nn.Parameter(nn_parameters.layer_2_bias.to(self.device))
            self.net.layer_2.weight = nn.Parameter(nn_parameters.layer_2_weight.to(self.device))
        else:
            raise NotImplementedError
    
    def forward(self, s2_data: torch.Tensor):
        """
        Forward method of SNAP NN to predict LAI
        """
        s2_data_norm = normalize(s2_data, self.input_min, self.input_max)
        lai_norm = self.net.forward(s2_data_norm)
        lai = denormalize(lai_norm, self.lai_min, self.lai_max)
        return lai
    
    def train_model(self, train_loader, valid_loader, optimizer, 
                    epochs:int=100, lr_scheduler=None, disable_tqdm:bool=False):
        """
        Fit and validate the model to data for a number of epochs
        """
        all_train_losses = []
        all_valid_losses = []
        all_lr = []
        for _ in trange(epochs, disable=disable_tqdm):
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
        """
        Apply mini-batch optimization from a train dataloader
        """
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
        """
        Compute loss on loader with mini-batches
        """
        self.eval()
        with torch.no_grad():
            loss_mean = torch.tensor(0.0).to(self.device) 
            for _, batch in enumerate(loader):
                loss = self.get_batch_loss(batch)
                loss_mean += loss / batch[0].size(0)
        return loss_mean

    def get_batch_loss(self, batch):
        """
        Computes loss on batch
        """
        s2_data, lai = batch
        lai_pred = self.forward(s2_data.to(self.device))
        return (lai_pred - lai.to(self.device)).pow(2).mean()

def get_model_metrics(test_data, model, all_valid_losses=[]):
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
                        disable_tqdm:bool=False,  patience:int=10, init_models:bool=False):
    """
    Trains several models on given train and validation dataloaders and assesses their regression metrics
    on provided test dataloader
    """
    if test_loader_list is None:
        raise ValueError("Please input a list of test loaders")
    metrics_names=["rmse", "r2", "mae", "reg_m", "reg_b", "best_valid_loss"]
    metrics = torch.zeros((n_models, len(test_loader_list), len(metrics_names)))
    for i in range(n_models):
        snap_nn = SnapNN(device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
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

def test_snap_nn():
    """
    Test if SNAP neural network's outputs are identical to that of the translated java
    code of SNAP
    """
    from weiss_lai_sentinel_hub import (get_norm_factors, get_layer_1_neuron_weights, get_layer_1_neuron_biases, 
                                        get_layer_2_weights, get_layer_2_bias, neuron, layer2) 
    
    s2_r, s2_a, lai = load_weiss_dataset(os.path.join(prosailvae.__path__[0], os.pardir) + "/field_data/lai/")   
    snap_nn = SnapNN()
    snap_nn.set_weiss_weights()
    sample = torch.cat((torch.from_numpy(s2_r), torch.cos(torch.from_numpy(s2_a))), 1).float()
    ver="2.1"
    norm_factors = get_norm_factors(ver=ver)
    w1, w2, w3, w4, w5 = get_layer_1_neuron_weights(ver=ver)
    b1, b2, b3, b4, b5 = get_layer_1_neuron_biases(ver=ver)
    wl2 = get_layer_2_weights(ver=ver)
    bl2 = get_layer_2_bias(ver=ver)

    b03_norm = normalize(sample[:,0].unsqueeze(1), norm_factors["min_sample_B03"], norm_factors["max_sample_B03"])
    b04_norm = normalize(sample[:,1].unsqueeze(1), norm_factors["min_sample_B04"], norm_factors["max_sample_B04"])
    b05_norm = normalize(sample[:,2].unsqueeze(1), norm_factors["min_sample_B05"], norm_factors["max_sample_B05"])
    b06_norm = normalize(sample[:,3].unsqueeze(1), norm_factors["min_sample_B06"], norm_factors["max_sample_B06"])
    b07_norm = normalize(sample[:,4].unsqueeze(1), norm_factors["min_sample_B07"], norm_factors["max_sample_B07"])
    b8a_norm = normalize(sample[:,5].unsqueeze(1), norm_factors["min_sample_B8A"], norm_factors["max_sample_B8A"])
    b11_norm = normalize(sample[:,6].unsqueeze(1), norm_factors["min_sample_B11"], norm_factors["max_sample_B11"])
    b12_norm = normalize(sample[:,7].unsqueeze(1), norm_factors["min_sample_B12"], norm_factors["max_sample_B12"])
    viewZen_norm = normalize(sample[:,8].unsqueeze(1), norm_factors["min_sample_viewZen"], norm_factors["max_sample_viewZen"])
    sunZen_norm  = normalize(sample[:,9].unsqueeze(1), norm_factors["min_sample_sunZen"], norm_factors["max_sample_sunZen"])
    relAzim_norm = sample[:,10].unsqueeze(1)
    band_dim = 1
    with torch.no_grad():
        x_norm = normalize(sample, snap_nn.input_min, snap_nn.input_max)
        snap_input = torch.cat((b03_norm, b04_norm, b05_norm, b06_norm, b07_norm, b8a_norm, b11_norm, b12_norm,
                    viewZen_norm, sunZen_norm, relAzim_norm), axis=band_dim)
        assert torch.isclose(snap_input, x_norm, atol=1e-5).all()
        nb_dim = len(b03_norm.size())
        neuron1 = neuron(snap_input, w1, b1, nb_dim, sum_dim=band_dim)
        neuron2 = neuron(snap_input, w2, b2, nb_dim, sum_dim=band_dim)
        neuron3 = neuron(snap_input, w3, b3, nb_dim, sum_dim=band_dim)
        neuron4 = neuron(snap_input, w4, b4, nb_dim, sum_dim=band_dim)
        neuron5 = neuron(snap_input, w5, b5, nb_dim, sum_dim=band_dim)
        linear_1_snap = nn.Linear(11,5)
        linear_1_snap.weight = snap_nn.net.layer_1.weight
        linear_1_snap.bias = snap_nn.net.layer_1.bias
        assert torch.isclose(linear_1_snap(x_norm), snap_nn.net.layer_1.bias
                             + x_norm @ snap_nn.net.layer_1.weight.transpose(1,0), atol=1e-4).all()

        n_snap_nn = torch.tanh(snap_nn.net.layer_1.bias + x_norm @ snap_nn.net.layer_1.weight.transpose(1,0))
        assert torch.isclose(n_snap_nn, torch.cat((neuron1, neuron2, neuron3, neuron4, neuron5), axis=1), atol=1e-4).all()

        linear_2_snap = snap_nn.net.layer_2
        # linear_2_snap.weight = snap_nn.net[2].weight
        # linear_2_snap.bias = snap_nn.net[2].bias
        assert torch.isclose(linear_2_snap(n_snap_nn), snap_nn.net.layer_2.bias
                             + n_snap_nn @ snap_nn.net.layer_2.weight.transpose(1,0), atol=1e-4).all()
        layer_2_output = layer2(neuron1, neuron2, neuron3, neuron4, neuron5, wl2, bl2, sum_dim=band_dim)
        l_snap_nn = snap_nn.net.layer_2.bias + n_snap_nn @ snap_nn.net.layer_2.weight.transpose(1,0)
        lai_prenorm_snap = snap_nn.net.forward(x_norm)
        assert torch.isclose(l_snap_nn.squeeze(), layer_2_output.squeeze(), atol=1e-4).all()
        assert torch.isclose(lai_prenorm_snap.squeeze(), layer_2_output.squeeze(), atol=1e-4).all()
        lai = denormalize(layer_2_output, norm_factors["min_sample_lai"], norm_factors["max_sample_lai"])
        snap_lai = denormalize(l_snap_nn, snap_nn.lai_min, snap_nn.lai_max)
        assert torch.isclose(snap_lai.squeeze(), lai.squeeze(), atol=1e-4).all()
        assert torch.isclose(snap_nn.forward(sample).squeeze(), lai.squeeze(), atol=1e-4).all()

def main():
    test_snap_nn()
    if socket.gethostname()=='CELL200973':
        args=["-d", "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/snap_validation_data/",
              "-r", "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/results/snap_validation/",
              "-e", "3",
              "-n", "2",
              "-i", 't',
              "-lr", "0.001"]
        disable_tqdm=False
        
        # tg_mu = torch.tensor([0,1])
        # tg_sigma = torch.tensor([0.5,1])
        tg_mu = torch.tensor([0,1,2,3,4,5])
        tg_sigma = torch.tensor([0.5,1,2,3,4])
        parser = get_parser().parse_args(args)
        s2_tensor_image_path = "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/torch_files/T31TCJ/after_SENTINEL2B_20171127-105827-648_L2A_T31TCJ_C_V2-2_roi_0.pth"  
    else:
        parser = get_parser().parse_args()

        tg_mu = torch.tensor([0,1,2,3,4,5])
        tg_sigma = torch.tensor([0.5,1,2,3,4])
        TENSOR_DIR="/work/CESBIO/projects/MAESTRIA/prosail_validation/validation_sites/torchfiles/T31TCJ/"
        image_filename = "/after_SENTINEL2B_20171127-105827-648_L2A_T31TCJ_C_V2-2_roi_0.pth"
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
        (data_eval_list, fold_data_list, tg_data_list, 
         list_kl, list_params) = prepare_datasets(n_eval=5000, n_samples_sub=5000,
                                                                save_dir=save_dir,
                                                                tg_mu = tg_mu,
                                                                tg_sigma = tg_sigma,
                                                                plot_dist=True,
                                                                s2_tensor_image_path=s2_tensor_image_path)
    fold_xp = parser.fold_xp
    if fold_xp:
        metrics_names = ["rmse", "r2", "mae", "reg_m", "reg_b", "best_valid_loss"]
        eval_data_name = ["simulated_data", "snap_s2"]
        snap_ref = SnapNN(device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        snap_ref.set_weiss_weights()
        test_loader_list = []
        all_metrics_ref = []
        for _, data_eval in enumerate(data_eval_list):
            test_loader, _ = get_loaders(data_eval, seed=86294692001, valid_ratio=0, batch_size=256)
            test_loader_list.append(test_loader)
            metrics_ref = []
            for i in range(len(fold_data_list)):
                _, valid_loader = get_loaders(fold_data_list[i], seed=86294692001, valid_ratio=0.1, batch_size=256)
                snap_valid_loss = snap_ref.validate(valid_loader)
                metrics_ref.append(get_model_metrics(test_loader.dataset[:], model=snap_ref, all_valid_losses=[snap_valid_loss]))
            metrics_ref = torch.stack(metrics_ref, dim=0)
            all_metrics_ref.append(metrics_ref)
        metrics_ref = torch.stack(all_metrics_ref, dim=0).transpose(1,0)
        if compute_metrics:
            mean_metrics = []
            all_metrics = []
            for i in range(len(fold_data_list)):
                train_loader, valid_loader = get_loaders(fold_data_list[i], seed=86294692001, valid_ratio=0.1, batch_size=256)
                metrics = get_n_model_metrics(train_loader, valid_loader, test_loader_list=test_loader_list, 
                                                n_models=n_models, epochs=epochs, lr=lr, disable_tqdm=disable_tqdm, patience=20, 
                                                                init_models=init_models)
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
                    ax.scatter(np.arange(len(fold_data_list)), metrics_ref[:,j,k], label=f'SNAP {metrics_names[k]}', c='k')
                ax.set_xlabel("Fold number")
                ax.set_ylabel(metrics_names[k])
                fig.savefig(res_dir + f"/means_{eval_data_name[j]}_{metrics_names[k]}_vs_fold.png")
                        
                fig, ax = plt.subplots(1, dpi=150, tight_layout=True)
                ax.scatter(np.arange(len(fold_data_list)), median_metrics[:,j,k])
                if metrics_ref[:,j,k].min() == metrics_ref[:,j,k].max():
                    ax.axhline(metrics_ref[:,j,k].min(), c='k', label=f'SNAP {metrics_names[k]}')
                else:
                    ax.scatter(np.arange(len(fold_data_list)), metrics_ref[:,j,k], label=f'SNAP {metrics_names[k]}', c='k')
                ax.set_xlabel("Fold number")
                ax.set_ylabel(metrics_names[k])
                fig.savefig(res_dir + f"/median_{eval_data_name[j]}_{metrics_names[k]}_vs_fold.png")
                
                fig, ax = plt.subplots(1, dpi=150, tight_layout=True)
                ax.boxplot(all_metrics[:,:,j,k], positions=np.arange(len(fold_data_list)), widths=0.1)
                if metrics_ref[:,j,k].min() == metrics_ref[:,j,k].max():
                    ax.axhline(metrics_ref[:,j,k].min(), c='k', label=f'SNAP {metrics_names[k]}')
                else:
                    ax.scatter(np.arange(len(fold_data_list)), metrics_ref[:,j,k], label=f'SNAP {metrics_names[k]}', c='k')
                ax.set_xticks(np.arange(0,11), np.arange(0,11))
                ax.set_xlabel("Fold number")
                ax.set_ylabel(metrics_names[k])
                fig.savefig(res_dir + f"/boxplot_{eval_data_name[j]}_{metrics_names[k]}_vs_fold.png")

                fig, ax = plt.subplots(1, dpi=150, tight_layout=True)
                ax.scatter(torch.arange(len(fold_data_list)).repeat(all_metrics.size(1),1).transpose(1,0).reshape(-1), all_metrics[:,:,j,k].reshape(-1), s=0.5)
                if metrics_ref[:,j,k].min() == metrics_ref[:,j,k].max():
                    ax.axhline(metrics_ref[:,j,k].min(), c='k', label=f'SNAP {metrics_names[k]}')
                else:
                    ax.scatter(np.arange(len(fold_data_list)), metrics_ref[:,j,k], label=f'SNAP {metrics_names[k]}', c='k')
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
            test_loader, _ = get_loaders(data_eval, seed=86294692001, valid_ratio=0, batch_size=256)
            test_loader_list.append(test_loader)
            metrics_ref = []
            for i, tg_data in enumerate(tg_data_list):
                _, valid_loader = get_loaders(tg_data, seed=86294692001, valid_ratio=0.1, batch_size=256)
                snap_valid_loss = snap_ref.validate(valid_loader)
                metrics_ref.append(get_model_metrics(test_loader.dataset[:], model=snap_ref, all_valid_losses=[snap_valid_loss]))
            metrics_ref = torch.stack(metrics_ref, dim=0)
            all_metrics_ref.append(metrics_ref)
        metrics_ref = torch.stack(all_metrics_ref, dim=0).transpose(1,0)
        if compute_metrics:
            kl = torch.zeros(len(tg_data_list))
            mean_metrics = []
            all_metrics = []
            for i, tg_data in enumerate(tg_data_list):
                kl[i] = list_kl[i]
                train_loader, valid_loader = get_loaders(tg_data, seed=86294692001, valid_ratio=0.1, batch_size=256)
                metrics = get_n_model_metrics(train_loader, valid_loader, test_loader_list=test_loader_list, 
                                                                n_models=n_models, epochs=epochs, lr=lr,disable_tqdm=disable_tqdm, patience=20, 
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
            plt.colorbar(sc, ax=ax, label='kl divergence between evaluation dataset and training dataset')
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
                ax.scatter(kl.repeat(all_metrics.size(1),1).transpose(1,0).reshape(-1), all_metrics[:,:,j,k].reshape(-1), s=0.5)
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
                ax.scatter(tg_sigma_rep.repeat(all_metrics.size(1),1).transpose(1,0).reshape(-1), all_metrics[:,:,j,k].reshape(-1), s=0.5)
                if metrics_ref[:,j,k].min() == metrics_ref[:,j,k].max():
                    ax.axhline(metrics_ref[:,j,k].min(), c='k', label=f'SNAP {metrics_names[k]}')
                else:
                    ax.scatter(tg_sigma_rep, metrics_ref[:,j,k], label=f'SNAP {metrics_names[k]}', c='k')
                ax.set_xlabel("sampling distribution sigma")
                ax.set_ylabel(metrics_names[k])
                ax.set_xscale('symlog')
                fig.savefig(sigma_res_dir + f"/scatter_{eval_data_name[j]}_{metrics_names[k]}_vs_sigma.png")
                plt.close('all')
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