#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 10:46:58 2022

@author: yoel
"""
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd

import torch
import os 
import argparse
import prosailvae
from prosailvae.ProsailSimus import ProsailSimulator, SensorSimulator, BANDS, PROSAILVARS
from prosailvae.prosail_var_dists import get_prosail_var_dist
from dataset.bvnet_dataset import load_bvnet_dataset
from tqdm import trange
import scipy.stats as stats
from dataset.dataset_utils import min_max_to_loc_scale
from prosailvae.spectral_indices import get_spectral_idx
import socket
from prosailvae.prosail_var_dists import VariableDistribution

def correlate_with_lai_V1(lai, V, V_mean, lai_conv):
    # V_corr = np.zeros_like(V)
    # V_corr[V >= V_mean] = V[V >= V_mean]
    # V_corr[V < V_mean] = V_mean + (V[V < V_mean] - V_mean) * np.maximum((lai_conv - lai[V < V_mean]), 0) / lai_conv
    # V_corr[lai > 10] = V_mean
    V_corr = V_mean + (V - V_mean) * np.maximum((lai_conv - lai), 0) / lai_conv
    return V_corr

def Vmin(LAI, Vmin_0, Vmin_lai_max, lai_max):
    return Vmin_0 + LAI / lai_max * (Vmin_lai_max - Vmin_0)

def Vmax(LAI, Vmax_0, Vmax_lai_max, lai_max):
    return Vmax_0 + LAI / lai_max * (Vmax_lai_max - Vmax_0)

def correlate_with_lai_V2(lai, V, Vmin_0, Vmax_0, Vmin_lai_max, Vmax_lai_max, lai_max, lai_thresh=None):
    Vmin_lai = Vmin(lai, Vmin_0, Vmin_lai_max, lai_max)
    Vmax_lai = Vmax(lai, Vmax_0, Vmax_lai_max, lai_max)
    V_corr = (V - Vmin_0) * (Vmax_lai - Vmin_lai) / (Vmax_0 - Vmin_0) + Vmin_lai
    if lai_thresh is not None: 
        V_corr[lai > lai_thresh] = Vmin(lai_thresh, Vmin_0, Vmin_lai_max, lai_max)
    return V_corr

def correlate_sample_with_lai(sample, lai, param_dist:VariableDistribution, 
                                lai_corr_mode:str="v2", lai_conv_override:bool|None=None, 
                                lai_max:float=15, lai_thresh:float|None=None):
    
    if lai_corr_mode=="v2":
        if param_dist.C_lai_min is not None:
            correlated_sample = correlate_with_lai_V2(lai, sample, param_dist.low, param_dist.high, param_dist.C_lai_min, 
                                           param_dist.C_lai_max, lai_max=lai_max, lai_thresh=lai_thresh)
    else:
        if lai is not None and (param_dist.lai_conv is not None or lai_conv_override is not None):
            lai_conv = lai_conv_override if lai_conv_override is not None else param_dist.lai_conv
            if param_dist.loc is None : 
                correlated_sample = correlate_with_lai_V1(lai, sample, 
                                            (param_dist.high - param_dist.low)/2, 
                                            lai_conv)
            else:
                correlated_sample = correlate_with_lai_V1(lai, sample, param_dist.loc, lai_conv)
    return correlated_sample

def correlate_all_variables_with_lai(samples, var_dists, lai_conv_override=None, lai_corr_mode="v2", lai_thresh=None):
    variable_idx_dict = {"N":0, "cab":1, "car":2, "cbrown":3, "cw":4, "cm":5, "lidfa":7, "hspot":8, "psoil":9, "rsoil":10}
    correlated_samples = samples.copy()
    for variable, idx in variable_idx_dict.items(): 
        variable_dist = VariableDistribution(**var_dists.asdict()[variable])
        correlated_samples[:,idx] = correlate_sample_with_lai(samples[:,idx], samples[:,6], variable_dist, lai_corr_mode=lai_corr_mode, 
                                                                lai_conv_override=lai_conv_override, lai_max=var_dists.lai.high, 
                                                                lai_thresh=lai_thresh)

    return correlated_samples

def np_sample_param(param_dist, lai=None, n_samples=1, uniform_mode=True, lai_conv_override=None, 
                    lai_corr_mode="v2", lai_max=15, lai_thresh=None):
    if param_dist.law == "uniform" or uniform_mode:
        sample = np.random.uniform(low=param_dist.low, high=param_dist.high, size=(n_samples))
    elif param_dist.law == "gaussian":
        sample = stats.truncnorm((param_dist.low - param_dist.loc) / param_dist.scale, 
                                 (param_dist.high - param_dist.loc) / param_dist.scale, 
                            loc=param_dist.loc, scale=param_dist.scale).rvs(n_samples)
    elif param_dist.law == "lognormal":
        low = max(param_dist.low, 1e-8)
        X = stats.truncnorm((np.log(low) - param_dist.loc) / param_dist.scale, (np.log(param_dist.high) - param_dist.loc) / param_dist.scale, 
                            loc=param_dist.loc, scale=param_dist.scale).rvs(n_samples)
        sample = np.exp(X)
    else:
        raise NotImplementedError("Please choose sample distribution among gaussian, uniform and lognormal")
    if lai is not None:
        correlate_sample_with_lai(sample, lai, param_dist, lai_corr_mode=lai_corr_mode, 
                                  lai_conv_override=lai_conv_override, lai_max=lai_max, 
                                  lai_thresh=lai_thresh)
    return sample


def sample_angles(n_samples=100):
    PATH_TO_DATA_DIR = os.path.join(prosailvae.__path__[0], os.pardir) + "/field_data/lai/"
    path_to_file = PATH_TO_DATA_DIR + "/InputNoNoise_2.csv"
    assert os.path.isfile(path_to_file)
    df_validation_data = pd.read_csv(path_to_file, sep=" ", engine="python")
    tts_w = np.rad2deg(np.arccos(df_validation_data['cos(thetas)'].values))
    tto_w = np.rad2deg(np.arccos(df_validation_data['cos(thetav)'].values))
    psi_w = np.rad2deg(np.arccos(df_validation_data['cos(phiv-phis)'].values))
    n_data = len(psi_w)
    sample_idx = np.random.randint(low=0,high=n_data,size=(n_samples))
    return tts_w[sample_idx], tto_w[sample_idx], psi_w[sample_idx]


def partial_sample_prosail_vars(var_dists, lai=None, tts=None, tto=None, psi=None, n_samples=1, 
                                uniform_mode=True, lai_corr=False, lai_conv_override=None, 
                                lai_var_dist:VariableDistribution|None=None, lai_corr_mode="v2", lai_thresh=None):
    prosail_vars = np.zeros((n_samples, 14))
    if lai is None:
        if lai_var_dist is not None:
            lai = np_sample_param(lai_var_dist, lai=None, n_samples=n_samples, uniform_mode=uniform_mode)
        else:
            lai = np_sample_param(var_dists.lai, lai=None, n_samples=n_samples, uniform_mode=uniform_mode)
    prosail_vars[:,6] = lai
    variable_idx_dict = {"N":0, "cab":1, "car":2, "cbrown":3, "cw":4, "cm":5, "lidfa":7, "hspot":8, "psoil":9, "rsoil":10}
    for variable, idx in variable_idx_dict.items(): 
        variable_dist = VariableDistribution(**var_dists.asdict()[variable])
        prosail_vars[:,idx] = np_sample_param(variable_dist, lai=lai if lai_corr else None, 
                                              n_samples=n_samples, uniform_mode=uniform_mode, 
                                              lai_conv_override=lai_conv_override, lai_max=var_dists.lai.high, 
                                              lai_corr_mode=lai_corr_mode, lai_thresh=lai_thresh)
    tts, tto, psi = sample_angles(n_samples)
    prosail_vars[:,11] = tts
    prosail_vars[:,12] = tto
    prosail_vars[:,13] = psi

    return prosail_vars


# def simulate_prosail_dataset(nb_simus=100, noise=0, psimulator=None, ssimulator=None, static_angles=False):
#     prosail_vars = np.zeros((nb_simus, 14))
#     prosail_s2_sim = np.zeros((nb_simus, 10))
#     with numpyro.handlers.seed(rng_seed=5):
#         for i in trange(nb_simus):
#             prosail_vars[i,:] = sample_prosail_vars(ProsailVarsDist, static_angles=static_angles)
#             mean = ssimulator(psimulator(torch.from_numpy(prosail_vars[i,:]).view(1,-1).float())).numpy()
#             if noise>0:
#                 sigma = numpyro.sample("sigma", dist.Uniform(0., noise))
#                 prosail_s2_sim[i,:] = numpyro.sample("obs", dist.Normal(mean, sigma))
#             else:
#                 prosail_s2_sim[i,:] = mean
#     return prosail_vars, prosail_s2_sim

def sample_prosail_vars(nb_simus=2048, prosail_var_dist_type="legacy", uniform_mode=False, lai_corr=True, 
                        lai_var_dist:VariableDistribution|None=None, lai_corr_mode="v2", lai_thresh=None):
    prosail_var_dist = get_prosail_var_dist(prosail_var_dist_type)
    samples = partial_sample_prosail_vars(prosail_var_dist, n_samples=nb_simus, 
                                          uniform_mode=uniform_mode, lai_corr=lai_corr, 
                                          lai_var_dist=lai_var_dist, lai_corr_mode=lai_corr_mode,
                                          lai_thresh=lai_thresh)
    return samples

def simulate_reflectances(prosail_vars, noise=0, psimulator=None, ssimulator=None, n_samples_per_batch=1024):
    nb_simus = prosail_vars.shape[0]
    prosail_s2_sim = np.zeros((nb_simus, ssimulator.rsr.size(1)))
    n_full_batch = nb_simus // n_samples_per_batch
    last_batch = nb_simus - nb_simus // n_samples_per_batch * n_samples_per_batch

    for i in range(n_full_batch):
        prosail_r = psimulator(torch.from_numpy(prosail_vars[i*n_samples_per_batch : (i+1) * n_samples_per_batch,
                                                             :]).view(n_samples_per_batch,-1).float())
        sim_s2_r = ssimulator(prosail_r).numpy()
        if noise>0:
            sigma = np.random.rand(n_samples_per_batch,1) * noise * np.ones_like(sim_s2_r)
            add_noise = np.random.normal(loc = np.zeros_like(sim_s2_r), scale=sigma, size=sim_s2_r.shape)
            sim_s2_r += add_noise
        prosail_s2_sim[i*n_samples_per_batch : (i+1) * n_samples_per_batch,:] = sim_s2_r
    if last_batch > 0:
        sim_s2_r = ssimulator(psimulator(torch.from_numpy(prosail_vars[n_full_batch*n_samples_per_batch:,
                                                                       :]).view(last_batch,-1).float())).numpy()
        if noise>0:
            sigma = np.random.rand(last_batch,1) * noise * np.ones_like(sim_s2_r)
            add_noise = np.random.normal(loc = np.zeros_like(sim_s2_r), scale=sigma, size=sim_s2_r.shape)
            sim_s2_r += add_noise
        prosail_s2_sim[n_full_batch*n_samples_per_batch:,:] = sim_s2_r
    return prosail_s2_sim

def np_simulate_prosail_dataset(nb_simus=2048, noise=0, psimulator=None, ssimulator=None, 
                                n_samples_per_batch=1024, uniform_mode=False, lai_corr=True,
                                prosail_var_dist_type="legacy", lai_var_dist:VariableDistribution|None=None,
                                lai_corr_mode="v2", lai_thresh=None):
    
    prosail_vars = sample_prosail_vars(nb_simus=nb_simus, prosail_var_dist_type=prosail_var_dist_type, 
                                       uniform_mode=uniform_mode, lai_corr=lai_corr, lai_var_dist=lai_var_dist, 
                                       lai_corr_mode=lai_corr_mode, lai_thresh=lai_thresh)
    
    prosail_s2_sim = simulate_reflectances(prosail_vars, noise=noise, psimulator=psimulator, ssimulator=ssimulator, 
                                           n_samples_per_batch=n_samples_per_batch)
    
    return prosail_vars, prosail_s2_sim

def save_prosail_data_set_with_all_prospect_versions(data_dir, data_file_prefix, rsr_dir, nb_simus, noise=0, bvnet_bands=False,
                                                     n_samples_per_batch=1024, uniform_mode=False, prosail_var_dist_type="legacy", 
                                                     lai_var_dist:VariableDistribution|None=None,
                                                     lai_thresh=None):
    
    bands = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12] # B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12
    if bvnet_bands:
        bands = [2, 3, 4, 5, 6, 8, 11, 12] #       B3, B4, B5, B6, B7,     B8A, B11, B12

    ssimulator = SensorSimulator(rsr_dir + "/sentinel2.rsr", bands=bands)   
    prosail_vars = sample_prosail_vars(nb_simus=nb_simus, prosail_var_dist_type=prosail_var_dist_type, 
                                       uniform_mode=uniform_mode, lai_corr=False, lai_var_dist=lai_var_dist, 
                                       lai_corr_mode="", lai_thresh=None)
    
    for lai_corr_mode in ["v2", "v1"]:
        prosail_var_dist = get_prosail_var_dist(prosail_var_dist_type)
        correlated_prosail_vars = correlate_all_variables_with_lai(prosail_vars, prosail_var_dist, 
                                                                    lai_corr_mode=lai_corr_mode,
                                                                    lai_thresh=lai_thresh)
        for prospect_version in ["5", "D", "PRO"]:
            psimulator = ProsailSimulator(prospect_version=prospect_version)
            prosail_s2_sim = simulate_reflectances(correlated_prosail_vars, noise=noise, psimulator=psimulator, 
                                                   ssimulator=ssimulator, n_samples_per_batch=n_samples_per_batch)
            (norm_mean, norm_std, cos_angles_loc, cos_angles_scale, idx_loc, 
            idx_scale) = get_bands_norm_factors(torch.from_numpy(prosail_s2_sim).float().transpose(1,0), mode='quantile')
            torch.save(torch.from_numpy(correlated_prosail_vars), 
                    os.path.join(data_dir, f"{data_file_prefix}PROSPECT{prospect_version}_corr_{lai_corr_mode}_prosail_sim_vars.pt"))
            torch.save(torch.from_numpy(prosail_s2_sim), 
                    os.path.join(data_dir, f"{data_file_prefix}PROSPECT{prospect_version}_corr_{lai_corr_mode}_prosail_s2_sim_refl.pt"))
            torch.save(norm_mean, 
                    os.path.join(data_dir, f"{data_file_prefix}PROSPECT{prospect_version}_corr_{lai_corr_mode}_norm_mean.pt"))
            torch.save(norm_std, 
                    os.path.join(data_dir, f"{data_file_prefix}PROSPECT{prospect_version}_corr_{lai_corr_mode}_norm_std.pt"))
            torch.save(cos_angles_loc, 
                    os.path.join(data_dir, f"{data_file_prefix}PROSPECT{prospect_version}_corr_{lai_corr_mode}_angles_loc.pt"))
            torch.save(cos_angles_scale, 
                    os.path.join(data_dir, f"{data_file_prefix}PROSPECT{prospect_version}_corr_{lai_corr_mode}_angles_scale.pt"))
            torch.save(idx_loc, 
                    os.path.join(data_dir, f"{data_file_prefix}PROSPECT{prospect_version}_corr_{lai_corr_mode}_idx_loc.pt"))
            torch.save(idx_scale, 
                    os.path.join(data_dir, f"{data_file_prefix}PROSPECT{prospect_version}_corr_{lai_corr_mode}_idx_scale.pt"))



def get_refl_normalization(prosail_refl):
    return prosail_refl.mean(0), prosail_refl.std(0)


def get_bands_norm_factors(s2_r_samples, mode='mean'):
    cos_angle_min = torch.tensor([0.342108564072183, 0.979624800125421, -1.0000]) # sun zenith, S2 senith, relative azimuth
    cos_angle_max = torch.tensor([0.9274847491748729, 1.0000, 1.0000])
    with torch.no_grad():       
        spectral_idx = get_spectral_idx(s2_r_samples, bands_dim=1).reshape(4, -1)
        if mode=='mean':
            norm_mean = s2_r_samples.mean(1)
            norm_std = s2_r_samples.std(1)
            idx_norm_mean = spectral_idx.mean(1)
            idx_norm_std = spectral_idx.std(1)
            
        elif mode=='quantile':
            max_samples=int(1e7)
            norm_mean = torch.quantile(s2_r_samples[:, :max_samples], q=torch.tensor(0.5), dim=1)
            norm_std = torch.quantile(s2_r_samples[:, :max_samples], q=torch.tensor(0.95), dim=1) - torch.quantile(s2_r_samples[:, :max_samples], q=torch.tensor(0.05), dim=1)
            idx_norm_mean = torch.quantile(spectral_idx[:, :max_samples], q=torch.tensor(0.5), dim=1)
            idx_norm_std = torch.quantile(spectral_idx[:, :max_samples], q=torch.tensor(0.95), dim=1) - torch.quantile(spectral_idx[:, :max_samples], q=torch.tensor(0.05), dim=1)

        cos_angles_loc, cos_angles_scale = min_max_to_loc_scale(cos_angle_min, cos_angle_max)

    return norm_mean, norm_std, cos_angles_loc, cos_angles_scale, idx_norm_mean, idx_norm_std

def save_dataset(data_dir, data_file_prefix, rsr_dir, nb_simus, noise=0, bvnet_bands=False, uniform_mode=False, 
                 lai_corr=True, prosail_var_dist_type="legacy", lai_var_dist:VariableDistribution|None=None, 
                 lai_corr_mode="v2", lai_thresh=None, prospect_version="5"):

    psimulator = ProsailSimulator(prospect_version=prospect_version)
    bands = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12] # B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12
    if bvnet_bands:
        bands = [2, 3, 4, 5, 6, 8, 11, 12] #       B3, B4, B5, B6, B7,     B8A, B11, B12
    ssimulator = SensorSimulator(rsr_dir + "/sentinel2.rsr", bands=bands)
    prosail_vars, prosail_s2_sim = np_simulate_prosail_dataset(nb_simus=nb_simus,
                                                                noise=noise,
                                                                psimulator=psimulator,
                                                                ssimulator=ssimulator,
                                                                n_samples_per_batch=1024,
                                                                uniform_mode=uniform_mode,
                                                                lai_corr=lai_corr,
                                                                prosail_var_dist_type=prosail_var_dist_type,
                                                                lai_var_dist=lai_var_dist,
                                                                lai_corr_mode=lai_corr_mode, 
                                                                lai_thresh=lai_thresh)
    (norm_mean, norm_std, cos_angles_loc, cos_angles_scale, idx_loc, 
     idx_scale) = get_bands_norm_factors(torch.from_numpy(prosail_s2_sim).float().transpose(1,0), mode='quantile')
    torch.save(torch.from_numpy(prosail_vars), os.path.join(data_dir, f"{data_file_prefix}prosail_sim_vars.pt"))
    torch.save(torch.from_numpy(prosail_s2_sim), os.path.join(data_dir, f"{data_file_prefix}prosail_s2_sim_refl.pt"))
    torch.save(norm_mean, os.path.join(data_dir, f"{data_file_prefix}norm_mean.pt"))
    torch.save(norm_std, os.path.join(data_dir, f"{data_file_prefix}norm_std.pt"))
    torch.save(cos_angles_loc, os.path.join(data_dir, f"{data_file_prefix}angles_loc.pt"))
    torch.save(cos_angles_scale, os.path.join(data_dir, f"{data_file_prefix}angles_scale.pt"))
    torch.save(idx_loc, os.path.join(data_dir, f"{data_file_prefix}idx_loc.pt"))
    torch.save(idx_scale, os.path.join(data_dir, f"{data_file_prefix}idx_scale.pt"))

def save_bvnet_dataset(data_dir, rsr_dir, noise=0, lai_corr=True, nb_test_samples=0, prosail_var_dist_type="legacy"):

    PATH_TO_DATA_DIR = os.path.join(prosailvae.__path__[0], os.pardir) + "/field_data/lai/"
    bvnet_data_dir = os.path.join(data_dir, os.pardir) + "/bvnet/"
    if not os.path.isdir(bvnet_data_dir):
        os.makedirs(bvnet_data_dir)
    prosail_s2_sim, prosail_vars = load_bvnet_dataset(PATH_TO_DATA_DIR)
    if nb_test_samples > 0:
        test_prosail_s2_sim = prosail_s2_sim[:nb_test_samples,:]
        test_prosail_vars = prosail_vars[:nb_test_samples,:]
        train_prosail_s2_sim = prosail_s2_sim[nb_test_samples:,:]
        train_prosail_vars = prosail_vars[nb_test_samples:,:]
        torch.save(torch.from_numpy(test_prosail_vars), os.path.join(bvnet_data_dir, "/bvnet_test_prosail_sim_vars.pt"))
        torch.save(torch.from_numpy(test_prosail_s2_sim), os.path.join(bvnet_data_dir, "bvnet_test_prosail_s2_sim_refl.pt"))
        # save_dataset(data_dir, "test_", rsr_dir, nb_test_samples, noise, bvnet_bands=True, lai_corr=lai_corr)
    else:
        train_prosail_s2_sim = prosail_s2_sim
        train_prosail_vars = prosail_vars
        save_dataset(data_dir, "test_", rsr_dir, 5000, noise, bvnet_bands=True, 
                     lai_corr=lai_corr, prosail_var_dist_type=prosail_var_dist_type)
    torch.save(torch.from_numpy(train_prosail_vars), os.path.join(bvnet_data_dir, "bvnet_prosail_sim_vars.pt"))
    torch.save(torch.from_numpy(train_prosail_s2_sim), os.path.join(bvnet_data_dir, "bvnet_prosail_s2_sim_refl.pt"))
    
    norm_mean, norm_std = get_refl_normalization(train_prosail_s2_sim)
    torch.save(torch.from_numpy(norm_mean), os.path.join(bvnet_data_dir, "bvnet_norm_mean.pt"))
    torch.save(torch.from_numpy(norm_std), os.path.join(bvnet_data_dir, "bvnet_norm_std.pt"))
    cos_angle_min = torch.tensor([0.342108564072183, 0.979624800125421, -1.0000]) # sun zenith, S2 senith, relative azimuth
    cos_angle_max = torch.tensor([0.9274847491748729, 1.0000, 1.0000])
    cos_angles_loc, cos_angles_scale = min_max_to_loc_scale(cos_angle_min, cos_angle_max)
    torch.save(cos_angles_loc, os.path.join(bvnet_data_dir, f"bvnet_angles_loc.pt"))
    torch.save(cos_angles_scale, os.path.join(bvnet_data_dir, f"bvnet_angles_scale.pt"))
    # torch.save(idx_loc, os.path.join(data_dir, f"bvnet_idx_loc.pt"))
    # torch.save(idx_scale, os.path.join(data_dir, f"bvnet_idx_scale.pt"))

def save_lai_ccc_bvnet_dataset(data_dir, rsr_dir, noise=0, lai_corr=True, nb_test_samples=0, prosail_var_dist_type="legacy"):

    PATH_TO_DATA_DIR = os.path.join(prosailvae.__path__[0], os.pardir) + "/field_data/lai/"
    bvnet_data_dir = os.path.join(data_dir, os.pardir) + "/bvnet_ccc_lai/"
    if not os.path.isdir(bvnet_data_dir):
        os.makedirs(bvnet_data_dir)
    prosail_s2_sim, prosail_vars = load_bvnet_dataset(PATH_TO_DATA_DIR)
    prosail_vars = prosail_vars[:,np.array([6,1,11,12,13])]
    if nb_test_samples > 0:
        test_prosail_s2_sim = prosail_s2_sim[:nb_test_samples,:]
        test_prosail_vars = prosail_vars[:nb_test_samples,:]
        train_prosail_s2_sim = prosail_s2_sim[nb_test_samples:,:]
        train_prosail_vars = prosail_vars[nb_test_samples:,:]
        torch.save(torch.from_numpy(test_prosail_vars), os.path.join(bvnet_data_dir, "/bvnet_test_prosail_sim_vars.pt"))
        torch.save(torch.from_numpy(test_prosail_s2_sim), os.path.join(bvnet_data_dir, "bvnet_test_prosail_s2_sim_refl.pt"))
        # save_dataset(data_dir, "test_", rsr_dir, nb_test_samples, noise, bvnet_bands=True, lai_corr=lai_corr)
    else:
        train_prosail_s2_sim = prosail_s2_sim
        train_prosail_vars = prosail_vars
        save_dataset(data_dir, "test_", rsr_dir, 5000, noise, bvnet_bands=True, lai_corr=lai_corr, prosail_var_dist_type=prosail_var_dist_type)
    torch.save(torch.from_numpy(train_prosail_vars), os.path.join(bvnet_data_dir, "bvnet_prosail_sim_vars.pt"))
    torch.save(torch.from_numpy(train_prosail_s2_sim), os.path.join(bvnet_data_dir, "bvnet_prosail_s2_sim_refl.pt"))
    
    norm_mean, norm_std = get_refl_normalization(train_prosail_s2_sim)
    torch.save(torch.from_numpy(norm_mean), os.path.join(bvnet_data_dir, "bvnet_norm_mean.pt"))
    torch.save(torch.from_numpy(norm_std), os.path.join(bvnet_data_dir, "bvnet_norm_std.pt"))
    cos_angle_min = torch.tensor([0.342108564072183, 0.979624800125421, -1.0000]) # sun zenith, S2 senith, relative azimuth
    cos_angle_max = torch.tensor([0.9274847491748729, 1.0000, 1.0000])
    cos_angles_loc, cos_angles_scale = min_max_to_loc_scale(cos_angle_min, cos_angle_max)
    torch.save(cos_angles_loc, os.path.join(bvnet_data_dir, f"bvnet_angles_loc.pt"))
    torch.save(cos_angles_scale, os.path.join(bvnet_data_dir, f"bvnet_angles_scale.pt"))

def get_data_generation_parser():
    """
    Creates a new argument parser.
    """
    parser = argparse.ArgumentParser(description='Parser for data generation')

    parser.add_argument("-n_samples", "-n", dest="n_samples",
                        help="number of samples in simulated dataset",
                        type=int, default=2000)
    
    parser.add_argument("-file_prefix", "-p", dest="file_prefix",
                        help="number of samples in simulated dataset",
                        type=str, default="")

    parser.add_argument("-prospect_version", "-pv", dest="prospect_version",
                        help="version of PROSPECT model (5, D or PRO)",
                        type=str, default="5")
    
    parser.add_argument("-data_dir", "-d", dest="data_dir",
                        help="number of samples in simulated dataset",
                        type=str, default="/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/sim_data/")
    
    parser.add_argument("-s", dest="noise",
                        help="gaussian noise level on simulated data",
                        type=float, default=0.01)
    
    parser.add_argument("-rsr", dest="rsr_dir",
                        help="directory of rsr_file",
                        type=str, default='/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data')
    
    parser.add_argument("-sa", dest="static_angles",
                        help="Set to True to generate prosail samples with a single angular configuration",
                        type=bool, default=False)
    
    parser.add_argument("-b", dest="bvnet_bands",
                        help="Set to True to generate prosail samples without B2 and B8 for validation with bvnet_dataset",
                        type=bool, default=False)
    
    parser.add_argument("-bd", dest="bvnet_dataset",
                        help="Set to True to generate a training dataset from bvnet data",
                        type=bool, default=False)
    
    parser.add_argument("-dt", dest="dist_type",
                        help="set distribution for prosail parameter sampling (legacy or new)",
                        type=str, default="legacy")

    parser.add_argument("-m", dest="lai_corr_mode",
                        help="co-distribution mode",
                        type=str, default="v2")   
    
    parser.add_argument("-lt", dest="lai_thresh",
                        help="toggle lai threshold for co distribution",
                        type=bool, default=False)     
    
    parser.add_argument("-psa", dest="simulate_with_all_prospect",
                        help="samples prosil parameters and uses all available prospect versions to simulate reflectances",
                        type=bool, default=False)    
    return parser

if  __name__ == "__main__":
    if socket.gethostname()=='CELL200973':
        args=[
            # "-wd", "True",
            #   "-w", "True",
              "-d", "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/sim_data_corr_v2_test_prospect_vD/",
              "-dt", "new_v2",
              "-n", "42000",
              "-m", "v2",
            #   "-pv", "D", 
              "-psa", "True"]
        parser = get_data_generation_parser().parse_args(args)
    else:
        parser = get_data_generation_parser().parse_args()
    if len(parser.data_dir)==0 : 
        data_dir = os.path.join(os.path.join(os.path.dirname(prosailvae.__file__),
                                         os.pardir), "data/")  
    else: 
        data_dir = parser.data_dir
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    lai_thresh = None
    if parser.lai_thresh:
        lai_thresh = 10
    if parser.bvnet_dataset:
        save_lai_ccc_bvnet_dataset(data_dir, parser.rsr_dir, parser.noise, lai_corr=True, prosail_var_dist_type=parser.dist_type)
        save_bvnet_dataset(data_dir, parser.rsr_dir, parser.noise, lai_corr=True, prosail_var_dist_type=parser.dist_type, lai_thresh=lai_thresh)
    if parser.simulate_with_all_prospect:
        save_prosail_data_set_with_all_prospect_versions(data_dir, parser.file_prefix, parser.rsr_dir,
                                                         parser.n_samples, parser.noise, bvnet_bands=parser.bvnet_bands, 
                                                         uniform_mode=False, prosail_var_dist_type=parser.dist_type,
                                                         lai_thresh=lai_thresh)
    else:
        save_dataset(data_dir, parser.file_prefix, parser.rsr_dir, parser.n_samples, parser.noise, 
                     bvnet_bands=parser.bvnet_bands, uniform_mode=False, lai_corr=True, 
                     prosail_var_dist_type=parser.dist_type, lai_corr_mode=parser.lai_corr_mode, 
                     lai_thresh=lai_thresh, prospect_version = parser.prospect_version)

# import matplotlib.pyplot as plt
# from prosailvae.ProsailSimus import ProsailSimulator, SensorSimulator, BANDS, PROSAILVARS
# from matplotlib.colors import LogNorm
# from metrics.metrics_utils import regression_metrics
# fig, ax = plt.subplots(2, 5, dpi=150, tight_layout=True, figsize=(20, 8))
# bins=100
# for i, band in enumerate(BANDS):
#     row = i // 5
#     col = i % 5
#     xmin = min(prosail_s2_sim_5[:, i].min().item(), prosail_s2_sim[:, i].min())
#     xmax = max(prosail_s2_sim_5[:, i].max().item(), prosail_s2_sim[:, i].max())
#     ax[row, col].hist2d(prosail_s2_sim_5[:, i].numpy(), prosail_s2_sim[:, i], range = [[xmin, xmax], [xmin, xmax]], 
#                                                       bins=bins, cmap='viridis', 
#                                                       norm=LogNorm())
#     ax[row, col].set_aspect('equal')
#     ax[row, col].plot([xmin, xmax], [xmin, xmax], 'k')
#     ax[row, col].set_xlabel(f'PROSPECT-5 {band}')
#     ax[row, col].set_ylabel(f'PROSPECT-D {band}')
#     _, _, r2_band, rmse_band = regression_metrics(prosail_s2_sim_5[:, i].reshape(-1).numpy(), 
#                                                       prosail_s2_sim[:, i].reshape(-1))
#     perf_text = "r2: {:.2f} - RMSE: {:.2f}".format(r2_band, rmse_band)
#     ax[row, col].text(.01, .99, perf_text, ha='left', va='top', transform=ax[row, col].transAxes)
# fig.savefig(os.path.join(data_dir, f"5_vs_D_band_scatter.png"))
