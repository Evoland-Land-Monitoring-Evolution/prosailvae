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
from prosailvae.ProsailSimus import ProsailSimulator, SensorSimulator
from prosailvae.prosail_var_dists import get_prosail_var_dist
from dataset.bvnet_dataset import load_bvnet_dataset
from tqdm import trange
import scipy.stats as stats
from dataset.dataset_utils import min_max_to_loc_scale
from prosailvae.spectral_indices import get_spectral_idx
import socket
from prosailvae.prosail_var_dists import VariableDistribution

PROSAILVARS = [
    "N", "cab", "car", "cbrown", "cw", "cm",
    "lai", "lidfa", "hspot", "psoil", "rsoil"
]
def correlate_with_lai(lai, V, V_mean, lai_conv):
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

def sample_param(param_dist, lai=None):
    if param_dist[5] == "uniform":
        sample = numpyro.sample(param_dist[6], dist.Uniform(param_dist[0], param_dist[1]))
    elif param_dist[5] == "gaussian":
        
        sample = numpyro.sample(param_dist[6], 
                                dist.TwoSidedTruncatedDistribution(dist.Normal(param_dist[2], param_dist[3]),
                                                                   low=param_dist[0], high=param_dist[1]))
    elif param_dist[5] == "lognormal":
        low = param_dist[0]
        if param_dist[0] == 0:
            low=1e-16
        sample = numpyro.sample(param_dist[6], dist.TwoSidedTruncatedDistribution(dist.Normal(param_dist[2], param_dist[3]),
                                           low=low, 
                                           high=np.log(param_dist[1])))
        sample = jnp.exp(sample)
    else:
        raise NotImplementedError("Please choose sample distribution among gaussian, uniform and lognormal")
    if lai is not None and param_dist[4] is not None:
        if param_dist[2] is None : 
            sample = correlate_with_lai(lai, sample, 
                                        (param_dist[1] - param_dist[0])/2, 
                                        param_dist[4])
        else:
            sample = correlate_with_lai(lai, sample, param_dist[2],  param_dist[4])
    return sample


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
        # if param_dist.low == 0:
        #     low=1e-16
        X = stats.truncnorm((np.log(low) - param_dist.loc) / param_dist.scale, (np.log(param_dist.high) - param_dist.loc) / param_dist.scale, 
                            loc=param_dist.loc, scale=param_dist.scale).rvs(n_samples)
        sample = np.exp(X)
    else:
        raise NotImplementedError("Please choose sample distribution among gaussian, uniform and lognormal")
    if lai_corr_mode=="v2":
        if param_dist.C_lai_min is not None:
            sample = correlate_with_lai_V2(lai, sample, param_dist.low, param_dist.high, param_dist.C_lai_min, 
                                           param_dist.C_lai_max, lai_max=lai_max, lai_thresh=lai_thresh)
    else:
        if lai is not None and (param_dist.lai_conv is not None or lai_conv_override is not None):
            lai_conv = lai_conv_override if lai_conv_override is not None else param_dist.lai_conv
            if param_dist.loc is None : 
                sample = correlate_with_lai(lai, sample, 
                                            (param_dist.high - param_dist.low)/2, 
                                            lai_conv)
            else:
                sample = correlate_with_lai(lai, sample, param_dist.loc, lai_conv)
    return sample

# def sample_prosail_vars(var_dists, static_angles=False):
#     lai = sample_param(var_dists.lai, lai=None)
#     N = sample_param(var_dists.N, lai=lai)
#     cab = sample_param(var_dists.cab, lai=lai)
#     car = sample_param(var_dists.car, lai=lai)
#     cbrown = sample_param(var_dists.cbrown, lai=lai)
#     caw = sample_param(var_dists.caw, lai=lai)
#     cm = sample_param(var_dists.cm, lai=lai)
#     lidfa = sample_param(var_dists.lidfa, lai=lai)
#     hspot = sample_param(var_dists.hspot, lai=lai)
#     psoil = sample_param(var_dists.psoil, lai=lai)
#     rsoil = sample_param(var_dists.rsoil, lai=lai)
#     if static_angles:
#         tts = np.array([48.0])
#         tto = np.array([5.0])
#         psi = np.array([-56])
#     else:
#         tts = sample_param(var_dists.tts, lai=lai)
#         tto = sample_param(var_dists.tto, lai=lai)
#         psi = sample_param(var_dists.psi, lai=lai)

#     return N, cab, car, cbrown, caw, cm, lai, lidfa, hspot, psoil, rsoil, tts, tto, psi

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
    prosail_vars[:,0] = np_sample_param(var_dists.N, lai=lai if lai_corr else None, n_samples=n_samples, uniform_mode=uniform_mode, lai_conv_override=lai_conv_override, lai_max=var_dists.lai.high, lai_corr_mode=lai_corr_mode, lai_thresh=lai_thresh)
    prosail_vars[:,1] = np_sample_param(var_dists.cab, lai=lai if lai_corr else None, n_samples=n_samples, uniform_mode=uniform_mode, lai_conv_override=lai_conv_override, lai_max=var_dists.lai.high, lai_corr_mode=lai_corr_mode, lai_thresh=lai_thresh)
    prosail_vars[:,2] = np_sample_param(var_dists.car, lai=lai if lai_corr else None, n_samples=n_samples, uniform_mode=uniform_mode, lai_conv_override=lai_conv_override, lai_max=var_dists.lai.high, lai_corr_mode=lai_corr_mode, lai_thresh=lai_thresh)
    prosail_vars[:,3] = np_sample_param(var_dists.cbrown, lai=lai if lai_corr else None, n_samples=n_samples, uniform_mode=uniform_mode, lai_conv_override=lai_conv_override, lai_max=var_dists.lai.high, lai_corr_mode=lai_corr_mode, lai_thresh=lai_thresh)
    prosail_vars[:,4] = np_sample_param(var_dists.cw, lai=lai if lai_corr else None, n_samples=n_samples, uniform_mode=uniform_mode, lai_conv_override=lai_conv_override, lai_max=var_dists.lai.high, lai_corr_mode=lai_corr_mode, lai_thresh=lai_thresh)
    prosail_vars[:,5] = np_sample_param(var_dists.cm, lai=lai if lai_corr else None, n_samples=n_samples, uniform_mode=uniform_mode, lai_conv_override=lai_conv_override, lai_max=var_dists.lai.high, lai_corr_mode=lai_corr_mode, lai_thresh=lai_thresh)
    prosail_vars[:,7] = np_sample_param(var_dists.lidfa, lai=lai if lai_corr else None, n_samples=n_samples, uniform_mode=uniform_mode, lai_conv_override=lai_conv_override, lai_max=var_dists.lai.high, lai_corr_mode=lai_corr_mode, lai_thresh=lai_thresh)
    prosail_vars[:,8] = np_sample_param(var_dists.hspot, lai=lai if lai_corr else None, n_samples=n_samples, uniform_mode=uniform_mode, lai_conv_override=lai_conv_override, lai_max=var_dists.lai.high, lai_corr_mode=lai_corr_mode, lai_thresh=lai_thresh)
    prosail_vars[:,9] = np_sample_param(var_dists.psoil, lai=lai if lai_corr else None, n_samples=n_samples, uniform_mode=uniform_mode, lai_conv_override=lai_conv_override, lai_max=var_dists.lai.high, lai_corr_mode=lai_corr_mode, lai_thresh=lai_thresh)
    prosail_vars[:,10] = np_sample_param(var_dists.rsoil, lai=lai if lai_corr else None, n_samples=n_samples, uniform_mode=uniform_mode, lai_conv_override=lai_conv_override, lai_max=var_dists.lai.high, lai_corr_mode=lai_corr_mode, lai_thresh=lai_thresh)

    # if tts is None:
    #     tts = np_sample_param(var_dists.tts, lai=None, n_samples=n_samples, uniform_mode=uniform_mode)
    # if tto is None:
    #     tto = np_sample_param(var_dists.tto, lai=None, n_samples=n_samples, uniform_mode=uniform_mode)
    # if psi is None:
    #     psi = np_sample_param(var_dists.psi, lai=None, n_samples=n_samples, uniform_mode=uniform_mode)
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

def np_simulate_prosail_dataset(nb_simus=2048, noise=0, psimulator=None, ssimulator=None, 
                                n_samples_per_batch=1024, uniform_mode=False, lai_corr=True,
                                prosail_var_dist_type="legacy", lai_var_dist:VariableDistribution|None=None,
                                lai_corr_mode="v2", lai_thresh=None):
    prosail_vars = np.zeros((nb_simus, 14))
    prosail_s2_sim = np.zeros((nb_simus, ssimulator.rsr.size(1)))
    prosail_var_dist = get_prosail_var_dist(prosail_var_dist_type)
    n_full_batch = nb_simus // n_samples_per_batch
    last_batch = nb_simus - nb_simus // n_samples_per_batch * n_samples_per_batch

    for i in range(n_full_batch):
        prosail_vars[i*n_samples_per_batch : (i+1) * n_samples_per_batch,
                     :] = partial_sample_prosail_vars(prosail_var_dist, n_samples=n_samples_per_batch, 
                                                      uniform_mode=uniform_mode, lai_corr=lai_corr, 
                                                      lai_var_dist=lai_var_dist, lai_corr_mode=lai_corr_mode,
                                                      lai_thresh=lai_thresh)
        prosail_r = psimulator(torch.from_numpy(prosail_vars[i*n_samples_per_batch : (i+1) * n_samples_per_batch,:]).view(n_samples_per_batch,-1).float())
        sim_s2_r = ssimulator(prosail_r).numpy()
        if noise>0:
            sigma = np.random.rand(n_samples_per_batch,1) * noise * np.ones_like(sim_s2_r)
            add_noise = np.random.normal(loc = np.zeros_like(sim_s2_r), scale=sigma, size=sim_s2_r.shape)
            sim_s2_r += add_noise
        prosail_s2_sim[i*n_samples_per_batch : (i+1) * n_samples_per_batch,:] = sim_s2_r
    if last_batch > 0:
        prosail_vars[n_full_batch*n_samples_per_batch:,
                     :] = partial_sample_prosail_vars(prosail_var_dist, n_samples=last_batch, uniform_mode=uniform_mode, 
                                                      lai_corr=lai_corr, lai_var_dist=lai_var_dist, lai_corr_mode=lai_corr_mode,
                                                      lai_thresh=lai_thresh)
        sim_s2_r = ssimulator(psimulator(torch.from_numpy(prosail_vars[n_full_batch*n_samples_per_batch:,:]).view(last_batch,-1).float())).numpy()
        if noise>0:
            sigma = np.random.rand(last_batch,1) * noise * np.ones_like(sim_s2_r)
            add_noise = np.random.normal(loc = np.zeros_like(sim_s2_r), scale=sigma, size=sim_s2_r.shape)
            sim_s2_r += add_noise
        prosail_s2_sim[n_full_batch*n_samples_per_batch:,:] = sim_s2_r
    return prosail_vars, prosail_s2_sim


def simulate_prosail_samples_close_to_ref(s2_r_ref, noise=0, psimulator=None, ssimulator=None, lai=None, tts=None, 
                                          tto=None, psi=None, eps_mae = 1e-3, max_iter=100, 
                                          samples_per_iter=1024, prosail_var_dist_type="legacy"):
    best_prosail_vars = np.ones((1, 14))
    best_prosail_s2_sim = np.ones((1, 10))
    best_mae = np.inf
    iter = 0
    bins=200
    aggregate_s2_hist = np.zeros((bins, 10))
    prosail_var_dist = get_prosail_var_dist(prosail_var_dist_type)
    with numpyro.handlers.seed(rng_seed=5):
        while best_mae > eps_mae and iter < max_iter :
            if iter%10==0:
                print(f"{iter} - {best_mae}")
            prosail_vars = partial_sample_prosail_vars(prosail_var_dist, lai=lai, tts=tts, tto=tto, psi=psi, n_samples=samples_per_iter)
            prosail_s2_sim = ssimulator(psimulator(torch.from_numpy(prosail_vars).view(-1,14).float().detach())).numpy()
            
            aggregate_s2_hist += np.apply_along_axis(lambda a: np.histogram(a, bins=bins, range=[0.0, 1.0])[0], 0, prosail_s2_sim)
            if noise > 0:
                raise NotImplementedError
            mare = np.abs((s2_r_ref - prosail_s2_sim)/(s2_r_ref+1e-8)).mean(1)
            best_mae_iter = mare.min()

            if best_mae_iter < best_mae:
                best_mae = best_mae_iter
                best_prosail_vars = prosail_vars[mare.argmin(),:]
                best_prosail_s2_sim = prosail_s2_sim[mare.argmin(),:]
            iter += 1
    if iter==max_iter:
        print(f"WARNING : No sample with mae better than {eps_mae} was generated in {max_iter} iterations with {samples_per_iter} samples each ({max_iter * samples_per_iter} samples) ")    
    else:
        print(f"A sample with mae better than {eps_mae} was generated in {max_iter} iterations with {samples_per_iter} samples each ({max_iter * samples_per_iter} samples) ")    

    return best_prosail_vars, best_prosail_s2_sim, max_iter * samples_per_iter, aggregate_s2_hist, best_mae


def simulate_lai_with_rec_error_hist(s2_r_ref, noise=0, psimulator=None, ssimulator=None, lai=None, tts=None, 
                                          tto=None, psi=None, max_iter=100, 
                                          samples_per_iter=1024, log_err = True, uniform_mode=True, lai_corr=False, 
                                          lai_conv_override=None, bvnet_bands=False, prosail_var_dist_type="legacy"):
    
    prosail_var_dist = get_prosail_var_dist(prosail_var_dist_type)
    best_mae = np.inf
    iter = 0
    bins=200
    aggregate_lai_hist = np.zeros((bins, 1))
    heatmap=0
    min_lai = prosail_var_dist.lai.low
    max_lai = prosail_var_dist.lai.high
    min_err = 0
    max_err = 2
    n_bin_err = 100
    n_bin_lai = 200
    xedges = np.linspace(min_lai, max_lai, n_bin_lai)
    if log_err:
        max_err=5
        min_err = 5e-3
        yedges = np.logspace(np.log10(min_err), np.log10(max_err), n_bin_err)
    else:
        yedges = np.linspace(min_err, max_err, n_bin_err)
    with numpyro.handlers.seed(rng_seed=5):
        for iter in range(max_iter) :
            if iter%10==0:
                print(f"{iter} - {best_mae}")
            prosail_vars = partial_sample_prosail_vars(prosail_var_dist, lai=lai, 
                                                       tts=tts, tto=tto, psi=psi, n_samples=samples_per_iter, 
                                                       uniform_mode=uniform_mode, lai_corr=lai_corr, lai_conv_override=lai_conv_override)
            lai_sim = prosail_vars[:,6]
            prosail_s2_sim = ssimulator(psimulator(torch.from_numpy(prosail_vars).view(-1,14).float().detach())).numpy()
            aggregate_lai_hist += np.histogram(lai_sim, bins=n_bin_lai, range=[min_lai, max_lai])[0].reshape(-1,1)
            if noise > 0:
                raise NotImplementedError
            if bvnet_bands:
                mare = np.abs((s2_r_ref[:,[1,2,3,4,5,7,8,9]] - prosail_s2_sim[:,[1,2,3,4,5,7,8,9]])/(s2_r_ref[:,[1,2,3,4,5,7,8,9]]+1e-8)).mean(1)
            else:
                mare = np.abs((s2_r_ref - prosail_s2_sim)/(s2_r_ref+1e-8)).mean(1)
            xs = lai_sim
            ys = mare
            hist, xedges, yedges = np.histogram2d(
                xs, ys, bins=[xedges, yedges])
            heatmap += hist
            best_mae_iter = mare.min()
            if best_mae_iter < best_mae:
                best_mae = best_mae_iter
                best_prosail_vars = prosail_vars[mare.argmin(),:]
                best_prosail_s2_sim = prosail_s2_sim[mare.argmin(),:]
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return best_prosail_vars, best_prosail_s2_sim, heatmap, extent, aggregate_lai_hist, best_mae


def simulate_lai_with_rec_error_hist_with_enveloppe(s2_r_ref, noise=0, psimulator=None, ssimulator=None, lai=None, tts=None, 
                                                    tto=None, psi=None, max_iter=100, 
                                                    samples_per_iter=1024, log_err = True, uniform_mode=True, lai_corr=False, 
                                                    lai_conv_override=None, bvnet_bands=False, sigma=2, prosail_var_dist_type="legacy"):
    AD=0.01
    MD=2  
    prosail_var_dist = get_prosail_var_dist(prosail_var_dist_type)
    best_mae = np.inf
    iter = 0
    bins=200
    aggregate_lai_hist = np.zeros((bins, 1))
    heatmap=0
    min_lai = prosail_var_dist.lai.low
    max_lai = prosail_var_dist.lai.high
    min_err = 0
    max_err = 2
    n_bin_err = 100
    n_bin_lai = 200
    xedges = np.linspace(min_lai, max_lai, n_bin_lai)
    all_cases_in_enveloppe_err = []
    all_cases_in_enveloppe_LAI = []
    if log_err:
        max_err=5
        min_err = 5e-3
        yedges = np.logspace(np.log10(min_err), np.log10(max_err), n_bin_err)
    else:
        yedges = np.linspace(min_err, max_err, n_bin_err)
    with numpyro.handlers.seed(rng_seed=5):
        for iter in range(max_iter) :
            if iter%10==0:
                print(f"{iter} - {best_mae}")
            prosail_vars = partial_sample_prosail_vars(prosail_var_dist, lai=lai, 
                                                       tts=tts, tto=tto, psi=psi, n_samples=samples_per_iter, 
                                                       uniform_mode=uniform_mode, lai_corr=lai_corr, lai_conv_override=lai_conv_override)
            lai_sim = prosail_vars[:,6]
            prosail_s2_sim = ssimulator(psimulator(torch.from_numpy(prosail_vars).view(-1,14).float().detach())).numpy()

            aggregate_lai_hist += np.histogram(lai_sim, bins=n_bin_lai, range=[min_lai, max_lai])[0].reshape(-1,1)
            if noise > 0:
                raise NotImplementedError
            if bvnet_bands:
                mare = np.abs((s2_r_ref[:,[1,2,3,4,5,7,8,9]] - prosail_s2_sim[:,[1,2,3,4,5,7,8,9]])/(s2_r_ref[:,[1,2,3,4,5,7,8,9]]+1e-8)).mean(1)
                enveloppe_low = s2_r_ref[:,[1,2,3,4,5,7,8,9]] * (1 - sigma * MD / 100 ) - sigma * AD
                enveloppe_high = s2_r_ref[:,[1,2,3,4,5,7,8,9]] * (1 + sigma * MD / 100 ) + sigma * AD
                cases_in_sigma_enveloppe = np.logical_and((prosail_s2_sim[:,[1,2,3,4,5,7,8,9]] < enveloppe_high).all(1),
                                                           (prosail_s2_sim[:,[1,2,3,4,5,7,8,9]] > enveloppe_low).all(1))            
            else:
                mare = np.abs((s2_r_ref - prosail_s2_sim)/(s2_r_ref+1e-8)).mean(1)
                enveloppe_low = s2_r_ref * (1 - sigma * MD / 100 ) - sigma * AD
                enveloppe_high = s2_r_ref * (1 + sigma * MD / 100 ) + sigma * AD
                cases_in_sigma_enveloppe = np.logical_and((prosail_s2_sim  < enveloppe_high).all(1), 
                                                          (prosail_s2_sim > enveloppe_low).all(1))

            if cases_in_sigma_enveloppe.any():
                all_cases_in_enveloppe_err.append(mare[cases_in_sigma_enveloppe].reshape(-1,1))
                all_cases_in_enveloppe_LAI.append(prosail_vars[cases_in_sigma_enveloppe, 6].reshape(-1,1))
            xs = lai_sim
            ys = mare
            hist, xedges, yedges = np.histogram2d(
                xs, ys, bins=[xedges, yedges])
            heatmap += hist
            best_mae_iter = mare.min()
            if best_mae_iter < best_mae:
                best_mae = best_mae_iter
                best_prosail_vars = prosail_vars[mare.argmin(),:]
                best_prosail_s2_sim = prosail_s2_sim[mare.argmin(),:]
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    if len(all_cases_in_enveloppe_err) > 0:
        all_cases_in_enveloppe_err = np.concatenate(all_cases_in_enveloppe_err, axis=0)
        all_cases_in_enveloppe_LAI = np.concatenate(all_cases_in_enveloppe_LAI, axis=0)
    return best_prosail_vars, best_prosail_s2_sim, heatmap, extent, aggregate_lai_hist, best_mae, all_cases_in_enveloppe_err, all_cases_in_enveloppe_LAI




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
                 lai_corr_mode="v2", lai_thresh=None):

    psimulator = ProsailSimulator()
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
    return parser

if  __name__ == "__main__":
    if socket.gethostname()=='CELL200973':
        args=[
            # "-wd", "True",
            #   "-w", "True",
              "-d", "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/sim_data_corr_v2_test/",
              "-dt", "new_v2",
              "-n", "40000",
              "-m", "v2"]
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
    else:
        save_dataset(data_dir, parser.file_prefix, parser.rsr_dir,
                        parser.n_samples, parser.noise, bvnet_bands=parser.bvnet_bands, 
                        uniform_mode=False, lai_corr=True, prosail_var_dist_type=parser.dist_type,
                        lai_corr_mode=parser.lai_corr_mode, lai_thresh=lai_thresh)


    