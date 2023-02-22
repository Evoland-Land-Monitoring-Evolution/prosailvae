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

import torch
import os 
import argparse
import prosailvae
from prosailvae.ProsailSimus import ProsailSimulator, SensorSimulator, ProsailVarsDist
from tqdm import trange
import scipy.stats as stats

PROSAIL_VARS = [
    "N", "cab", "car", "cbrown", "caw", "cm",
    "lai", "lidfa", "hspot", "psoil", "rsoil"
]
def correlate_with_lai(lai, V, V_mean, lai_conv):
    return V_mean + (V - V_mean) * (lai_conv - lai) / lai_conv

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


def np_sample_param(param_dist, lai=None, n_samples=1, uniform_mode=True):
    if param_dist[5] == "uniform" or uniform_mode:
        sample = np.random.uniform(low=param_dist[0], high=param_dist[1], size=(n_samples))
    elif param_dist[5] == "gaussian":
        sample = stats.truncnorm((param_dist[0] - param_dist[2]) / param_dist[3], (param_dist[1] - param_dist[2]) / param_dist[3], 
                            loc=param_dist[2], scale=param_dist[3]).rvs(n_samples)
    elif param_dist[5] == "lognormal":
        low = param_dist[0]
        if param_dist[0] == 0:
            low=1e-16
        X = stats.truncnorm((low - param_dist[2]) / param_dist[3], (np.log(param_dist[1]) - param_dist[2]) / param_dist[3], 
                            loc=param_dist[2], scale=param_dist[3]).rvs(n_samples)
        sample = np.exp(X)
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

def sample_prosail_vars(var_dists, static_angles=False):
    lai = sample_param(var_dists.lai, lai=None)
    N = sample_param(var_dists.N, lai=lai)
    cab = sample_param(var_dists.cab, lai=lai)
    car = sample_param(var_dists.car, lai=lai)
    cbrown = sample_param(var_dists.cbrown, lai=lai)
    caw = sample_param(var_dists.caw, lai=lai)
    cm = sample_param(var_dists.cm, lai=lai)
    lidfa = sample_param(var_dists.lidfa, lai=lai)
    hspot = sample_param(var_dists.hspot, lai=lai)
    psoil = sample_param(var_dists.psoil, lai=lai)
    rsoil = sample_param(var_dists.rsoil, lai=lai)
    if static_angles:
        tts = np.array([48.0])
        tto = np.array([5.0])
        psi = np.array([-56])
    else:
        tts = sample_param(var_dists.tts, lai=lai)
        tto = sample_param(var_dists.tto, lai=lai)
        psi = sample_param(var_dists.psi, lai=lai)

    return N, cab, car, cbrown, caw, cm, lai, lidfa, hspot, psoil, rsoil, tts, tto, psi

def partial_sample_prosail_vars(var_dists, lai=None, tts=None, tto=None, psi=None, n_samples=1, uniform_mode=True):
    prosail_vars = np.zeros((n_samples, 14))
    if lai is None:
        lai = np_sample_param(var_dists.lai, lai=None, n_samples=n_samples, uniform_mode=uniform_mode)
    prosail_vars[:,6] = lai
    prosail_vars[:,0] = np_sample_param(var_dists.N, lai=None, n_samples=n_samples, uniform_mode=uniform_mode)
    prosail_vars[:,1] = np_sample_param(var_dists.cab, lai=None, n_samples=n_samples, uniform_mode=uniform_mode)
    prosail_vars[:,2] = np_sample_param(var_dists.car, lai=None, n_samples=n_samples, uniform_mode=uniform_mode)
    prosail_vars[:,3] = np_sample_param(var_dists.cbrown, lai=None, n_samples=n_samples, uniform_mode=uniform_mode)
    prosail_vars[:,4] = np_sample_param(var_dists.caw, lai=None, n_samples=n_samples, uniform_mode=uniform_mode)
    prosail_vars[:,5] = np_sample_param(var_dists.cm, lai=None, n_samples=n_samples, uniform_mode=uniform_mode)
    prosail_vars[:,7] = np_sample_param(var_dists.lidfa, lai=None, n_samples=n_samples, uniform_mode=uniform_mode)
    prosail_vars[:,8] = np_sample_param(var_dists.hspot, lai=None, n_samples=n_samples, uniform_mode=uniform_mode)
    prosail_vars[:,9] = np_sample_param(var_dists.psoil, lai=None, n_samples=n_samples, uniform_mode=uniform_mode)
    prosail_vars[:,10] = np_sample_param(var_dists.rsoil, lai=None, n_samples=n_samples, uniform_mode=uniform_mode)

    if tts is None:
        tts = np_sample_param(var_dists.tts, lai=None, n_samples=n_samples, uniform_mode=uniform_mode)
    if tto is None:
        tto = np_sample_param(var_dists.tto, lai=None, n_samples=n_samples, uniform_mode=uniform_mode)
    if psi is None:
        psi = np_sample_param(var_dists.psi, lai=None, n_samples=n_samples, uniform_mode=uniform_mode)
    prosail_vars[:,11] = tts
    prosail_vars[:,12] = tto
    prosail_vars[:,13] = psi

    return prosail_vars

def simulate_prosail_dataset(data_dir, nb_simus=100, noise=0, rsr_dir='', static_angles=False):
    prosail_vars = np.zeros((nb_simus, 14))
    prosail_s2_sim = np.zeros((nb_simus, 10))
    with numpyro.handlers.seed(rng_seed=5):
        for i in trange(nb_simus):
            prosail_vars[i,:] = sample_prosail_vars(ProsailVarsDist, static_angles=static_angles)
            
            psimulator = ProsailSimulator()
            ssimulator = SensorSimulator(rsr_dir + "/sentinel2.rsr")
            mean = ssimulator(psimulator(torch.from_numpy(prosail_vars[i,:]).view(1,-1).float())).numpy()
            if noise>0:
                sigma = numpyro.sample("sigma", dist.Uniform(0., noise))
                prosail_s2_sim[i,:] = numpyro.sample("obs", dist.Normal(mean, sigma))
            else:
                prosail_s2_sim[i,:] = mean
    return prosail_vars, prosail_s2_sim

def simulate_prosail_samples_close_to_ref(s2_r_ref, noise=0, rsr_dir='', lai=None, tts=None, 
                                          tto=None, psi=None, eps_mae = 1e-3, max_iter=100, samples_per_iter=1024):
    best_prosail_vars = np.ones((1, 14))
    best_prosail_s2_sim = np.ones((1, 10))
    best_mae = np.inf
    iter = 0
    psimulator = ProsailSimulator()
    ssimulator = SensorSimulator(rsr_dir + "/sentinel2.rsr")
    bins=200
    aggregate_s2_hist = np.zeros((bins, 10))
    with numpyro.handlers.seed(rng_seed=5):
        while best_mae > eps_mae and iter < max_iter :
            if iter%10==0:
                print(f"{iter} - {best_mae}")
            prosail_vars = partial_sample_prosail_vars(ProsailVarsDist, lai=lai, tts=tts, tto=tto, psi=psi, n_samples=samples_per_iter)
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

    return best_prosail_vars, best_prosail_s2_sim, max_iter * samples_per_iter, aggregate_s2_hist

def get_refl_normalization(prosail_refl):
    return prosail_refl.mean(0), prosail_refl.std(0)

def save_dataset(data_dir, data_file_prefix, rsr_dir, nb_simus, noise=0, static_angles=False):
    prosail_vars, prosail_s2_sim = simulate_prosail_dataset(data_dir,
                                                            nb_simus=nb_simus, 
                                                            noise=noise,
                                                            rsr_dir=rsr_dir,
                                                            static_angles=static_angles)
    norm_mean, norm_std = get_refl_normalization(prosail_s2_sim)
    torch.save(torch.from_numpy(prosail_vars), 
               data_dir + data_file_prefix + "prosail_sim_vars.pt") 
    
    torch.save(torch.from_numpy(prosail_s2_sim), 
               data_dir + data_file_prefix + "prosail_s2_sim_refl.pt") 
    torch.save(norm_mean, parser.data_dir + data_file_prefix+ "norm_mean.pt") 
    torch.save(norm_std, parser.data_dir + data_file_prefix + "norm_std.pt") 
    

def get_data_generation_parser():
    """
    Creates a new argument parser.
    """
    parser = argparse.ArgumentParser(description='Parser for data generation')

    parser.add_argument("-n_samples", "-n", dest="n_samples",
                        help="number of samples in simulated dataset",
                        type=int, default=1000)
    parser.add_argument("-file_prefix", "-p", dest="file_prefix",
                        help="number of samples in simulated dataset",
                        type=str, default="small_test_")
    parser.add_argument("-data_dir", "-d", dest="data_dir",
                        help="number of samples in simulated dataset",
                        type=str, default="/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data")
    parser.add_argument("-s", dest="noise",
                        help="gaussian noise level on simulated data",
                        type=float, default=0)
    parser.add_argument("-rsr", dest="rsr_dir",
                        help="directory of rsr_file",
                        type=str, default='/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data')
    parser.add_argument("-sa", dest="static_angles",
                        help="Set to True to generate prosail samples with a single angular configuration",
                        type=bool, default=False)
    return parser

if  __name__ == "__main__":
    parser = get_data_generation_parser().parse_args()
    if len(parser.data_dir)==0 : 
        data_dir = os.path.join(os.path.join(os.path.dirname(prosailvae.__file__),
                                         os.pardir), "data/")  
    else: 
        data_dir = parser.data_dir
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    save_dataset(data_dir, parser.file_prefix,parser. rsr_dir,
                 parser.n_samples, parser.noise, static_angles=parser.static_angles)
    