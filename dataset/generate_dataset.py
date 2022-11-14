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

def sample_prosail_vars(var_dists):
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
    tts = sample_param(var_dists.tts, lai=lai)
    tto = sample_param(var_dists.tto, lai=lai)
    psi = sample_param(var_dists.psi, lai=lai)

    return N, cab, car, cbrown, caw, cm, lai, lidfa, hspot, psoil, rsoil, tts, tto, psi

def simulate_prosail_dataset(data_dir, nb_simus=100):
    prosail_vars = np.zeros((nb_simus, 14))
    prosail_s2_sim = np.zeros((nb_simus, 10))
    with numpyro.handlers.seed(rng_seed=5):
        for i in range(nb_simus):
            prosail_vars[i,:] = sample_prosail_vars(ProsailVarsDist)
            sigma = numpyro.sample("sigma", dist.Uniform(0., 0.01))
            psimulator = ProsailSimulator()
            ssimulator = SensorSimulator(data_dir + "/sentinel2.rsr")
            mean = ssimulator(psimulator(torch.from_numpy(prosail_vars[i,:]).view(1,-1).float())).numpy()
            prosail_s2_sim[i,:] = numpyro.sample("obs", dist.Normal(mean, sigma))
    return prosail_vars, prosail_s2_sim

def save_dataset(save_dir, data_file_prefix, nb_simus):
    prosail_vars, prosail_s2_sim = simulate_prosail_dataset(data_dir,
                                                            nb_simus=nb_simus)
    
    torch.save(torch.from_numpy(prosail_vars), 
               save_dir + data_file_prefix + "prosail_sim_vars.pt") 
    
    torch.save(torch.from_numpy(prosail_s2_sim), 
               save_dir + data_file_prefix + "prosail_s2_sim_refl.pt") 
    

def get_data_generation_parser():
    """
    Creates a new argument parser.
    """
    parser = argparse.ArgumentParser(description='Parser for data generation')

    parser.add_argument("-n_samples", "-n", dest="n_samples",
                        help="number of samples in simulated dataset",
                        type=int, default=10000)
    parser.add_argument("-file_prefix", "-p", dest="file_prefix",
                        help="number of samples in simulated dataset",
                        type=str, default="")
    parser.add_argument("-data_dir", "-d", dest="data_dir",
                        help="number of samples in simulated dataset",
                        type=str, default="")
    return parser

if  __name__ == "__main__":
    parser = get_data_generation_parser().parse_args()
    if len(parser.data_dir)==0 : 
        data_dir = os.path.join(os.path.join(os.path.dirname(prosailvae.__file__),
                                         os.pardir), "data/")  
    else: 
        data_dir = parser.data_dir
    save_dataset(data_dir, parser.file_prefix, parser.n_samples)
    