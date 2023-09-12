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
from prosailvae.ProsailSimus import SensorSimulator, ProsailSimulator
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
    
    parser.add_argument("-t", dest="third_layer",
                        type=bool, default=False)
    
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
    save_dataset(save_dir, "evaluation_", nb_simus=n_eval, rsr_dir=rsr_dir,weiss_mode=True, 
                 prosail_var_dist_type="new",lai_var_dist=lai_var_dist)

    for i, mu in enumerate(tg_mu):
        for j, sigma in enumerate(tg_sigma):
            lai_var_dist = VariableDistribution(low=lai_low,
                                                high=lai_high,
                                                scale=sigma,
                                                loc=mu,
                                                law="gaussian")
            save_dataset(save_dir, "train_mu_{mu}_sigma_{sigma}_", nb_simus=n_samples_sub, 
                         rsr_dir=rsr_dir, weiss_mode=True, 
                         prosail_var_dist_type="new", lai_var_dist=lai_var_dist)

def main():
    
    simulate_data_sets()

if __name__ == "__main__":
    main()