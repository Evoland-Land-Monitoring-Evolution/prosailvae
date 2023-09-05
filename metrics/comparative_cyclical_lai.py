import socket
import os
import pandas as pd
import torch
from dataset.project_s2_dataset import save_common_cyclical_dataset, load_cyclical_data_set
import argparse
import numpy as np
from utils.utils import load_dict, load_standardize_coeffs
from prosailvae.ProsailSimus import get_bands_idx
from prosailvae.prosail_vae import (load_prosail_vae_with_hyperprior, get_prosail_vae_config, load_params)
from dataset.loaders import  get_train_valid_test_loader_from_patches

def get_parser():
    """
    Gets arguments for terminal-based launch of script
    """
    parser = argparse.ArgumentParser(description='Parser for data generation')

    parser.add_argument("-m", dest="model_dict_path",
                        help="path to model dict file",
                        type=str, default="")
    parser.add_argument("-d", dest="data_dir",
                        help="path to model dict file",
                        type=str, default="")
    parser.add_argument("-rsr", dest="rsr_dir",
                        help="directory of rsr_file",
                        type=str, default='/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/')
    parser.add_argument("-r", dest="res_dir",
                        help="path to results directory",
                        type=str, default="")
    parser.add_argument("-sp", dest="save_projected",
                        help="save projected data_set from_all_models",
                        type=bool, default=False)
    return parser


def get_models_validation_cyclical_rmse(model_dict, loader, lai_precomputed=False):
    cyclical_rmse = []
    with torch.no_grad():
        for _, model_info in model_dict.items(): 
            model_info['cyclical_rmse'] = model_info["model"].get_cyclical_rmse_from_loader(loader, 
                                                                                            lai_precomputed=lai_precomputed).detach().cpu().item()
            cyclical_rmse.append(model_info['cyclical_rmse'])
    return cyclical_rmse

def get_models_validation_rec_loss(model_dict, loader):
    losses = []
    for model_name, model_info in model_dict.items(): 
        if model_info["supervised"]:
            model_info['loss'] = 0.0
        else:
            loss_dict = model_info["model"].validate(loader, n_samples=10 if not socket.gethostname()=='CELL200973' else 2)
            if "rec_loss" in loss_dict.keys():
                model_info['loss'] = loss_dict["rec_loss"]
            else:
                model_info['loss'] = loss_dict["loss_sum"]
        losses.append(model_info['loss'])
    return losses

def get_model_and_dataloader(parser):
    """
    Get test data (patches) in a loader and loads all trained models
    """
    _, valid_loader, test_loader = get_train_valid_test_loader_from_patches(parser.data_dir,
                                                                 bands = torch.arange(10),
                                                                 batch_size=1, num_workers=0)
    model_dict = load_dict(parser.model_dict_path)
    for model_name, model_info in model_dict.items():
        if model_info["type"] == "simvae":
            config = load_params(model_info["dir_path"], "config.json")
            bands, prosail_bands = get_bands_idx(config["weiss_bands"])
            params_path = os.path.join(model_info["dir_path"], "prosailvae_weights.tar")
            config["load_model"] = True
            model_info["supervised"] = config["supervised"]
            config["vae_load_file_path"] = params_path
            io_coeffs = load_standardize_coeffs(model_info["dir_path"])
            pv_config = get_prosail_vae_config(config, bands=bands, prosail_bands=prosail_bands,
                                                inference_mode = False, rsr_dir=parser.rsr_dir,
                                                io_coeffs=io_coeffs)
            model = load_prosail_vae_with_hyperprior(pv_config=pv_config, pv_config_hyper=None,
                                                     logger_name="No logger")
            model_info["model"] = model
    info_test_data = np.load(os.path.join(parser.data_dir,"test_info.npy"))
    return model_dict, test_loader, valid_loader, info_test_data

def main():
    """
    main.
    """
    if socket.gethostname()=='CELL200973':
        args = ["-m","/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/config/model_dict_dev.json",
                "-d", "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/patches/",
                "-r", "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/results/comparaison/",
                # "-sp", "True",
                ]
        parser = get_parser().parse_args(args)
        projected_data_dir =  "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/projected_data"
    else:
        parser = get_parser().parse_args()
        projected_data_dir =  "/work/scratch/zerahy/prosailvae/data/projected_data"
    res_dir = parser.res_dir
    if not os.path.isdir(res_dir):
        os.makedirs(res_dir)

    model_dict, test_loader, valid_loader, info_test_data = get_model_and_dataloader(parser)
    if parser.save_projected:
        save_common_cyclical_dataset(model_dict, valid_loader, projected_data_dir)
    cyclical_loader = load_cyclical_data_set(projected_data_dir, batch_size=1)
    cyclical_rmse = get_models_validation_cyclical_rmse(model_dict, cyclical_loader, lai_precomputed=True)
    pd.DataFrame(data={"model":model_dict.keys(),
                       "cyclical_rmse":cyclical_rmse}).to_csv(os.path.join(res_dir, "common_cyclical_rmse.csv"))

    cyclical_loader = valid_loader
    losses = get_models_validation_rec_loss(model_dict, valid_loader)
    pd.DataFrame(data={"model":model_dict.keys(),
                       "loss":losses}).to_csv(os.path.join(res_dir, "loss.csv"))
    cyclical_rmse = get_models_validation_cyclical_rmse(model_dict, cyclical_loader, lai_precomputed=False)
    pd.DataFrame(data={"model":model_dict.keys(),
                       "cyclical_rmse":cyclical_rmse}).to_csv(os.path.join(res_dir, "self_cyclical_rmse.csv"))

if __name__=="__main__":
    main()