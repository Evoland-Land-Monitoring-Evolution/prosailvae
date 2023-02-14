import os
import prosailvae
from prosailvae.ProsailSimus import PROSAILVARS, BANDS
from prosailvae.utils import load_dict, save_dict
from prosailvae.prosail_vae import load_PROSAIL_VAE_with_supervised_kl
from dataset.preprocess_small_validation_file import get_small_validation_data
import torch
import logging
LOGGER_NAME = "PROSAIL-VAE validation"

def get_model(model_dir):
    logging.basicConfig(filename=model_dir+'/pv_validation.log', 
                              level=logging.INFO, force=True)
    logger_name = LOGGER_NAME
    # create logger
    logger = logging.getLogger(logger_name)
    params = load_dict(model_dir + "/config.json")
    if params["supervised"]:
        params["simulated_dataset"]=True
    params_sup_kl_model = None
    if params["supervised_kl"]:
        params_sup_kl_model = load_dict(model_dir+"/sup_kl_model_config.json")
        params_sup_kl_model['sup_model_weights_path'] = model_dir+"/sup_kl_model_weights.tar"
    vae_file_path = model_dir + '/prosailvae_weights.tar'
    rsr_dir = '/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/'
    PROSAIL_VAE = load_PROSAIL_VAE_with_supervised_kl(params, rsr_dir, model_dir, 
                                logger_name=LOGGER_NAME, vae_file_path=vae_file_path, params_sup_kl_model=params_sup_kl_model)
    return PROSAIL_VAE

def main():
    model_dir = "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/results/best_regression/"
    PROSAIL_VAE = get_model(model_dir)
    s2_r, s2_a, lais = get_small_validation_data()
    dist_params, z, sim, rec = PROSAIL_VAE.point_estimate_rec(s2_r, s2_a, mode='sim_mode')
    pass

if __name__ == "__main__":
    main()