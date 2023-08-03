import os
import torch
import argparse
from utils.utils import load_dict, save_dict, load_standardize_coeffs
from dataset.loaders import  get_train_valid_test_loader_from_patches
from prosailvae.ProsailSimus import get_bands_idx, BANDS
from prosailvae.prosail_vae import (load_prosail_vae_with_hyperprior, get_prosail_vae_config, load_params)
import numpy as np 
import socket
from utils.image_utils import get_encoded_image_from_batch
from tqdm import tqdm

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
    parser.add_argument("-o", dest="output_dir",
                        help="path to ouptput directory",
                        type=str, default="")
    return parser

def get_model_and_dataloader(parser):
    """
    Get test data (patches) in a loader and loads all trained models
    """
    train_loader, valid_loader, test_loader = get_train_valid_test_loader_from_patches(parser.data_dir,
                                                                            bands = torch.arange(10),
                                                                            batch_size=1, num_workers=0)
    model_dict = load_dict(parser.model_dict_path)
    assert model_dict["type"] == "simvae"
    config = load_params(model_dict["dir_path"], "config.json")
    bands, prosail_bands = get_bands_idx(config["weiss_bands"])
    params_path = os.path.join(model_dict["dir_path"], "prosailvae_weights.tar")
    config["load_model"] = True
    model_dict["supervised"] = config["supervised"]
    config["vae_load_file_path"] = params_path
    io_coeffs = load_standardize_coeffs(model_dict["dir_path"])
    pv_config = get_prosail_vae_config(config, bands=bands, prosail_bands=prosail_bands,
                                        inference_mode = False, rsr_dir=parser.rsr_dir,
                                        io_coeffs=io_coeffs)
    model = load_prosail_vae_with_hyperprior(pv_config=pv_config, pv_config_hyper=None,
                                                logger_name="No logger")
    model_dict["model"] = model
    info_test_data = np.load(os.path.join(parser.data_dir, "test_info.npy"))
    return model_dict, train_loader, test_loader, valid_loader, info_test_data
   

def project_loader_dataset(model, loader, mode = 'lat_mode', pixellic=True):
    inferred_BV = []
    projected_images = []
    angles = []
    for i, batch in tqdm(enumerate(loader)):
        if socket.gethostname()=='CELL200973':
            if i==50:
                break
        (rec_image, sim_image, cropped_s2_r, cropped_s2_a,
                sigma_image) = get_encoded_image_from_batch(batch, model, patch_size=32,
                                                            bands=torch.arange(10),
                                                            mode=mode, no_rec=False)
        cropped_s2_ = cropped_s2_a.squeeze()
        if pixellic:
            sim_image = sim_image.reshape(11,-1).permute(1,0)
            rec_image = rec_image.reshape(10, -1).permute(1,0)
            cropped_s2_a = cropped_s2_a.reshape(3, -1).permute(1,0)
        else:
            raise NotImplementedError
        # else:
        #     swapped_s2_a = torch.zeros_like(cropped_s2_a)
        #     swapped_s2_a[0,...] = cropped_s2_a[0,...]
        #     swapped_s2_a[1,...] = cropped_s2_a[2,...]
        #     swapped_s2_a[2,...] = cropped_s2_a[1,...]
        #     swapped_s2_a = torch.cat((swapped_s2_a, torch.zeros_like(cropped_s2_a[2,...].unsqueeze(0))),0)
        inferred_BV.append(sim_image)
        projected_images.append(rec_image)
        angles.append(cropped_s2_a)
    inferred_BV = torch.cat(inferred_BV, 0)
    prosail_s2_sim_refl = torch.cat(projected_images, 0)
    angles = torch.cat(angles, 0)
    prosail_sim_vars = torch.cat((inferred_BV, angles), 1)
    
    return prosail_s2_sim_refl.detach().cpu(), prosail_sim_vars.detach().cpu()

def main():
    if socket.gethostname()=='CELL200973':
        args = ["-m","/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/config/model_data_generate_dict.json",
                "-d", "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/patches/",
                "-o", "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/projected_data/"]
        parser = get_parser().parse_args(args)
    else:
        parser = get_parser().parse_args()
    model_dict, train_loader, test_loader, valid_loader, info_test_data = get_model_and_dataloader(parser)
    io_coeffs = load_standardize_coeffs(model_dict["dir_path"])
    if not os.path.isdir(parser.output_dir):
        os.makedirs(parser.output_dir)
    for loader, name in zip([train_loader, test_loader], ["train_", "test_"]):
        print(name)
        prosail_s2_sim_refl, prosail_sim_vars = project_loader_dataset(model_dict["model"], loader, 
                                                                    mode='lat_mode', pixellic=True)
        torch.save(prosail_sim_vars, os.path.join(parser.output_dir, name + "prosail_sim_vars.pt"))
        torch.save(prosail_s2_sim_refl, os.path.join(parser.output_dir, name + "prosail_s2_sim_refl.pt"))
        torch.save(io_coeffs.bands.loc, os.path.join(parser.output_dir, f"{name}norm_mean.pt"))
        torch.save(io_coeffs.bands.scale, os.path.join(parser.output_dir, f"{name}norm_std.pt"))
        torch.save(io_coeffs.angles.loc, os.path.join(parser.output_dir, f"{name}angles_loc.pt"))
        torch.save(io_coeffs.angles.scale, os.path.join(parser.output_dir, f"{name}angles_scale.pt"))
        torch.save(io_coeffs.idx.loc, os.path.join(parser.output_dir, f"{name}idx_loc.pt"))
        torch.save(io_coeffs.idx.scale, os.path.join(parser.output_dir, f"{name}idx_scale.pt"))
    pass

if __name__=="__main__":
    main()