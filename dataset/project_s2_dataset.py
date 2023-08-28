import os
import torch
import argparse
from utils.utils import load_dict, save_dict, load_standardize_coeffs
from dataset.loaders import  get_train_valid_test_loader_from_patches
from prosailvae.ProsailSimus import get_bands_idx, BANDS
from prosailvae.prosail_vae import (load_prosail_vae_with_hyperprior, get_prosail_vae_config, load_params)
import numpy as np 
import socket
from utils.image_utils import get_encoded_image_from_batch, crop_s2_input
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

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
    config["load_model"] = False
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
   
def project_loader_patches(model, loader, mode="lat_mode", loader_output=False, batch_size=1, max_batches=50):
    inferred_BV = []
    projected_images = []
    angles = []
    for i, batch in tqdm(enumerate(loader)):
        if socket.gethostname()=='CELL200973':
            if i==max_batches:
                break
        (rec_image, sim_image, cropped_s2_r, cropped_s2_a,
                sigma_image) = get_encoded_image_from_batch(batch, model, patch_size=32,
                                                            bands=torch.arange(10),
                                                            mode=mode, no_rec=False)
        inferred_BV.append(sim_image.unsqueeze(0))
        projected_images.append(rec_image.unsqueeze(0))
        angles.append(cropped_s2_a)
    inferred_BV = torch.cat(inferred_BV, 0)
    prosail_s2_sim_refl = torch.cat(projected_images, 0)
    angles = torch.cat(angles, 0)
    if loader_output:
        return DataLoader(TensorDataset(prosail_s2_sim_refl.detach().cpu(), 
                                        angles.detach().cpu(),
                                        inferred_BV.detach().cpu(),), 
                                        batch_size=batch_size)
    else:
        return (prosail_s2_sim_refl.detach().cpu(), 
                angles.detach().cpu(),
                inferred_BV.detach().cpu(), )

def project_loader_pixellic(model, loader, mode = 'lat_mode', loader_output=False, batch_size=1):
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
        cropped_s2_a = cropped_s2_a.squeeze()
        sim_image = sim_image.reshape(11,-1).permute(1,0)
        rec_image = rec_image.reshape(10, -1).permute(1,0)
        cropped_s2_a = cropped_s2_a.reshape(3, -1).permute(1,0)
        inferred_BV.append(sim_image)
        projected_images.append(rec_image)
        angles.append(cropped_s2_a)
    inferred_BV = torch.cat(inferred_BV, 0)
    prosail_s2_sim_refl = torch.cat(projected_images, 0)
    angles = torch.cat(angles, 0)
    prosail_sim_vars = torch.cat((inferred_BV, angles), 1)
    if loader_output:
        return DataLoader(TensorDataset(prosail_s2_sim_refl.detach().cpu(), 
                                        prosail_sim_vars.detach().cpu()), 
                                        batch_size=batch_size)
    else:
        return (prosail_s2_sim_refl.detach().cpu(), 
                prosail_sim_vars.detach().cpu())

def save_common_cyclical_dataset(model_dict, loader, output_dir=""):
    all_s2_r = []
    all_s2_a = []
    all_sim = []
    max_hw=0
    for _, model_info in model_dict.items():
        model = model_info["model"]
        max_hw=max(max_hw, model.encoder.nb_enc_cropped_hw)
    for _, model_info in model_dict.items():
        model = model_info["model"]
        delta_hw = max_hw - model.encoder.nb_enc_cropped_hw
        s2_r, s2_a, sim = project_loader_patches(model, loader, mode="lat_mode", loader_output=False, batch_size=1)
        if delta_hw > 0:
            s2_r = crop_s2_input(s2_r, delta_hw)
            s2_a = crop_s2_input(s2_a, delta_hw)
            sim = crop_s2_input(sim, delta_hw)
        all_s2_r.append(s2_r)
        all_s2_a.append(s2_a)
        all_sim.append(sim)
    all_s2_r = torch.cat(all_s2_r, 0)
    all_s2_a = torch.cat(all_s2_a, 0)
    all_sim = torch.cat(all_sim, 0)
    torch.save(torch.cat((all_s2_r, all_s2_a, all_sim), 1), os.path.join(output_dir, "common_projected_data_set.pth"))

def load_cyclical_data_set(dir, batch_size=1):
    data = torch.load(os.path.join(dir, "common_projected_data_set.pth"))
    return DataLoader(TensorDataset(data[:,:10,...], 
                                    data[:,10:13,...],
                                    data[:,13:,...]), 
                                    batch_size=batch_size)


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
    pixellic=False
    for loader, name in zip([train_loader, test_loader], ["train_", "test_"]):
        print(name)
        if pixellic:
            prosail_s2_sim_refl, angles, prosail_sim_vars = project_loader_pixellic(model_dict["model"], loader, 
                                                                            mode='lat_mode')
        else:
            prosail_s2_sim_refl, prosail_sim_vars = project_loader_patches(model_dict["model"], loader, 
                                                                           mode='lat_mode')

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