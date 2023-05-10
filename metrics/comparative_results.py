import torch
import numpy as np
import os
import argparse
import socket
from utils.utils import load_dict, save_dict
from utils.image_utils import get_encoded_image_from_batch
from prosailvae.prosail_vae import (load_prosail_vae_with_hyperprior, get_prosail_vae_config, ProsailVAEConfig)
from dataset.loaders import  get_train_valid_test_loader_from_patches
from prosail_plots import plot_patches
from prosailvae.ProsailSimus import get_bands_idx
from results import get_weiss_biophyiscal_from_batch
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import prosailvae
from snap_regression.snap_nn import SnapNN


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
    return parser

def get_model_and_dataloader(parser):
    """
    Get test data (patches) in a loader and loads all trained models
    """
    _, _, test_loader = get_train_valid_test_loader_from_patches(parser.data_dir,
                                                                 bands = torch.arange(10),
                                                                 batch_size=1, num_workers=0)
    model_dict = load_dict(parser.model_dict_path)
    for model_name, model_info in model_dict.items():
        if model_info["type"] == "simvae":
            config = load_dict(os.path.join(model_info["dir_path"], "config.json"))
            bands, prosail_bands = get_bands_idx(config["weiss_bands"])
            norm_mean = torch.load(os.path.join(model_info["dir_path"], "norm_mean.pt"))
            norm_std = torch.load(os.path.join(model_info["dir_path"], "norm_std.pt"))
            params_path = os.path.join(model_info["dir_path"], "prosailvae_weights.tar")
            config["load_model"] = True
            config["vae_load_file_path"] = params_path
            pv_config = get_prosail_vae_config(config, bands=bands, prosail_bands=prosail_bands,
                                                inference_mode = False, rsr_dir=parser.rsr_dir,
                                                norm_mean = norm_mean, norm_std=norm_std,
                                                spatial_mode=True)
            model = load_prosail_vae_with_hyperprior(pv_config=pv_config, pv_config_hyper=None,
                                                     logger_name="No logger")
            model_info["model"] = model
    info_test_data = np.load(os.path.join(parser.data_dir,"test_info.npy"))
    return model_dict, test_loader, info_test_data

def get_model_results(model_dict: dict, test_loader, info_test_data):
    """
    Compute results for all models
    """
    rec_mode = 'lat_mode' if not socket.gethostname()=='CELL200973' else "random"
    with torch.no_grad():
        for model_name, model_info in model_dict.items():
            print(f"Computing results for {model_name}.")
            model = model_info["model"]
            all_rec = []
            all_vars = []
            all_sigma = []
            for _, batch in enumerate(tqdm(test_loader)):
                (rec_image, sim_image, _, _,
                sigma_image) = get_encoded_image_from_batch(batch, model, patch_size=32,
                                                            bands=torch.arange(10),
                                                            mode=rec_mode)
                all_rec.append(rec_image)
                all_vars.append(sim_image)
                all_sigma.append(sigma_image)
            model_info["reconstruction"] = torch.stack(all_rec, axis=0)
            model_info["prosail_vars"] = torch.stack(all_vars, axis=0)
            model_info["latent_sigma"] = torch.stack(all_sigma, axis=0)
        all_snap_lai = []
        all_snap_cab = []
        all_snap_cw = []
        all_s2_r = []
        print(f"Computing results for SNAP.")
        for i, batch in enumerate(tqdm(test_loader)):
            (_, _, cropped_s2_r, cropped_s2_a,
            _) = get_encoded_image_from_batch(batch, model, patch_size=32,
                                              bands=torch.arange(10), mode=rec_mode)
            info = info_test_data[i,:]
            (snap_lai, snap_cab,
            snap_cw) = get_weiss_biophyiscal_from_batch((cropped_s2_r.unsqueeze(0),
                                                          cropped_s2_a.unsqueeze(0)), 
                                                          patch_size=32, sensor=info[0])
            all_snap_lai.append(snap_lai)
            all_snap_cab.append(snap_cab)
            all_snap_cw.append(snap_cw)
            all_s2_r.append(cropped_s2_r)
        all_snap_lai = torch.stack(all_snap_lai, axis=0)
        all_snap_cab = torch.stack(all_snap_cab, axis=0)
        all_snap_cw = torch.stack(all_snap_cw, axis=0)
        all_s2_r = torch.stack(all_s2_r, axis=0)
    return model_dict, all_s2_r, all_snap_lai, all_snap_cab, all_snap_cw

def regression_pair_plot(scatter_dict, global_lim):
    """
    plots pair wise regression.
    """
    g = sns.pairplot(pd.DataFrame(data=scatter_dict), kind="reg", diag_kind="kde",
                         plot_kws=dict(line_kws={'color':'red', 'linestyle':'--'},
                                       scatter_kws={"edgecolor":"none","s":1}))
    g.map_lower(sns.kdeplot, levels=3, color="red")
    g.map_upper(sns.kdeplot, levels=3, color="red")
    axs = g.axes
    for j in range(axs.shape[0]):
        for k in range(axs.shape[1]):
            axs[j,k].set_xlim(global_lim)
            if j!=k:
                axs[j,k].set_ylim(global_lim)
                ax_min = min(global_lim[0], global_lim[0])
                ax_max = max(global_lim[1], global_lim[1])
                axs[j,k].plot([ax_min, ax_max], [ax_min, ax_max], 'k')
    return g.fig, g.axes 


def plot_comparative_results(model_dict, all_s2_r, all_snap_lai, all_snap_cab,
                             all_snap_cw, info_test_data, res_dir=None):

    for i in range(all_s2_r.size(0)):
        info = info_test_data[i]
        fig, _ = plot_patches(patch_list = [all_s2_r[i,...]] 
                              + [model_info["reconstruction"][i,...] for _, model_info in model_dict.items()],
                              title_list = [f"Sentinel {info[0]} \n"
                                            f"{info[1][:4]}/{info[1][4:6]}/{info[1][6:]} - {info[2]}"] 
                                            + [model_info["plot_name"] for _, model_info in model_dict.items()])
        if res_dir is not None:
            fig.savefig(os.path.join(res_dir, f"{i}_{info[1]}_{info[2]}_patch_reconstructions_rgb.png"))

        fig, _ = plot_patches(patch_list = [all_s2_r[i,torch.tensor([8,3,6]),...]] 
                              + [model_info["reconstruction"][i,torch.tensor([8,3,6]),...] 
                                 for _, model_info in model_dict.items()],
                              title_list = [f"Sentinel {info[0]} \n"
                                            f"{info[1][:4]}/{info[1][4:6]}/{info[1][6:]} - {info[2]}"] 
                                            + [model_info["plot_name"] for _, model_info in model_dict.items()])
        if res_dir is not None:
            fig.savefig(os.path.join(res_dir, f"{i}_{info[1]}_{info[2]}_patch_rec_B8B5B11.png"))

        fig, _ = plot_patches(patch_list = [all_s2_r[i,...]
                                            ] + [(all_s2_r[i,...] - model_info["reconstruction"][i,...]).abs().mean(0).unsqueeze(0)
                                 for _, model_info in model_dict.items()],
                              title_list = [f"Sentinel {info[0]} \n"
                                            f"{info[1][:4]}/{info[1][4:6]}/{info[1][6:]} - {info[2]}"
                                            ] + [model_info["plot_name"] for _, model_info in model_dict.items()])
        if res_dir is not None:
            fig.savefig(os.path.join(res_dir, f"{i}_{info[1]}_{info[2]}_patch_err.png"))
        lai_patch_tensors = [model_info["prosail_vars"][i,6,...].unsqueeze(0) 
                             for _, model_info in model_dict.items()] + [all_snap_lai[i,...]]
        vmin = min([lai_tensor.min() for lai_tensor in lai_patch_tensors])
        vmax = max([lai_tensor.max() for lai_tensor in lai_patch_tensors])
        fig, _ = plot_patches(patch_list = [all_s2_r[i,...]] 
                              + [model_info["prosail_vars"][i,6,...].unsqueeze(0) for _, model_info in model_dict.items()]
                              + [all_snap_lai[i,...]],
                              title_list = [f"Sentinel {info[0]} \n"
                                            f"{info[1][:4]}/{info[1][4:6]}/{info[1][6:]} - {info[2]}"] 
                                            + [model_info["plot_name"] for _, model_info in model_dict.items()]
                                            + ["SNAP"], vmin=vmin, vmax=vmax)
        if res_dir is not None:
            fig.savefig(os.path.join(res_dir, f"{i}_{info[1]}_{info[2]}_LAI.png"))

    lai_scatter_dict = {}
    lai_scatter_dict["SNAP's Biophysical Processor"] = all_snap_lai.squeeze(1).reshape(-1)
    global_lim = [lai_scatter_dict["SNAP's Biophysical Processor"].min().item(),
                    lai_scatter_dict["SNAP's Biophysical Processor"].max().item()]
    for _, model_info in model_dict.items():
            lai_scatter_dict[model_info["plot_name"]] = model_info["prosail_vars"][:,6,...].reshape(-1)
            global_lim[0] = min(global_lim[0], lai_scatter_dict[model_info["plot_name"]].min().item())
            global_lim[1] = max(global_lim[1], lai_scatter_dict[model_info["plot_name"]].max().item())
    fig, _ = regression_pair_plot(lai_scatter_dict, global_lim)
    if res_dir is not None:
        fig.savefig(os.path.join(res_dir, "model_lai_comparison.png"))

def compare_snap_versions_on_real_data(test_loader, res_dir):
    snap_ver = ["2.1", "3A", "3B"]
    snap_lai_dict = {}
    global_lai_lim = [torch.inf, -torch.inf]
    for ver in snap_ver:
        sensor_lai = []
        sensor_cab = []
        sensor_cw = []
        for _, batch in enumerate(tqdm(test_loader)):
            (snap_lai, snap_cab,
            snap_cw) = get_weiss_biophyiscal_from_batch(batch, patch_size=32,
                                                        ver=ver)
            sensor_lai.append(snap_lai.reshape(-1))
            sensor_cab.append(snap_cab.reshape(-1))
            sensor_cw.append(snap_cw.reshape(-1))
        snap_lai_dict[ver] = torch.cat(sensor_lai, 0)
        global_lai_lim[0] = min(global_lai_lim[0], snap_lai_dict[ver].min().item())
        global_lai_lim[1] = max(global_lai_lim[1], snap_lai_dict[ver].max().item())
    fig, _ = regression_pair_plot(snap_lai_dict, global_lai_lim)
    fig.savefig(os.path.join(res_dir, "scatter_lai_snap_versions_s2.png"))
    pass


def compare_snap_versions_on_weiss_data(res_dir):
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
    s2_r, s2_a, lai = load_weiss_dataset(os.path.join(prosailvae.__path__[0], os.pardir) + "/field_data/lai/")
    s2_data = torch.from_numpy(np.concatenate((s2_r, np.cos(np.deg2rad(s2_a))), 1)).float()
    snap_ver = ["2.1", "3A", "3B"]
    snap_lai_dict = {"ref":lai.reshape(-1)}
    global_lai_lim = [lai.min().item(), lai.max().item()]
    for ver in snap_ver:
        sensor_lai = []
        with torch.no_grad():
            lai_snap = SnapNN(variable='lai', ver=ver)
            lai_snap.set_weiss_weights()
            snap_lai = lai_snap.forward(s2_data)
        sensor_lai.append(snap_lai.reshape(-1))
        snap_lai_dict[ver] = torch.cat(sensor_lai, 0)
        global_lai_lim[0] = min(global_lai_lim[0], snap_lai_dict[ver].min().item())
        global_lai_lim[1] = max(global_lai_lim[1], snap_lai_dict[ver].max().item())
    
    fig, _ = regression_pair_plot(snap_lai_dict, global_lai_lim)
    fig.savefig(os.path.join(res_dir, "scatter_lai_snap_versions_weiss.png"))

def main():
    """
    main.
    """
    if socket.gethostname()=='CELL200973':
        args = ["-m","/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/config/model_dict_dev.json",
                "-d", "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/patches/",
                "-r", "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/results/comparaison/"]
        parser = get_parser().parse_args(args)
    else:
        parser = get_parser().parse_args()
    res_dir = parser.res_dir
    if not os.path.isdir(res_dir):
        os.makedirs(res_dir)
    model_dict, test_loader, info_test_data = get_model_and_dataloader(parser)

    (model_dict, all_s2_r, all_snap_lai, all_snap_cab,
     all_snap_cw) = get_model_results(model_dict, test_loader, info_test_data)
    plot_comparative_results(model_dict, all_s2_r, all_snap_lai, all_snap_cab, all_snap_cw, info_test_data, res_dir)
    compare_snap_versions_on_real_data(test_loader, res_dir)
    compare_snap_versions_on_weiss_data(res_dir)
    
if __name__=="__main__":
    main()