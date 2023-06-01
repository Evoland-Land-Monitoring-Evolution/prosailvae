import os
import torch
import numpy as np
import datetime
import argparse
import socket
from utils.utils import load_dict, save_dict
from utils.image_utils import get_encoded_image_from_batch, crop_s2_input
from prosailvae.prosail_vae import (load_prosail_vae_with_hyperprior, get_prosail_vae_config)
from dataset.loaders import  get_train_valid_test_loader_from_patches
from prosail_plots import plot_patches, patch_validation_reg_scatter_plot
from prosailvae.ProsailSimus import get_bands_idx
from dataset.weiss_utils import get_weiss_biophyiscal_from_batch
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import prosailvae
from snap_regression.snap_nn import SnapNN
from dataset.prepare_silvia_validation import load_validation_data
from tqdm import trange, tqdm
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
                                                norm_mean = norm_mean, norm_std=norm_std)
            model = load_prosail_vae_with_hyperprior(pv_config=pv_config, pv_config_hyper=None,
                                                     logger_name="No logger")
            model_info["model"] = model
    info_test_data = np.load(os.path.join(parser.data_dir,"test_info.npy"))
    return model_dict, test_loader, info_test_data

def get_model_validation_results(model_dict: dict, 
                                 data_dir, filename, sensor):

    rec_mode = 'lat_mode' #if not socket.gethostname()=='CELL200973' else "random"
    idx_dict = {}
    for variable in ['lai', 'lai_eff', 'ccc', 'ccc_eff']:
        gdf, _, _ = load_validation_data(data_dir, filename, variable=variable)
        gdf = gdf.iloc[:51]
        idx_dict[variable] = {"x_idx" : torch.from_numpy(gdf["x_idx"].values).int(),
                              "y_idx" : torch.from_numpy(gdf["y_idx"].values).int()}
    _, s2_r, s2_a = load_validation_data(data_dir, filename, variable="lai")
    s2_r = torch.from_numpy(s2_r).float().unsqueeze(0)
    s2_a = torch.from_numpy(s2_a).float().unsqueeze(0)
    model_results = {}
    largest_hw = 0
    model_inference_info = {}
    for _, (model_name, model_info) in enumerate(tqdm(model_dict.items())):
        hw = 0
        model = model_info["model"]
        if model.spatial_mode:
            hw = model.encoder.nb_enc_cropped_hw
        if hw > largest_hw:
            largest_hw = hw    
        with torch.no_grad():
            (_, sim_image, cropped_s2_r, cropped_s2_a,
             _) = get_encoded_image_from_batch((s2_r, s2_a), model,
                                                patch_size=32, bands=torch.arange(10),
                                                mode=rec_mode, padding=True, no_rec=True)
        model_inference_info[model_name] = {"s2_r":cropped_s2_r,
                                            "s2_a":cropped_s2_a,
                                            "hw": hw}
    
        lai_pred = sim_image[6, idx_dict['lai']['y_idx'], idx_dict['lai']['x_idx']]
        lai_eff_pred = sim_image[6, idx_dict['lai_eff']['y_idx'], idx_dict['lai_eff']['x_idx']]
        ccc_pred = (sim_image[1, idx_dict['ccc']['y_idx'], idx_dict['ccc']['x_idx']] 
                    * sim_image[6, idx_dict['ccc']['y_idx'], idx_dict['ccc']['x_idx']])
        ccc_eff_pred = (sim_image[1, idx_dict['ccc_eff']['y_idx'], idx_dict['ccc_eff']['x_idx']] 
                        * sim_image[6, idx_dict['ccc_eff']['y_idx'], idx_dict['ccc_eff']['x_idx']])
        model_results[model_name] = {'lai': lai_pred,
                                     'lai_eff': lai_eff_pred,
                                     'ccc': ccc_pred,
                                     'ccc_eff': ccc_eff_pred}


    for model_name, _ in model_dict.items():
        delta_hw = largest_hw - model_inference_info[model_name]['hw']
        s2_r = model_inference_info[model_name]["s2_r"]
        s2_a = model_inference_info[model_name]["s2_a"]
        # lai = model_results[model_name]["lai"]
        # ccc = model_results[model_name]["ccc"]
        if delta_hw > 0 :
            # model_results[model_name]["lai"] = crop_s2_input(lai, delta_hw)
            # model_results[model_name]["ccc"] = crop_s2_input(ccc, delta_hw)
            model_inference_info[model_name]["s2_r"] = crop_s2_input(s2_r, delta_hw)
            model_inference_info[model_name]["s2_a"] = crop_s2_input(s2_a, delta_hw)
    
    (snap_lai, snap_cab,
        _) = get_weiss_biophyiscal_from_batch((model_inference_info[model_name]["s2_r"], 
                                               model_inference_info[model_name]["s2_a"]),
                                               patch_size=32, sensor=sensor)
    model_results["SNAP"] = {'lai': snap_lai[..., idx_dict['lai']['y_idx'], idx_dict['lai']['x_idx']],
                             'lai_eff': snap_lai[..., idx_dict['lai_eff']['y_idx'], idx_dict['lai_eff']['x_idx']],
                             'ccc': snap_cab[..., idx_dict['ccc']['y_idx'], idx_dict['ccc']['x_idx']],
                             'ccc_eff': snap_cab[..., idx_dict['ccc_eff']['y_idx'], idx_dict['ccc_eff']['x_idx']]}
    return model_results



def get_model_results(model_dict: dict, test_loader, info_test_data):
    """
    Compute results for all models
    """
    rec_mode = 'lat_mode' if not socket.gethostname()=='CELL200973' else "random"
    for model_name, model_info in model_dict.items():
        model_info["reconstruction"] = []
        model_info["prosail_vars"] = []
        model_info["latent_sigma"] = []
    all_s2_r = []
    all_snap_lai = []
    all_snap_cab = []
    all_snap_cw = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader)):
            current_patch_results = {}
            largest_hw = 0
            for model_name, model_info in model_dict.items():
                hw = 0
                model = model_info["model"]
                if model.spatial_mode:
                    hw = model.encoder.nb_enc_cropped_hw
                if hw > largest_hw:
                    largest_hw = hw                
                (rec_image, sim_image, cropped_s2_r, cropped_s2_a,
                sigma_image) = get_encoded_image_from_batch(batch, model, patch_size=32,
                                                            bands=torch.arange(10),
                                                            mode=rec_mode)
                current_patch_results[model_name] = {"reconstruction": rec_image,
                                                       "prosail_vars": sim_image, 
                                                       "latent_sigma": sigma_image,
                                                       "cropped_s2_r" : cropped_s2_r,
                                                       "cropped_s2_a": cropped_s2_a,
                                                       "hw":hw}
            for model_name, model_info in model_dict.items():
                delta_hw = largest_hw - current_patch_results[model_name]['hw']
                rec_image = current_patch_results[model_name]["reconstruction"]
                sim_image = current_patch_results[model_name]["prosail_vars"]
                sigma_image = current_patch_results[model_name]["latent_sigma"]
                cropped_s2_r = current_patch_results[model_name]["cropped_s2_r"]
                cropped_s2_a = current_patch_results[model_name]["cropped_s2_a"]
                if delta_hw > 0 :
                    rec_image = crop_s2_input(rec_image, delta_hw)
                    sim_image = crop_s2_input(sim_image, delta_hw)
                    sigma_image = crop_s2_input(sigma_image, delta_hw)
                    cropped_s2_r = crop_s2_input(cropped_s2_r, delta_hw)
                    cropped_s2_a = crop_s2_input(cropped_s2_a, delta_hw)

                model_info["reconstruction"].append(rec_image)
                model_info["prosail_vars"].append(sim_image)
                model_info["latent_sigma"].append(sigma_image)
            all_s2_r.append(cropped_s2_r.squeeze())
            info = info_test_data[i,:]
            try:
                (snap_lai, snap_cab,
                snap_cw) = get_weiss_biophyiscal_from_batch((cropped_s2_r, cropped_s2_a),
                                                             patch_size=32, sensor=info[0])
            except Exception as exc:
                print(exc)
                print(i)
                print(batch)
                print(cropped_s2_r.size())
                print(cropped_s2_a.size())
                ValueError
            all_snap_lai.append(snap_lai)
            all_snap_cab.append(snap_cab)
            all_snap_cw.append(snap_cw)
        all_snap_lai = torch.stack(all_snap_lai, axis=0)
        all_snap_cab = torch.stack(all_snap_cab, axis=0)
        all_snap_cw = torch.stack(all_snap_cw, axis=0)
        for model_name, model_info in model_dict.items():
            model_info["reconstruction"] = torch.stack(model_info["reconstruction"], 0)
            model_info["prosail_vars"] = torch.stack(model_info["prosail_vars"], 0)
            model_info["latent_sigma"] = torch.stack(model_info["latent_sigma"], 0)
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

def plot_validation_results_comparison(model_dict, model_results, data_dir, filename, res_dir=None):
    for variable in ["lai", "lai_eff", "ccc", "ccc_eff"]:
        n_models = len(model_dict) + 1
        fig, axs = plt.subplots(nrows=1, ncols=n_models, dpi=150, figsize=(6*n_models, 6))
        gdf, _, _ = load_validation_data(data_dir, filename, variable=variable)
        for i, (model_name, model_info) in enumerate(model_dict.items()):
            # sub_variable = "lai" if variable in ["lai", "lai_eff"] else "ccc"
            pred_at_site = model_results[model_name][variable].numpy()
            fig, ax, g = patch_validation_reg_scatter_plot(gdf, pred_at_site=pred_at_site,
                                                            variable=variable,
                                                            fig=fig, ax=axs[i], legend=True)
            
            axs[i].set_title(model_info["plot_name"])
        pred_at_site = model_results["SNAP"][variable].numpy()
        fig, ax, g = patch_validation_reg_scatter_plot(gdf, pred_at_site=pred_at_site,
                                                        variable=variable,
                                                        fig=fig, ax=axs[-1], legend=True)
        
        axs[-1].set_title("SNAP")
        if res_dir is not None:
            fig.savefig(os.path.join(res_dir, f"{variable}_{filename}_validation.png"))

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
        models_errs = [(all_s2_r[i,...] - model_info["reconstruction"][i,...]).abs().mean(0).unsqueeze(0)
                                 for _, model_info in model_dict.items()]
        vmin = min([err.cpu().min().item() for err in models_errs])
        vmax = max([err.cpu().max().item() for err in models_errs])
        fig, _ = plot_patches(patch_list = [all_s2_r[i,...]] + models_errs,
                              title_list = [f"Sentinel {info[0]} \n"
                                            f"{info[1][:4]}/{info[1][4:6]}/{info[1][6:]} - {info[2]}"
                                            ] + [model_info["plot_name"] for _, model_info in model_dict.items()],
                                            vmin=vmin, vmax=vmax)
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

    err_scatter_dict = {}
    global_lim = [10, 0]
    for _, model_info in model_dict.items():
        err_scatter_dict[model_info["plot_name"]] = (model_info["reconstruction"][:,:10,...]
                                                     - all_s2_r[:,:10,...]).pow(2).mean(1).sqrt().reshape(-1)
        global_lim[0] = min(global_lim[0], err_scatter_dict[model_info["plot_name"]].min().item())
        global_lim[1] = max(global_lim[1], err_scatter_dict[model_info["plot_name"]].max().item())
    fig, _ = regression_pair_plot(err_scatter_dict, global_lim)
    if res_dir is not None:
        fig.savefig(os.path.join(res_dir, "rec_error_comparison.png"))

    fig, axs = plt.subplots(1, len(err_scatter_dict), figsize=(3*len(err_scatter_dict), 3), dpi=200)
    for i, (_, model_info) in enumerate(model_dict.items()):
        axs[i].hist(err_scatter_dict[model_info["plot_name"]], bins=200, range=global_lim)
        axs[i].plot(global_lim, global_lim, 'k--')
        axs[i].set_title(model_info["plot_name"])
    if res_dir is not None:
        fig.savefig(os.path.join(res_dir, "model_err_hist.png"))

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

def interpolate_validation_pred(model_dict, silvia_data_dir, filename, sensor):
    d0 = datetime.date.fromisoformat('2018-05-15')
    d1 = datetime.date.fromisoformat('2018-06-13')
    dt_image = (d1 - d0).days
    gdf, _, _ = load_validation_data(silvia_data_dir, filename[0], variable="lai")
    gdf = gdf.iloc[:51]
    t_sample = gdf["date"].apply(lambda x: (x.date()-d0).days).values
    validation_results_1 = get_model_validation_results(model_dict, silvia_data_dir, filename[0], sensor[0])
    validation_results_2 = get_model_validation_results(model_dict, silvia_data_dir, filename[1], sensor[0])
    validation_results = {}
    for model_name, _ in validation_results_1.items():
        model_results = {}
        for variable in ["lai", "lai_eff", "ccc", "ccc_eff"]:
            gdf, _, _ = load_validation_data(silvia_data_dir, filename[0], variable=variable)
            gdf = gdf.iloc[:51]
            t_sample = torch.from_numpy(gdf["date"].apply(lambda x: (x.date()-d0).days).values)
            m = (validation_results_1[model_name][variable].squeeze() 
                 - validation_results_2[model_name][variable].squeeze()) / dt_image
            b = validation_results_2[model_name][variable].squeeze() - m * d1.day
            try:
                model_results[variable] = (m * t_sample + b).reshape(-1)
            except Exception as exc:
                # print(exc)
                print(model_name, variable, m.size(), t_sample.size(), b.size())
                print(validation_results_1[model_name][variable].size(), validation_results_2[model_name][variable].size(), dt_image)
        validation_results[model_name] = model_results
    return validation_results

def main():
    """
    main.
    """
    if socket.gethostname()=='CELL200973':
        args = ["-m","/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/config/model_dict_dev.json",
                "-d", "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/patches/",
                "-r", "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/results/comparaison/"]
        parser = get_parser().parse_args(args)
        silvia_data_dir = "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/silvia_validation"
    else:
        parser = get_parser().parse_args()
        silvia_data_dir = "/work/scratch/zerahy/prosailvae/data/silvia_validation"
    res_dir = parser.res_dir
    if not os.path.isdir(res_dir):
        os.makedirs(res_dir)
    model_dict, test_loader, info_test_data = get_model_and_dataloader(parser)
   
    filename = ["2B_20180516_FRM_Veg_Barrax_20180605", "2A_20180613_FRM_Veg_Barrax_20180605"]
    sensor = ["2B", "2A"]
    if isinstance(filename, list):
        validation_results = interpolate_validation_pred(model_dict, silvia_data_dir, filename, sensor)
    else:
        validation_results = get_model_validation_results(model_dict, silvia_data_dir, filename, sensor)
    plot_validation_results_comparison(model_dict, validation_results, silvia_data_dir, filename[0], res_dir=res_dir)
    (model_dict, all_s2_r, all_snap_lai, all_snap_cab,
     all_snap_cw) = get_model_results(model_dict, test_loader, info_test_data)
    plot_comparative_results(model_dict, all_s2_r, all_snap_lai, all_snap_cab, all_snap_cw, info_test_data, res_dir)
    compare_snap_versions_on_real_data(test_loader, res_dir)
    compare_snap_versions_on_weiss_data(res_dir)
    
if __name__=="__main__":
    main()