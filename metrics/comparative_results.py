import os
import torch
import numpy as np
import datetime
import argparse
import socket
from utils.utils import load_dict, save_dict, load_standardize_coeffs
from utils.image_utils import get_encoded_image_from_batch, crop_s2_input
from metrics.metrics_utils import regression_metrics
from prosailvae.prosail_vae import (load_prosail_vae_with_hyperprior, get_prosail_vae_config, load_params)
from dataset.loaders import  get_train_valid_test_loader_from_patches
from prosail_plots import plot_patches, patch_validation_reg_scatter_plot, plot_belsar_metrics, regression_plot
from prosailvae.ProsailSimus import get_bands_idx, BANDS
from dataset.weiss_utils import get_weiss_biophyiscal_from_batch
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import prosailvae
from snap_regression.snap_nn import SnapNN
from validation.frm4veg_validation import (load_frm4veg_data, interpolate_frm4veg_pred, BARRAX_FILENAMES, WYTHAM_FILENAMES, 
                                           BARRAX_2021_FILENAME, get_frm4veg_results_at_date)
from tqdm import trange, tqdm
from validation.belsar_validation import interpolate_belsar_metrics, save_belsar_predictions, save_snap_belsar_predictions
from validation.validation import get_belsar_x_frm4veg_lai_results, get_validation_global_metrics, get_frm4veg_ccc_results

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
    _, valid_loader, test_loader = get_train_valid_test_loader_from_patches(parser.data_dir,
                                                                 bands = torch.arange(10),
                                                                 batch_size=1, num_workers=0)
    model_dict = load_dict(parser.model_dict_path)
    for model_name, model_info in model_dict.items():
        if model_info["type"] == "simvae":
            config = load_params(model_info["dir_path"], "config.json")
            bands, prosail_bands = get_bands_idx(config["weiss_bands"])
            # norm_mean = torch.load(os.path.join(model_info["dir_path"], "norm_mean.pt"))
            # norm_std = torch.load(os.path.join(model_info["dir_path"], "norm_std.pt"))
            params_path = os.path.join(model_info["dir_path"], "prosailvae_weights.tar")
            config["load_model"] = True
            model_info["supervised"] = config["supervised"]
            config["vae_load_file_path"] = params_path
            # if "disabled_latent" not in config.keys():
            #     config["disabled_latent"] = []
            # if "disabled_latent_values" not in config.keys():
            #     config["disabled_latent_values"] = []
            # if "R_down" not in config.keys():
            #     config["R_down"] = 1
            io_coeffs = load_standardize_coeffs(model_info["dir_path"])
            pv_config = get_prosail_vae_config(config, bands=bands, prosail_bands=prosail_bands,
                                                inference_mode = False, rsr_dir=parser.rsr_dir,
                                                io_coeffs=io_coeffs)
            model = load_prosail_vae_with_hyperprior(pv_config=pv_config, pv_config_hyper=None,
                                                     logger_name="No logger")
            model_info["model"] = model
    info_test_data = np.load(os.path.join(parser.data_dir,"test_info.npy"))
    return model_dict, test_loader, valid_loader, info_test_data

def get_model_results(model_dict: dict, test_loader, info_test_data, max_patch = 50, mode = 'lat_mode'):
    """
    Compute results for all models
    """
     #if not socket.gethostname()=='CELL200973' else "random"
    for model_name, model_info in model_dict.items():
        model_info["reconstruction"] = []
        model_info["prosail_vars"] = []
        model_info["latent_sigma"] = []
        model_info["cyclical_ref_lai"] = []
        model_info["cyclical_lai"] = []
    all_s2_r = []
    all_snap_lai = []
    all_snap_cab = []
    all_snap_cw = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader)):
            if i>=max_patch:
                break
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
                                                            mode=mode, no_rec=False)
                
                (_, cyclical_sim_image, _, _, _) = get_encoded_image_from_batch((rec_image.unsqueeze(0), cropped_s2_a), 
                                                                                model, patch_size=32,
                                                                                    bands=torch.arange(10),
                                                                                    mode=mode, no_rec=True)
                current_patch_results[model_name] = {"reconstruction": rec_image,
                                                       "prosail_vars": sim_image, 
                                                       "latent_sigma": sigma_image,
                                                       "cropped_s2_r" : cropped_s2_r,
                                                       "cropped_s2_a": cropped_s2_a,
                                                       "cyclical_ref_lai":crop_s2_input(sim_image, hw)[6,...].reshape(-1),
                                                       "cyclical_lai":cyclical_sim_image[6,...].reshape(-1),
                                                       "hw":hw}
            for model_name, model_info in model_dict.items():
                delta_hw = largest_hw - current_patch_results[model_name]['hw']
                rec_image = current_patch_results[model_name]["reconstruction"]
                sim_image = current_patch_results[model_name]["prosail_vars"]
                sigma_image = current_patch_results[model_name]["latent_sigma"]
                cropped_s2_r = current_patch_results[model_name]["cropped_s2_r"]
                cropped_s2_a = current_patch_results[model_name]["cropped_s2_a"]
                cyclical_ref_lai = current_patch_results[model_name]["cyclical_ref_lai"]
                cyclical_lai = current_patch_results[model_name]["cyclical_lai"]
                if delta_hw > 0 :
                    rec_image = crop_s2_input(rec_image, delta_hw)
                    sim_image = crop_s2_input(sim_image, delta_hw)
                    sigma_image = crop_s2_input(sigma_image, delta_hw)
                    cropped_s2_r = crop_s2_input(cropped_s2_r, delta_hw)
                    cropped_s2_a = crop_s2_input(cropped_s2_a, delta_hw)
                    cyclical_ref_lai = cyclical_ref_lai
                    cyclical_lai = cyclical_lai

                model_info["reconstruction"].append(rec_image)
                model_info["prosail_vars"].append(sim_image)
                model_info["latent_sigma"].append(sigma_image)
                model_info["cyclical_ref_lai"].append(cyclical_ref_lai)
                model_info["cyclical_lai"].append(cyclical_lai)
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
            model_info["cyclical_ref_lai"] = torch.cat(model_info["cyclical_ref_lai"], 0)
            model_info["cyclical_lai"] = torch.cat(model_info["cyclical_lai"], 0)
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

def plot_frm4veg_results_comparison(model_dict, model_results, data_dir, filename, res_dir=None, prefix="", margin = 0.02):
    for variable in ["lai", "lai_eff", "ccc", "ccc_eff"]:
        n_models = len(model_dict) + 1
        fig, axs = plt.subplots(nrows=1, ncols=n_models, dpi=150, figsize=(6*n_models, 6))
        gdf, _, _, xcoords, ycoords = load_frm4veg_data(data_dir, filename, variable=variable)
        gdf=gdf.iloc[:51]
        xmin = min(np.min(gdf[variable].values), np.min(model_results["SNAP"][variable].reshape(-1)))
        xmax = max(np.max(gdf[variable].values), np.max(model_results["SNAP"][variable].reshape(-1)))
        for i, (model_name, model_info) in enumerate(model_dict.items()):
            pred_at_site = model_results[model_name][variable]
            xmax = max(np.max(pred_at_site), xmax)
            xmin = min(np.min(pred_at_site), xmin)
        xmin = xmin - margin * (xmax - xmin)
        xmax = xmax + margin * (xmax - xmin)
        for i, (model_name, model_info) in enumerate(model_dict.items()):
            # sub_variable = "lai" if variable in ["lai", "lai_eff"] else "ccc"
            pred_at_site = model_results[model_name][variable]
            fig, ax, g = patch_validation_reg_scatter_plot(gdf, pred_at_site=pred_at_site,
                                                            variable=variable,
                                                            fig=fig, ax=axs[i], legend=True)
            
            axs[i].set_title(model_info["plot_name"])
        pred_at_site = model_results["SNAP"][variable].reshape(-1)
        fig, ax, g = patch_validation_reg_scatter_plot(gdf, pred_at_site=pred_at_site,
                                                        variable=variable,
                                                        fig=fig, ax=axs[-1], legend=True)
        
        axs[-1].set_title("SNAP")
        if res_dir is not None:
            fig.savefig(os.path.join(res_dir, f"{prefix}{variable}_{filename}_validation.png"))


def plot_lai_validation_comparison(model_dict, model_results, res_dir=None, prefix="", margin = 0.02, 
                                   hue="Site", legend_col=3, hue_perfs=False):
    n_models = len(model_dict) + 1
    xmin = np.min(model_results["SNAP"]["Predicted LAI"])
    xmax = np.max(model_results["SNAP"]["Predicted LAI"])
    for i, (model_name, model_info) in enumerate(model_dict.items()):
        pred_at_site = model_results[model_name]["Predicted LAI"]
        xmax = max(np.max(pred_at_site), xmax)
        xmin = min(np.min(pred_at_site), xmin)
    ref_lai = model_results[model_name]['LAI']
    xmax = max(np.max(ref_lai), xmax)
    xmin = min(np.min(ref_lai), xmin)
    xmin = xmin - margin * (xmax - xmin)
    xmax = xmax + margin * (xmax - xmin)

    fig, axs = plt.subplots(nrows=1, ncols=n_models, dpi=150, figsize=(6*n_models, 6))
    for i, (model_name, model_info) in enumerate(model_dict.items()):
        df_metrics = model_results[model_name]
        fig, ax = regression_plot(df_metrics, x="LAI", y="Predicted LAI", fig=fig, ax=axs[i], hue=hue,
                                  legend_col=legend_col, xmin=xmin, xmax=xmax, error_x="LAI std", 
                                  error_y="Predicted LAI std", hue_perfs=hue_perfs)
        ax.set_title(model_info["plot_name"] + "\n loss: {:.2f}".format(model_info['loss']))
    df_metrics = model_results["SNAP"]
    fig, _ = regression_plot(df_metrics, x="LAI", y="Predicted LAI", fig=fig, ax=axs[-1], hue=hue,
                             legend_col=legend_col, xmin=xmin, xmax=xmax, error_x="LAI std", hue_perfs=hue_perfs)
    axs[-1].set_title("SNAP")
    if res_dir is not None:
        fig.savefig(os.path.join(res_dir, f"{prefix}_validation.png"), transparent=False)

def get_belsar_validation_results(model_dict: dict, belsar_dir, res_dir, method="closest", mode=None, get_error=True):
    model_results = {}
    for _, (model_name, model_info) in enumerate(model_dict.items()):
        model_results[model_name] = interpolate_belsar_metrics(belsar_data_dir=belsar_dir, belsar_pred_dir=res_dir,
                                                               file_suffix=f"_{model_name}_{mode}", method=method, get_error=get_error)

    model_results["SNAP"] = interpolate_belsar_metrics(belsar_data_dir=belsar_dir, belsar_pred_dir=res_dir,
                                                       file_suffix="_SNAP", method=method, get_error=get_error)
    return model_results

def plot_belsar_validation_results_comparison(model_dict, model_results, res_dir=None, suffix="", margin=0.02):
    n_models = len(model_dict) + 1
    
    xmin = min(np.min(model_results["SNAP"]['parcel_lai_mean']), np.min(model_results["SNAP"]['lai_mean']))
    xmax = max(np.max(model_results["SNAP"]['parcel_lai_mean']), np.max(model_results["SNAP"]['lai_mean']))
    for i, (model_name, model_info) in enumerate(model_dict.items()):
        # sub_variable = "lai" if variable in ["lai", "lai_eff"] else "ccc"
        metrics = model_results[model_name]
        xmin = min(xmin, np.min(metrics['parcel_lai_mean'].values))
        xmax = max(xmax, np.max(metrics['parcel_lai_mean'].values))
    xmin = xmin - margin * (xmax - xmin)
    xmax = xmax + margin * (xmax - xmin)
    fig, axs = plt.subplots(nrows=1, ncols=n_models, dpi=150, figsize=(6*n_models, 6))
    for i, (model_name, model_info) in enumerate(model_dict.items()):
        # sub_variable = "lai" if variable in ["lai", "lai_eff"] else "ccc"
        metrics = model_results[model_name]
        fig, _ = plot_belsar_metrics(metrics, fig=fig, ax=axs[i], xmin=xmin, xmax=xmax)
        axs[i].set_title(model_info["plot_name"])
    metrics = model_results["SNAP"]
    fig, _ = plot_belsar_metrics(metrics, fig=fig, ax=axs[-1], xmin=xmin, xmax=xmax)
    axs[-1].set_title("SNAP")
    if res_dir is not None:
        fig.savefig(os.path.join(res_dir, f"lai_belsar_validation{suffix}.png"))

    fig, axs = plt.subplots(nrows=1, ncols=n_models-1, dpi=150, figsize=(6*n_models, 6))
    for i, (model_name, model_info) in enumerate(model_dict.items()):
        # sub_variable = "lai" if variable in ["lai", "lai_eff"] else "ccc"
        metrics = model_results[model_name]
        fig, _ = plot_belsar_metrics(metrics, fig=fig, ax=axs[i], variable="cm")
        axs[i].set_title(model_info["plot_name"])
    if res_dir is not None:
        fig.savefig(os.path.join(res_dir, f"cm_belsar_validation{suffix}.png"))

def plot_comparative_results(model_dict, all_s2_r, all_snap_lai, all_snap_cab,
                             all_snap_cw, info_test_data, rmse_dict, picp_dict, res_dir=None):
    if len(model_dict) < 7:
        for i in range(all_s2_r.size(0)):
            info = info_test_data[i]
            if res_dir is not None:
                res_dir_i = os.path.join(res_dir, f"{i}_{info[1]}_{info[2]}")
                if not os.path.isdir(res_dir_i):
                    os.makedirs(res_dir_i)
            
            fig, _ = plot_patches(patch_list = [all_s2_r[i,...]] 
                                + [model_info["reconstruction"][i,...] for _, model_info in model_dict.items()],
                                title_list = [f"Sentinel {info[0]} \n"
                                                f"{info[1][:4]}/{info[1][4:6]}/{info[1][6:]} - {info[2]}"] 
                                                + [model_info["plot_name"] for _, model_info in model_dict.items()])
            if res_dir is not None:
                fig.savefig(os.path.join(res_dir_i, f"{i}_{info[1]}_{info[2]}_patch_reconstructions_rgb.png"))

            fig, _ = plot_patches(patch_list = [all_s2_r[i,torch.tensor([8,3,6]),...]]
                                + [model_info["reconstruction"][i,torch.tensor([8,3,6]),...]
                                    for _, model_info in model_dict.items()],
                                title_list = [f"Sentinel {info[0]} \n"
                                                f"{info[1][:4]}/{info[1][4:6]}/{info[1][6:]} - {info[2]}"] 
                                                + [model_info["plot_name"] for _, model_info in model_dict.items()])
            if res_dir is not None:
                fig.savefig(os.path.join(res_dir_i, f"{i}_{info[1]}_{info[2]}_patch_rec_B8B5B11.png"))
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
                fig.savefig(os.path.join(res_dir_i, f"{i}_{info[1]}_{info[2]}_patch_err.png"))
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
                fig.savefig(os.path.join(res_dir_i, f"{i}_{info[1]}_{info[2]}_LAI.png"))
            plt.close('all')
    lai_scatter_dict = {}
    lai_scatter_dict["SNAP's Biophysical Processor"] = all_snap_lai.squeeze(1).reshape(-1)
    global_lim = [lai_scatter_dict["SNAP's Biophysical Processor"].min().item(),
                    lai_scatter_dict["SNAP's Biophysical Processor"].max().item()]
    # for _, model_info in model_dict.items():
    #     lai_scatter_dict[model_info["plot_name"]] = model_info["prosail_vars"][:,6,...].reshape(-1)
    #     global_lim[0] = min(global_lim[0], lai_scatter_dict[model_info["plot_name"]].min().item())
    #     global_lim[1] = max(global_lim[1], lai_scatter_dict[model_info["plot_name"]].max().item())
    # fig, _ = regression_pair_plot(lai_scatter_dict, global_lim)
    # if res_dir is not None:
    #     fig.savefig(os.path.join(res_dir, "model_lai_comparison.png"))

    err_scatter_dict = {}
    err_boxplot_dict = {}
    global_lim = [0, 1]
    for method in ['simple_interpolate']:
        for variable in ['lai', "lai_eff"]:
            fig_rmse_c, ax_rmse_c = plt.subplots(dpi=150)
            fig_picp_c, ax_picp_c = plt.subplots(dpi=150)
            fig_rmse_l, ax_rmse_l = plt.subplots(dpi=150)
            fig_picp_l, ax_picp_l = plt.subplots(dpi=150)
            for i, (model_name, model_info) in enumerate(model_dict.items()):
                rmse = rmse_dict[method][variable][model_name]['Campaign']['All'].values[0]
                picp = picp_dict[method][variable][model_name]['Campaign']['All'].values[0]
                loss = model_info['loss']
                _, _, r2_cyclical, rmse_cyclical = regression_metrics(model_info["cyclical_ref_lai"].detach().cpu().numpy(), 
                                                                      model_info["cyclical_lai"].detach().cpu().numpy())

                ax_rmse_c.scatter(rmse_cyclical, rmse, label=str(i))
                ax_picp_c.scatter(rmse_cyclical, picp, label=str(i))
                ax_rmse_l.scatter(loss, rmse, label=str(i))
                ax_picp_l.scatter(loss, picp, label=str(i))
            if not i > 5:
                ax_rmse_c.legend()
                ax_picp_c.legend()
                ax_rmse_l.legend()
                ax_picp_l.legend()
            ax_rmse_c.set_xlabel("RMSE on simulated LAI")
            ax_picp_c.set_xlabel("RMSE on simulated LAI")
            ax_rmse_l.set_xlabel("Reconstruction loss (NLL)")
            ax_picp_l.set_xlabel("Reconstruction loss (NLL)")
            ax_rmse_c.set_ylabel("Validation RMSE")
            ax_picp_c.set_ylabel("Validation PICP")
            ax_rmse_l.set_ylabel("Validation RMSE")
            ax_picp_l.set_ylabel("Validation PICP")
            if res_dir is not None:
                fig_picp_c.savefig(os.path.join(res_dir, f"picp_vs_cyclical_rmse_{method}_{variable}.png"))
                fig_picp_l.savefig(os.path.join(res_dir, f"picp_vs_rec_loss_{method}_{variable}.png"))
                fig_rmse_c.savefig(os.path.join(res_dir, f"rmse_vs_cyclical_rmse_{method}_{variable}.png"))
                fig_rmse_l.savefig(os.path.join(res_dir, f"rmse_vs_rec_loss_{method}_{variable}.png"))
    for _, model_info in model_dict.items():
        err_boxplot_dict[model_info["plot_name"]] = (model_info["reconstruction"][:,:10,...]
                                                     - all_s2_r[:,:10,...])
        err_scatter_dict[model_info["plot_name"]] = err_boxplot_dict[model_info["plot_name"]].pow(2).mean(1).sqrt().reshape(-1)
        
        global_lim[0] = min(global_lim[0], err_scatter_dict[model_info["plot_name"]].min().item())
        global_lim[1] = max(global_lim[1], err_scatter_dict[model_info["plot_name"]].max().item())
    # fig, _ = regression_pair_plot(err_scatter_dict, global_lim)
    # if res_dir is not None:
    #     fig.savefig(os.path.join(res_dir, "rec_error_comparison.png"))

    fig, axs = plt.subplots(1, len(err_scatter_dict), figsize=(3*len(err_scatter_dict), 3), dpi=200, tight_layout=True)
    if len(err_scatter_dict)==1:
        axs = [axs]
    for i, (_, model_info) in enumerate(model_dict.items()):
        axs[i].hist(err_scatter_dict[model_info["plot_name"]], bins=200, range=global_lim)
        axs[i].plot(global_lim, global_lim, 'k--')
        axs[i].set_title(model_info["plot_name"])
    if res_dir is not None:
        fig.savefig(os.path.join(res_dir, "model_err_hist.png"))

    fig, axs = plt.subplots(1, len(err_scatter_dict), figsize=(3*len(err_scatter_dict), 3), dpi=200, tight_layout=True)
    if len(err_scatter_dict)==1:
        axs = [axs]
    for i, (_, model_info) in enumerate(model_dict.items()):
        regression_plot(pd.DataFrame({"Simulated LAI":model_info["cyclical_ref_lai"].detach().cpu().numpy(), 
                                      "Predicted LAI":model_info["cyclical_lai"].detach().cpu().numpy()}), 
                        "Simulated LAI", "Predicted LAI", hue=None, fig=fig, ax=axs[i], s=5)
        axs[i].set_title(model_info["plot_name"])
    if res_dir is not None:
        fig.savefig(os.path.join(res_dir, "model_cyclical_lai_scatter.png"))

    fig, axs = plt.subplots(2,5, dpi=150, tight_layout=True)
    for i in range(10):
        row = i % 2
        col = i//2
        for j, (_, model_info) in enumerate(model_dict.items()):
            err = err_boxplot_dict[model_info["plot_name"]][:,i,...].abs().reshape(-1)
            axs[row, col].boxplot(err, positions=[j], showfliers=False)
        # axs[row, col].set_xticklabels(recs_rdown.keys())
        axs[row, col].ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)
        axs[row, col].set_title(BANDS[i])
        axs[row, 0].set_ylabel('Absolute Error')
        axs[1, col].set_xlabel('Model')
    if res_dir is not None:
        fig.savefig(os.path.join(res_dir, "model_err_boxplot.png"))

    fig, axs = plt.subplots(1, len(err_scatter_dict), figsize=(3*len(err_scatter_dict), 3), dpi=200)
    if len(err_scatter_dict) == 1:
        axs = [axs]
    for i, (_, model_info) in enumerate(model_dict.items()):
        axs[i].hist(err_boxplot_dict[model_info["plot_name"]][:,i,...].reshape(-1), bins=200, range=global_lim)
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


# def get_models_global_metrics(models_dict, results_dict, sites, variable='lai', n_models=2, n_sigma=3):
#     methods = results_dict.keys()
#     rmse = np.zeros((len(methods), n_models, len(sites)+1))
#     picp = np.zeros((len(methods), n_models-1, len(sites)+1))
#     for i, method in enumerate(methods):
#         for j, model in enumerate(results_dict[method][variable].keys()):
#             results = results_dict[method][variable][model]
#             for k, site in enumerate(sites):
#                 site_results = results[results['Site']==site]
#                 rmse[i,j,k] = np.sqrt((site_results['Predicted LAI'] - site_results['LAI']).pow(2).mean())
#             rmse[i,j,-1] = np.sqrt((results['Predicted LAI'] - results['LAI']).pow(2).mean())
#     for i, method in enumerate(methods):
#         for j, model in enumerate(models_dict.keys()):
#             results = results_dict[method][variable][model]
#             for k, site in enumerate(sites):
#                 site_results = results[results['Site']==site]
#                 picp[i,j,k] = np.logical_and(site_results['LAI'] < site_results['Predicted LAI'] + n_sigma/2 * site_results['Predicted LAI std'],
#                                              site_results['LAI'] > site_results['Predicted LAI'] - n_sigma/2 * site_results['Predicted LAI std']).astype(int).mean()
#             picp[i,j,-1] = np.logical_and(results['LAI'] < results['Predicted LAI'] + n_sigma/2 * results['Predicted LAI std'],
#                                              results['LAI'] > results['Predicted LAI'] - n_sigma/2 * results['Predicted LAI std']).astype(int).mean()
    
#     return rmse, picp

def get_frm4veg_validation_metrics(model_dict, frm4veg_data_dir, filenames, method, mode, get_error=True):
    frm4veg_results = {}
    if isinstance(filenames, str):
        for _, (model_name, model_info) in enumerate(tqdm(model_dict.items())):
            frm4veg_results[model_name] = get_frm4veg_results_at_date(model_info["model"], frm4veg_data_dir, BARRAX_2021_FILENAME,
                                                                      is_SNAP=False, get_reconstruction=get_error)
        frm4veg_results["SNAP"] = get_frm4veg_results_at_date(model_info["model"], frm4veg_data_dir, BARRAX_2021_FILENAME,
                                                                      is_SNAP=True, get_reconstruction=get_error)
    else:
        for _, (model_name, model_info) in enumerate(tqdm(model_dict.items())):
            frm4veg_results[model_name] = interpolate_frm4veg_pred(model_info["model"], frm4veg_data_dir, filenames[0], 
                                                                filenames[1],  method=method,  is_SNAP=False, mode=mode)
        frm4veg_results["SNAP"] = interpolate_frm4veg_pred(model_info["model"], frm4veg_data_dir, filenames[0], 
                                                           filenames[1],  method=method, is_SNAP=True, mode=mode)
    return frm4veg_results

def get_belsar_x_frm4veg_lai_validation_results(model_dict, belsar_results, barrax_results, barrax_2021_results, wytham_results,
                                                frm4veg_lai="lai", get_reconstruction_error=False):
    results = {}
    for _, (model_name, model_info) in enumerate(tqdm(model_dict.items())):
        results[model_name] = get_belsar_x_frm4veg_lai_results(belsar_results[model_name], 
                                                                barrax_results[model_name], 
                                                                barrax_2021_results[model_name],
                                                                wytham_results[model_name],
                                                                frm4veg_lai=frm4veg_lai, 
                                                                get_reconstruction_error=get_reconstruction_error)
        
    results["SNAP"] = get_belsar_x_frm4veg_lai_results(belsar_results["SNAP"],  barrax_results["SNAP"], 
                                                       barrax_2021_results["SNAP"], 
                                                        wytham_results["SNAP"], frm4veg_lai=frm4veg_lai,
                                                        get_reconstruction_error=False)
    return results

def get_frm4veg_ccc_validation_results(model_dict, barrax_results, barrax_2021_results, wytham_results,
                                                frm4veg_lai="lai", get_reconstruction_error=False):
    results = {}
    for _, (model_name, model_info) in enumerate(tqdm(model_dict.items())):
        results[model_name] = get_frm4veg_ccc_results(barrax_results[model_name], 
                                                                barrax_2021_results[model_name],
                                                                wytham_results[model_name],
                                                                frm4veg_lai=frm4veg_lai, 
                                                                get_reconstruction_error=get_reconstruction_error)
    results["SNAP"] = get_frm4veg_ccc_results(barrax_results["SNAP"], 
                                                       barrax_2021_results["SNAP"], 
                                                        wytham_results["SNAP"], frm4veg_lai=frm4veg_lai,
                                                        get_reconstruction_error=False)
    return results

def compare_validation_regressions(model_dict, belsar_dir, frm4veg_data_dir, frm4veg_2021_data_dir,
                                   res_dir, list_belsar_filenames, recompute=True, mode="lat_mode", get_error=True):
    if recompute:
        save_snap_belsar_predictions(belsar_dir, res_dir, list_belsar_filenames)
    for _, (model_name, model_info) in enumerate(tqdm(model_dict.items())):
        model = model_info["model"]
        if recompute:
            save_belsar_predictions(belsar_dir, model, res_dir, list_belsar_filenames, model_name=model_name, mode=mode, save_reconstruction=get_error)


    belsar_results = {}
    barrax_results = {}
    barrax_2021_results = {}
    wytham_results = {}
    validation_lai_results = {}
    validation_ccc_results = {}
    lai_picp_dict = {}
    lai_rmse_dict = {}
    ccc_picp_dict = {}
    ccc_rmse_dict = {}
    for method in ["simple_interpolate"]:#, "best", "worst"]: #'closest', 
        lai_picp_dict[method] = {}
        lai_rmse_dict[method] = {}
        belsar_results[method] = get_belsar_validation_results(model_dict, belsar_dir, res_dir, method=method, mode=mode, get_error=get_error)
        # plot_belsar_validation_results_comparison(model_dict, belsar_results[method], res_dir, suffix="_" + method)

        # barrax_filenames = ["2B_20180516_FRM_Veg_Barrax_20180605", "2A_20180613_FRM_Veg_Barrax_20180605"]
        barrax_results[method] = get_frm4veg_validation_metrics(model_dict, frm4veg_data_dir, BARRAX_FILENAMES,
                                                                method=method, mode=mode)
        barrax_2021_results[method] = get_frm4veg_validation_metrics(model_dict, frm4veg_2021_data_dir, BARRAX_2021_FILENAME,
                                                                    method=method, mode=mode)
        # plot_frm4veg_results_comparison(model_dict, barrax_results[method], frm4veg_data_dir, barrax_filenames[0],
        #                                 res_dir=res_dir, prefix= "barrax_"+method+"_")
        # wytham_filenames = ["2A_20180629_FRM_Veg_Wytham_20180703", "2A_20180706_FRM_Veg_Wytham_20180703"]
        wytham_results[method] = get_frm4veg_validation_metrics(model_dict, frm4veg_data_dir, WYTHAM_FILENAMES,
                                                                method=method, mode=mode)
        # plot_frm4veg_results_comparison(model_dict, wytham_results[method], frm4veg_data_dir, wytham_filenames[0],
        #                                 res_dir=res_dir, prefix= "wytham_"+method+"_")
        validation_lai_results[method] = {}
        validation_ccc_results[method] = {}
        for variable in ['lai', "lai_eff"]:
            print(method, variable)
            lai_picp_dict[method][variable] = {}
            lai_rmse_dict[method][variable] = {}
            validation_lai_results[method][variable] = get_belsar_x_frm4veg_lai_validation_results(model_dict, belsar_results[method],
                                                                                                   barrax_results[method],
                                                                                                   barrax_2021_results[method],
                                                                                                   wytham_results=wytham_results[method],
                                                                                                   frm4veg_lai=variable,
                                                                                                   get_reconstruction_error=True)
            for model, df_results in validation_lai_results[method][variable].items():
                df_results.to_csv(os.path.join(res_dir, f"{mode}_{method}_{variable}_{model}.csv"))
                rmse, picp = get_validation_global_metrics(df_results, decompose_along_columns=["Campaign"])
                lai_picp_dict[method][variable][model] = picp
                lai_rmse_dict[method][variable][model] = rmse
            if not len(model_dict) > 6: 
                plot_lai_validation_comparison(model_dict, validation_lai_results[method][variable],
                                            res_dir=res_dir, prefix=f"{mode}_{method}_{variable}",
                                            margin = 0.02, hue_perfs=True)
                
                plot_lai_validation_comparison(model_dict, validation_lai_results[method][variable],
                                            res_dir=res_dir, prefix=f"{mode}_{method}_{variable}_Land_cover",
                                            margin = 0.02, hue="Land cover")
                
                plot_lai_validation_comparison(model_dict, validation_lai_results[method][variable],
                                            res_dir=res_dir, prefix=f"{mode}_{method}_{variable}_Time_delta",
                                            margin = 0.02, hue="Time delta")
                
                plot_lai_validation_comparison(model_dict, validation_lai_results[method][variable],
                                            res_dir=res_dir, prefix=f"{mode}_{method}_{variable}_Campaign",
                                            margin = 0.02, hue="Campaign", legend_col=2, hue_perfs=True)
            plt.close('all')
            
            # get_models_global_metrics(model_dict, validation_lai_results, 
            #                                        sites=["Spain", "England", "Belgium"], 
            #                                        variable=variable, n_models=len(model_dict)+1, n_sigma=3)

            np.save(os.path.join(res_dir, f"{mode}_{method}_{variable}_Campaign_rmse.npy"), rmse)
            np.save(os.path.join(res_dir, f"{mode}_{method}_{variable}_Campaign_picp.npy"), picp)

        for variable in ['ccc', "ccc_eff"]:
            print(method, variable)
            ccc_picp_dict[method][variable] = {}
            ccc_rmse_dict[method][variable] = {}
            validation_lai_results[method][variable] = get_frm4veg_ccc_validation_results(model_dict, belsar_results[method],
                                                                                                   barrax_results[method],
                                                                                                   barrax_2021_results[method],
                                                                                                   wytham_results=wytham_results[method],
                                                                                                   frm4veg_lai=variable,
                                                                                                   get_reconstruction_error=True)
            for model, df_results in validation_lai_results[method][variable].items():
                df_results.to_csv(os.path.join(res_dir, f"{mode}_{method}_{variable}_{model}.csv"))
                rmse, picp = get_validation_global_metrics(df_results, decompose_along_columns=["Campaign"])
                ccc_picp_dict[method][variable][model] = picp
                ccc_rmse_dict[method][variable][model] = rmse
            if not len(model_dict) > 6: 
                plot_lai_validation_comparison(model_dict, validation_lai_results[method][variable],
                                            res_dir=res_dir, prefix=f"{mode}_{method}_{variable}",
                                            margin = 0.02, hue_perfs=True)
                
                plot_lai_validation_comparison(model_dict, validation_lai_results[method][variable],
                                            res_dir=res_dir, prefix=f"{mode}_{method}_{variable}_Land_cover",
                                            margin = 0.02, hue="Land cover")
                
                plot_lai_validation_comparison(model_dict, validation_lai_results[method][variable],
                                            res_dir=res_dir, prefix=f"{mode}_{method}_{variable}_Time_delta",
                                            margin = 0.02, hue="Time delta")
                
                plot_lai_validation_comparison(model_dict, validation_lai_results[method][variable],
                                            res_dir=res_dir, prefix=f"{mode}_{method}_{variable}_Campaign",
                                            margin = 0.02, hue="Campaign", legend_col=2, hue_perfs=True)
            plt.close('all')
            
            # get_models_global_metrics(model_dict, validation_lai_results, 
            #                                        sites=["Spain", "England", "Belgium"], 
            #                                        variable=variable, n_models=len(model_dict)+1, n_sigma=3)

            np.save(os.path.join(res_dir, f"{mode}_{method}_{variable}_Campaign_rmse.npy"), rmse)
            np.save(os.path.join(res_dir, f"{mode}_{method}_{variable}_Campaign_picp.npy"), picp)
    # save_dict(rmse_dict, os.path.join(res_dir, f"LAI_Campaign_rmse.json"))
    # save_dict(picp_dict, os.path.join(res_dir, f"LAI_Campaign_picp.json"))
    return lai_rmse_dict, lai_picp_dict, ccc_rmse_dict, ccc_picp_dict
    # else:
    # barrax_filename_before = "2B_20180516_FRM_Veg_Barrax_20180605"
    # sensor = "2B"
    # barrax_results_before = get_model_frm4veg_results(model_dict, frm4veg_data_dir, barrax_filename_before, sensor)
    # plot_frm4veg_results_comparison(model_dict, barrax_results_before, frm4veg_data_dir, barrax_filename_before,
    #                                    res_dir=res_dir)

    # barrax_filename_after = "2A_20180613_FRM_Veg_Barrax_20180605"
    # sensor = "2A"
    # barrax_results_after = get_model_frm4veg_results(model_dict, frm4veg_data_dir, barrax_filename_after, sensor)
    # plot_frm4veg_results_comparison(model_dict, barrax_results_after, frm4veg_data_dir, barrax_filename_after, 
    #                                    res_dir=res_dir)

def get_models_validation_rec_loss(model_dict, loader):
    for model_name, model_info in model_dict.items(): 
        if model_info["supervised"]:
            model_info['loss'] = 0.0
        else:
            loss_dict = model_info["model"].validate(loader, n_samples=10 if not socket.gethostname()=='CELL200973' else 2)
            if "rec_loss" in loss_dict.keys():
                model_info['loss'] = loss_dict["rec_loss"]
            else:
                model_info['loss'] = loss_dict["loss_sum"]
    return

def main():
    """
    main.
    """
    if socket.gethostname()=='CELL200973':
        args = ["-m","/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/config/model_dict_dev.json",
                "-d", "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/patches/",
                "-r", "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/results/comparaison/"]
        parser = get_parser().parse_args(args)
        frm4veg_data_dir = "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/frm4veg_validation"
        frm4veg_2021_data_dir = "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/frm4veg_2021_validation"
        belsar_dir = "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/belSAR_validation"
    else:
        parser = get_parser().parse_args()
        frm4veg_data_dir = "/work/scratch/zerahy/prosailvae/data/frm4veg_validation"
        frm4veg_2021_data_dir = "/work/scratch/zerahy/prosailvae/data/frm4veg_2021_validation"
        belsar_dir = "/work/scratch/zerahy/prosailvae/data/belSAR_validation"
    res_dir = parser.res_dir
    if not os.path.isdir(res_dir):
        os.makedirs(res_dir)
    list_belsar_filenames = ["2A_20180508_both_BelSAR_agriculture_database",
                            "2A_20180518_both_BelSAR_agriculture_database",
                            "2A_20180528_both_BelSAR_agriculture_database",
                            "2A_20180620_both_BelSAR_agriculture_database",
                            "2A_20180627_both_BelSAR_agriculture_database",
                            "2B_20180715_both_BelSAR_agriculture_database",
                            "2B_20180722_both_BelSAR_agriculture_database",
                            "2A_20180727_both_BelSAR_agriculture_database",
                            "2B_20180804_both_BelSAR_agriculture_database"]  
    model_dict, test_loader, valid_loader, info_test_data = get_model_and_dataloader(parser)
    get_models_validation_rec_loss(model_dict, valid_loader)
    for mode in ["sim_tg_mean"]: # , "lat_mode"]
        recompute = True if not socket.gethostname()=='CELL200973' else False
        (lai_rmse_dict, lai_picp_dict, 
         ccc_rmse_dict, ccc_picp_dict) = compare_validation_regressions(model_dict, belsar_dir, 
                                                              frm4veg_data_dir, frm4veg_2021_data_dir, 
                                                              res_dir, list_belsar_filenames, 
                                                              recompute=recompute, mode=mode)
    
    (model_dict, all_s2_r, all_snap_lai, all_snap_cab,
     all_snap_cw) = get_model_results(model_dict, test_loader, info_test_data, 
                                      max_patch=30 if not socket.gethostname()=='CELL200973' else 2)
    plot_comparative_results(model_dict, all_s2_r, all_snap_lai, all_snap_cab, all_snap_cw, info_test_data, 
                             lai_rmse_dict, lai_picp_dict, res_dir,
                             )
    # compare_snap_versions_on_real_data(test_loader, res_dir)
    # compare_snap_versions_on_weiss_data(res_dir)
    
if __name__=="__main__":
    main()