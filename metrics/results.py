import os
import logging
import argparse
import socket
import torch
from prosailvae import __path__ as PPATH
TOP_PATH = os.path.join(PPATH[0], os.pardir)
from .metrics_utils import get_metrics, save_metrics, get_juan_validation_metrics, get_weiss_validation_metrics
from .prosail_plots import (plot_metrics, plot_rec_and_latent, loss_curve, plot_param_dist, plot_pred_vs_tgt, 
                                    plot_refl_dist, pair_plot, plot_rec_error_vs_angles, plot_lat_hist2D, plot_rec_hist2D, 
                                    plot_metric_boxplot, plot_patch_pairs, plot_lai_preds, plot_single_lat_hist_2D,
                                    all_loss_curve, plot_patches, plot_lai_vs_ndvi, PROSAIL_2D_res_plots, PROSAIL_2D_aggregated_results,
                                    frm4veg_plots, plot_belsar_metrics, regression_plot)
from dataset.loaders import  get_simloader
from dataset.weiss_utils import get_weiss_biophyiscal_from_batch
from dataset.frm4veg_validation import load_frm4veg_data
from metrics.belsar_metrics import compute_metrics_at_date
from prosailvae.ProsailSimus import PROSAILVARS, BANDS

from utils.utils import load_dict, save_dict
from utils.image_utils import get_encoded_image_from_batch
from prosailvae.prosail_vae import load_prosail_vae_with_hyperprior


from datetime import datetime 
import shutil
from time import sleep
import warnings
from mmdc_singledate.datamodules.mmdc_datamodule import destructure_batch
from torchutils.patches import patchify, unpatchify 
import pandas as pd
import numpy as np
from dataset.belsar_validation import load_belsar_validation_data
from utils.image_utils import tensor_to_raster
from snap_regression.snap_nn import SnapNN

LOGGER_NAME = "PROSAIL-VAE results logger"

def get_prosailvae_results_parser():
    """
    Creates a new argument parser.
    """
    parser = argparse.ArgumentParser(description='Parser for data generation')

    parser.add_argument("-c", dest="config_file",
                        help="name of config json file on config directory.",
                        type=str, default="config.json")

    parser.add_argument("-d", dest="data_dir",
                        help="path to data direcotry",
                        type=str, default="/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/")
    
    parser.add_argument("-r", dest="root_results_dir",
                        help="path to root results direcotry",
                        type=str, default="")

    parser.add_argument("-rsr", dest="rsr_dir",
                        help="directory of rsr_file",
                        type=str, default='/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/')
    
    parser.add_argument("-t", dest="tensor_dir",
                        help="directory of mmdc tensor files",
                        type=str, default="/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/real_data/torchfiles/")

    parser.add_argument("-p", dest="plot_results",
                        help="toggle results plotting",
                        type=bool, default=False)  
    return parser


def get_belsar_x_frm4veg_lai_metrics(belsar_metrics, frm4veg_data_dir, frm4veg_barrax_lai_pred, frm4veg_barrax_filename,
                                     frm4veg_wytham_lai_pred=None, frm4veg_wytham_filename=None,  lai_eff=False):
    # belsar_metrics['crop'] = belsar_metrics["name"].apply(lambda x: "wheat" if x[0]=="W" else "maize")
    belsar_pred_at_site = belsar_metrics[f"parcel_{'lai'}_mean"].values
    belsar_ref = belsar_metrics[f"{'lai'}_mean"].values
    site = ["Belgium"] * len(belsar_ref)
    variable = "lai"
    if lai_eff:
        variable = "lai_eff"

    if isinstance(frm4veg_barrax_lai_pred, torch.Tensor):
        frm4veg_barrax_lai_pred = frm4veg_barrax_lai_pred.numpy()
    gdf_barrax_lai, _, _, _, _ = load_frm4veg_data(frm4veg_data_dir, frm4veg_barrax_filename, variable=variable)
    barrax_ref = gdf_barrax_lai[variable].values.reshape(-1)
    # ref_uncert = gdf_barrax_lai["uncertainty"].values
    x_idx = gdf_barrax_lai["x_idx"].values.astype(int)
    y_idx = gdf_barrax_lai["y_idx"].values.astype(int)
    barrax_pred_at_site = frm4veg_barrax_lai_pred[:, y_idx, x_idx].reshape(-1)
    site = site + ["Spain"] * len(barrax_ref)

    pred_at_site = np.concatenate((belsar_pred_at_site, barrax_pred_at_site))
    ref = np.concatenate((belsar_ref, barrax_ref))
    if frm4veg_wytham_lai_pred is not None and frm4veg_wytham_filename is not None:
        if isinstance(frm4veg_wytham_lai_pred, torch.Tensor):
            frm4veg_wytham_lai_pred = frm4veg_wytham_lai_pred.numpy()
        gdf_wytham_lai, _, _, _, _ = load_frm4veg_data(frm4veg_data_dir, frm4veg_wytham_filename, variable=variable)
        wytham_ref = gdf_wytham_lai[variable].values.reshape(-1)
        # ref_uncert = gdf_wytham_lai["uncertainty"].values
        x_idx = gdf_wytham_lai["x_idx"].values.astype(int)
        y_idx = gdf_wytham_lai["y_idx"].values.astype(int)
        wytham_pred_at_site = frm4veg_wytham_lai_pred[:, y_idx, x_idx].reshape(-1)
        pred_at_site = np.concatenate((pred_at_site, wytham_pred_at_site))
        ref = np.concatenate((ref, wytham_ref))
        site = site + ["England"] * len(wytham_ref)
    site = np.array(site)
    validation_metrics = pd.DataFrame(data={"Predicted LAI":pred_at_site, "LAI": ref, "Site":site})
    return validation_metrics


def save_results_2d(PROSAIL_VAE, loader, res_dir, all_train_loss_df=None, 
                    all_valid_loss_df=None, info_df=None, LOGGER_NAME='PROSAIL-VAE logger', 
                    plot_results=False, info_test_data=None,
                    frm4veg_data_dir = "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/frm4veg_validation",
                    frm4veg_barrax_filename = "FRM_Veg_Barrax_20180605", 
                    frm4veg_wytham_filename = None, 
                    max_test_patch=50,
                    belsar_dir="",
                    list_belsar_filenames=[]):
    rec_mode = 'lat_mode' # if not socket.gethostname()=='CELL200973' else "random"
    image_tensor_file_names = ["after_SENTINEL2B_20171127-105827-648_L2A_T31TCJ_C_V2-2_roi_0.pth"]
    image_tensor_aliases = ["S2B_27_nov_2017_T31TCJ"]
    device = PROSAIL_VAE.device
    logger = logging.getLogger(LOGGER_NAME)
    logger.info("Saving Loss")
    # Saving Loss
    loss_dir = res_dir + "/loss/"
    if not os.path.isdir(loss_dir):
        os.makedirs(loss_dir)
    
    if all_train_loss_df is not None:
        all_train_loss_df.to_csv(loss_dir + "train_loss.csv")
        if plot_results:
            loss_curve(all_train_loss_df, save_file=loss_dir+"train_loss.svg")
            loss_curve(all_train_loss_df[["epoch", "loss_sum"]], save_file=loss_dir+"train_loss_sum.svg")
    if all_valid_loss_df is not None:
        all_valid_loss_df.to_csv(loss_dir + "valid_loss.csv")
        if plot_results:
            loss_curve(all_valid_loss_df, save_file=loss_dir+"valid_loss.svg")
            loss_curve(all_valid_loss_df[["epoch", "loss_sum"]], save_file=loss_dir+"train_loss_sum.svg")
    if info_df is not None:
        if plot_results:
            loss_curve(info_df, save_file=loss_dir+"lr.svg")
            all_loss_curve(all_train_loss_df[["epoch", "loss_sum"]], all_valid_loss_df[["epoch", "loss_sum"]], 
                           info_df, save_file=loss_dir+"all_loss_sum.svg")
            all_loss_curve(all_train_loss_df, all_valid_loss_df, info_df, save_file=loss_dir+"all_loss.svg")
    
    # Computing metrics
    PROSAIL_VAE.eval()
    logger.info("Computing inference metrics with test dataset...")
    # test_loss = PROSAIL_VAE.validate(loader, mmdc_dataset=True, n_samples=10)
    # pd.DataFrame(test_loss, index=[0]).to_csv(loss_dir + "/test_loss.csv")
    if not plot_results:
        return
    
    plot_dir = res_dir + "/plots/"
    if not os.path.isdir(plot_dir):
        os.makedirs(plot_dir)
    
    rm4veg_validation_plot_dir = plot_dir + "/frm4veg_validation/"
    if not os.path.isdir(rm4veg_validation_plot_dir):
        os.makedirs(rm4veg_validation_plot_dir)
    
    belsar_res_dir = plot_dir + "belsar_validation"
    if not os.path.isdir(belsar_res_dir):
        os.makedirs(belsar_res_dir)

    _, s2_r, s2_a, _, _ = load_frm4veg_data(frm4veg_data_dir, frm4veg_barrax_filename, variable="lai")
    s2_r = torch.from_numpy(s2_r).float().unsqueeze(0)
    s2_a = torch.from_numpy(s2_a).float().unsqueeze(0)
    with torch.no_grad():
        (_, sim_image, cropped_s2_r, cropped_s2_a, 
         _) = get_encoded_image_from_batch((s2_r, s2_a), PROSAIL_VAE, patch_size=32, bands=torch.arange(10),
                                            mode=rec_mode, padding=True, no_rec=True)
        
        frm4veg_barrax_lai_pred = sim_image[6,...].unsqueeze(0)
        frm4veg_barrax_ccc_pred = sim_image[1,...].unsqueeze(0) * frm4veg_barrax_lai_pred
        # (snap_validation_lai, _,
        #     _) = get_weiss_biophyiscal_from_batch((cropped_s2_r, cropped_s2_a),
        #                                                     patch_size=32, sensor="2A")
        # gdf_lai, _, _, _, _ = load_frm4veg_data(frm4veg_data_dir, frm4veg_barrax_filename, variable="lai")
    
    frm4veg_plots(frm4veg_barrax_lai_pred, frm4veg_barrax_ccc_pred, frm4veg_data_dir, frm4veg_barrax_filename,
                  s2_r=cropped_s2_r, res_dir=rm4veg_validation_plot_dir)
    
    frm4veg_wytham_lai_pred = None
    if frm4veg_wytham_filename is not None:
        _, s2_r, s2_a, _, _ = load_frm4veg_data(frm4veg_data_dir, frm4veg_wytham_filename, variable="lai")
        s2_r = torch.from_numpy(s2_r).float().unsqueeze(0)
        s2_a = torch.from_numpy(s2_a).float().unsqueeze(0)
        with torch.no_grad():
            (_, sim_image, cropped_s2_r, cropped_s2_a, 
             _) = get_encoded_image_from_batch((s2_r, s2_a), PROSAIL_VAE, patch_size=32, 
                                               bands=torch.arange(10), mode=rec_mode, padding=True, no_rec=True)
            
            frm4veg_wytham_lai_pred = sim_image[6,...].unsqueeze(0)
            frm4veg_wytham_ccc_pred = sim_image[1,...].unsqueeze(0) * frm4veg_wytham_lai_pred
        frm4veg_plots(frm4veg_wytham_lai_pred, frm4veg_wytham_ccc_pred, frm4veg_data_dir, frm4veg_wytham_filename,
                      s2_r=cropped_s2_r, res_dir=rm4veg_validation_plot_dir)


    save_belsar_predictions(belsar_dir, PROSAIL_VAE, belsar_res_dir, list_filenames=list_belsar_filenames)
    belsar_metrics = compute_metrics_at_date(belsar_dir, belsar_res_dir, method="closest", file_suffix="_pvae")
    fig, ax = plot_belsar_metrics(belsar_metrics)
    fig.savefig(os.path.join(belsar_res_dir, "belsar_regression_crop.png"))
    fig, ax = plot_belsar_metrics(belsar_metrics, hue='date')
    fig.savefig(os.path.join(belsar_res_dir, "belsar_regression_date.png"))
    fig, ax = plot_belsar_metrics(belsar_metrics, hue='delta')
    fig.savefig(os.path.join(belsar_res_dir, "belsar_regression_delta.png"))
    belsar_metrics_inter = compute_metrics_at_date(belsar_dir, belsar_res_dir, 
                                                   method="simple_interpolate", file_suffix="_pvae")
    fig, ax = plot_belsar_metrics(belsar_metrics_inter)
    fig.savefig(os.path.join(belsar_res_dir, "belsar_regression_interpolated_crop.png"))
    fig, ax = plot_belsar_metrics(belsar_metrics_inter, hue='date')
    fig.savefig(os.path.join(belsar_res_dir, "belsar_regression_interpolated_date.png"))
    # fig, ax = plot_belsar_metrics(belsar_metrics_inter, hue='delta')
    # fig.savefig(os.path.join(belsar_res_dir, "belsar_regression_interpolated_delta.png"))

    metrics_df = get_belsar_x_frm4veg_lai_metrics(belsar_metrics, frm4veg_data_dir,
                                                  frm4veg_barrax_lai_pred, frm4veg_barrax_filename,
                                                  frm4veg_wytham_lai_pred=frm4veg_wytham_lai_pred, 
                                                  frm4veg_wytham_filename=frm4veg_wytham_filename,  
                                                  lai_eff=False)
    
    fig, ax = regression_plot(metrics_df, x="LAI", y="Predicted LAI", 
                              fig=None, ax=None, hue="Site", legend_col=True, xmin=None, xmax=None)
    fig.savefig(os.path.join(belsar_res_dir, "belsar_x_frm4veg_regression.png"))
    # plot_rec_hist2D(PROSAIL_VAE, loader, res_dir, nbin=50)
    all_rec = []
    all_lai = []
    all_cab = []
    all_cw = []
    all_vars = []
    all_weiss_lai = []
    all_weiss_cab = []
    all_weiss_cw = []
    all_s2_r = []
    all_sigma = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            (rec_image, sim_image, cropped_s2_r, cropped_s2_a,
                sigma_image) = get_encoded_image_from_batch(batch, PROSAIL_VAE, patch_size=32,
                                                            bands=torch.arange(10),
                                                            mode=rec_mode)
            info = info_test_data[i,:]
            (weiss_lai, weiss_cab,
                weiss_cw) = get_weiss_biophyiscal_from_batch((cropped_s2_r,
                                                             cropped_s2_a),
                                                             patch_size=32, sensor=info[0])
            
            patch_plot_dir = plot_dir + f"/{i}_{info[1]}_{info[2]}/"
            if not os.path.isdir(patch_plot_dir):
                os.makedirs(patch_plot_dir)
            PROSAIL_2D_res_plots(patch_plot_dir, sim_image, cropped_s2_r.squeeze(), rec_image,
                                 weiss_lai, weiss_cab, weiss_cw, sigma_image, i, info=info)
            all_rec.append(rec_image.reshape(10,-1))
            all_lai.append(sim_image[6,...].reshape(-1))
            all_cab.append(sim_image[1,...].reshape(-1))
            all_cw.append(sim_image[4,...].reshape(-1))
            all_vars.append(sim_image.reshape(11,-1))
            all_weiss_lai.append(weiss_lai.reshape(-1))
            all_weiss_cab.append(weiss_cab.reshape(-1))
            all_weiss_cw.append(weiss_cw.reshape(-1))
            all_s2_r.append(cropped_s2_r.reshape(10,-1))
            all_sigma.append(sigma_image.reshape(11,-1))
            if i == max_test_patch-1:
                break
        all_rec = torch.cat(all_rec, axis=1)
        all_lai = torch.cat(all_lai)
        all_cab = torch.cat(all_cab)
        all_ccc = all_lai * all_cab
        
        all_cw = torch.cat(all_cw)
        all_vars = torch.cat(all_vars, axis=1)
        all_cw_rel =  1 - all_vars[5,...] / all_cw
        all_weiss_lai = torch.cat(all_weiss_lai)
        all_weiss_cab = torch.cat(all_weiss_cab)
        all_weiss_cw = torch.cat(all_weiss_cw)
        all_s2_r = torch.cat(all_s2_r, axis=1)
        all_sigma = torch.cat(all_sigma, axis=1)

        PROSAIL_2D_aggregated_results(plot_dir, all_s2_r, all_rec, all_lai, all_cab, all_cw, all_vars,
                                      all_weiss_lai, all_weiss_cab, all_weiss_cw, all_sigma, all_ccc, all_cw_rel,
                                    #   gdf_lai, lai_validation_pred, snap_validation_lai
                                      )

    logger.info("Metrics computed.")
    
    return 


def get_snap_belsar_predictions(belsar_dir, res_dir, list_belsar_filename):
    NO_DATA = -10000
    # filename = "2A_20180613_FRM_Veg_Barrax_20180605"
    for filename in list_belsar_filename: 
        ver = "3A" if filename[:2] == "2A" else "3B"
        model_lai = SnapNN(ver=ver, variable="lai")
        model_lai.set_weiss_weights()

        df, s2_r_image, s2_a, mask, xcoords, ycoords, crs = load_belsar_validation_data(belsar_dir, filename)
        s2_r = torch.from_numpy(s2_r_image)[torch.tensor([1,2,3,4,5,7,8,9]), ...].float()
        mask[mask==1.] = np.nan
        mask[mask==0.] = 1.
        if np.isnan(mask).all():
            print(f"No valid pixels in {filename}!")
        s2_r = s2_r * torch.from_numpy(mask).float()
        s2_a = torch.cos(torch.deg2rad(torch.from_numpy(s2_a).float()))
        s2_data = torch.concat((s2_r, s2_a), 0)
        with torch.no_grad():
            lai_pred = model_lai.forward(s2_data, spatial_mode=True)
        dummy_tensor = NO_DATA * torch.ones(3, lai_pred.size(1), lai_pred.size(2))
        tensor = torch.cat((lai_pred, dummy_tensor), 0)
        tensor[tensor.isnan()] = NO_DATA
        resolution = 10
        file_path = res_dir + f"/{filename}_SNAP.tif"
        tensor_to_raster(tensor, file_path,
                         crs=crs,
                         resolution=resolution,
                         dtype=np.float32,
                         bounds=None,
                         xcoords=xcoords,
                         ycoords=ycoords,
                         nodata=NO_DATA,
                         hw = 0, 
                         half_res_coords=True)

def save_belsar_predictions(belsar_dir, PROSAIL_VAE, res_dir, list_filenames, suffix="_pvae", mode="lat_mode"):
    NO_DATA = -10000
    for filename in list_filenames:
        df, s2_r, s2_a, mask, xcoords, ycoords, crs = load_belsar_validation_data(belsar_dir, filename)
        s2_r = torch.from_numpy(s2_r).float()
        mask[mask==1.] = np.nan
        mask[mask==0.] = 1.
        if np.isnan(mask).all():
            print(f"No valid pixels in {filename}!")
        s2_r = (s2_r * torch.from_numpy(mask).float()).unsqueeze(0)
        s2_a = torch.from_numpy(s2_a).float().unsqueeze(0)
        
        with torch.no_grad():
            (_, sim_image, _, _, sigma_image) = get_encoded_image_from_batch((s2_r, s2_a), PROSAIL_VAE,
                                                        patch_size=32, bands=torch.arange(10),
                                                        mode=mode, padding=True, no_rec=True)
        
        # lai_validation_pred = sim_image[6,...].unsqueeze(0)
        # cm_validation_pred = sim_image[5,...].unsqueeze(0)
        tensor = torch.cat((sim_image[6,...].unsqueeze(0),
                            sim_image[5,...].unsqueeze(0),
                            sigma_image[6,...].unsqueeze(0), 
                            sigma_image[5,...].unsqueeze(0)), 0)
        tensor[tensor.isnan()] = NO_DATA
        tensor_to_raster(tensor, res_dir + f"/{filename}{suffix}.tif",
                         crs=crs,
                            resolution=10,
                            dtype=np.float32,
                            bounds=None,
                            xcoords=xcoords,
                            ycoords=ycoords,
                            nodata= NO_DATA,
                            hw = 0,
                            half_res_coords=True)
    return

def save_results(PROSAIL_VAE, res_dir, data_dir, all_train_loss_df=None,
                 all_valid_loss_df=None, info_df=None, LOGGER_NAME='PROSAIL-VAE logger', plot_results=False,
                 juan_validation=True, weiss_mode=False, n_samples=1):
    bands_name = BANDS
    if weiss_mode:
        bands_name = ["B03", "B04", "B05", "B06", "B07", "B8A", "B11", "B12"]
    device = PROSAIL_VAE.device
    logger = logging.getLogger(LOGGER_NAME)
    logger.info("Saving Loss")
    # Saving Loss
    loss_dir = res_dir + "/loss/"
    if not os.path.isdir(loss_dir):
        os.makedirs(loss_dir)
    
    if all_train_loss_df is not None:
        all_train_loss_df.to_csv(loss_dir + "train_loss.csv")
        if plot_results:
            loss_curve(all_train_loss_df, save_file=loss_dir+"train_loss.svg")
    if all_valid_loss_df is not None:
        all_valid_loss_df.to_csv(loss_dir + "valid_loss.csv")
        if plot_results:
            loss_curve(all_valid_loss_df, save_file=loss_dir+"valid_loss.svg")
    if info_df is not None:
        if plot_results:
            loss_curve(info_df, save_file=loss_dir+"lr.svg")
            all_loss_curve(all_train_loss_df, all_valid_loss_df, info_df, 
                           save_file=loss_dir+"all_loss.svg")
    # Computing metrics
    logger.info("Loading test loader...")
    loader = get_simloader(file_prefix="test_", data_dir=data_dir)
    logger.info("Test loader, loaded.")
    
    alpha_pi = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5,
                0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    alpha_pi.reverse()
    PROSAIL_VAE.eval()
    logger.info("Computing inference metrics with test dataset...")
    test_loss = PROSAIL_VAE.validate(loader, n_samples=n_samples)
    pd.DataFrame(test_loss, index=[0]).to_csv(loss_dir + "/test_loss.csv")
    nlls = PROSAIL_VAE.compute_lat_nlls(loader).mean(0).squeeze()
    torch.save(nlls, res_dir + "/params_nll.pt")

    if weiss_mode:
        weiss_validation_dir = res_dir + "/weiss_validation/"
        if not os.path.isdir(weiss_validation_dir):
            os.makedirs(weiss_validation_dir)
        weiss_data_dir_path = os.path.join(data_dir, os.pardir) + "/weiss/"
        prosail_ref_params = torch.load(weiss_data_dir_path+ "weiss_test_prosail_sim_vars.pt").float().to(PROSAIL_VAE.device)
        s2_r = torch.load(weiss_data_dir_path + "weiss_test_prosail_s2_sim_refl.pt").float().to(PROSAIL_VAE.device)
        s2_a = prosail_ref_params[:,-3:]
        prosail_ref_params = prosail_ref_params[:,:-3]
        lai_nlls, lai_preds, sim_pdfs, sim_supports = get_weiss_validation_metrics(PROSAIL_VAE, s2_r, s2_a, prosail_ref_params)
        torch.save(lai_nlls.cpu(), weiss_validation_dir + f"/weiss_lai_nll.pt")
        torch.save(lai_preds.cpu(), weiss_validation_dir + f"/weiss_lai_ref_pred.pt")
        if plot_results:
            fig, ax = plot_lai_preds(lai_preds[:,1].cpu(), lai_preds[:,0].cpu(), site="weiss")
            fig.savefig(weiss_validation_dir + f"/weiss_lai_pred_vs_true.png")
            plot_single_lat_hist_2D(heatmap=None, extent=None, tgt_dist=lai_preds[:,1].cpu(), 
                                    sim_pdf=sim_pdfs[:,6,:].cpu(), sim_support=sim_supports[:,6,:].cpu(),
                                    res_dir=weiss_validation_dir, fig=None, ax=None, var_name="LAI", nbin=100)

    if juan_validation:
        juan_data_dir_path = TOP_PATH + "/field_data/processed/"
        juan_validation_dir = res_dir + "/juan_validation/"
        if not os.path.isdir(juan_validation_dir):
            os.makedirs(juan_validation_dir)
        sites = ["france", "spain1", "italy1", "italy2"]
        j_list_lai_nlls, list_lai_preds, j_dt_list, j_ndvi_list = get_juan_validation_metrics(PROSAIL_VAE, juan_data_dir_path, lai_min=0, dt_max=10, 
                                                                                 sites=sites, weiss_mode=weiss_mode)
        all_lai_preds = torch.cat(list_lai_preds)
        all_dt_list  = torch.cat(j_dt_list)
        all_ndvi = torch.cat(j_ndvi_list)
        for i, site in enumerate(sites):
            torch.save(j_list_lai_nlls[i].cpu(), juan_validation_dir + f"/{site}_lai_nll.pt")
            torch.save(list_lai_preds[i].cpu(), juan_validation_dir + f"/{site}_lai_ref_pred.pt")
            torch.save(j_dt_list[i].cpu(), juan_validation_dir + f"/{site}_dt.pt")
            if plot_results:
                fig, ax = plot_lai_preds(list_lai_preds[i][:,1].cpu(), list_lai_preds[i][:,0].cpu(), j_dt_list[i], site)
                fig.savefig(juan_validation_dir + f"/{site}_lai_pred_vs_true.png")
        if plot_results:
            fig, ax = plot_lai_preds(all_lai_preds[:,1].cpu(), all_lai_preds[:,0].cpu(), all_dt_list, "all")
            fig.savefig(juan_validation_dir + f"/all_lai_pred_vs_true.png")
            lai_err = all_lai_preds[:,1].cpu() - all_lai_preds[:,0].cpu()
            fig, ax = plot_lai_vs_ndvi(all_lai_preds[lai_err.abs() > 1,1].cpu(), all_ndvi[lai_err.abs() > 1].cpu(), all_dt_list[lai_err.abs() > 1], "all")
            fig.savefig(juan_validation_dir + f"/all_lai_true_vs_ndvi.png")
            lai_filter = torch.logical_not(torch.logical_and(all_lai_preds[:,1] < 0.5, all_ndvi > 0.4)).cpu()
            fig, ax = plot_lai_preds(all_lai_preds[lai_filter, 1].cpu(), all_lai_preds[lai_filter, 0].cpu(), all_dt_list[lai_filter], "all")
            fig.savefig(juan_validation_dir + f"/filtered_all_lai_pred_vs_true.png")
    if plot_results:
        plot_rec_hist2D(PROSAIL_VAE, loader, res_dir, nbin=50, bands_name=bands_name)
    (mae, mpiw, picp, mare, 
    sim_dist, tgt_dist, rec_dist,
    angles_dist, s2_r_dist,
    sim_pdfs, sim_supports, ae_percentiles, 
    are_percentiles, piw_percentiles) = get_metrics(PROSAIL_VAE, loader, 
                              n_pdf_sample_points=3001,
                              alpha_conf=alpha_pi)
    logger.info("Metrics computed.")

    save_metrics(res_dir, mae, mpiw, picp, alpha_pi, 
                ae_percentiles, are_percentiles, piw_percentiles)
    maer = pd.read_csv(res_dir+"/metrics/maer.csv").drop(columns=["Unnamed: 0"])
    mpiwr = pd.read_csv(res_dir+"/metrics/mpiwr.csv").drop(columns=["Unnamed: 0"])
    if plot_results:
        # Plotting results
        metrics_dir = res_dir + "/metrics_plot/"
        if not os.path.isdir(metrics_dir):
            os.makedirs(metrics_dir)
        
        logger.info("Plotting metrics.")
        
        plot_metrics(metrics_dir, alpha_pi, maer, mpiwr, picp, mare)
        plot_metric_boxplot(ae_percentiles, res_dir, metric_name='ae', logscale=True)
        plot_metric_boxplot(are_percentiles, res_dir, metric_name='are')
        # plot_metric_boxplot(piw_percentiles, res_dir, metric_name='piw')
        rec_dir = res_dir + "/reconstruction/"
        if not os.path.isdir(rec_dir):
            os.makedirs(rec_dir)
        logger.info("Plotting reconstructions")
        plot_rec_and_latent(PROSAIL_VAE, loader, rec_dir, n_plots=20, bands_name=bands_name)
        
        logger.info("Plotting PROSAIL parameter distributions")
        plot_param_dist(metrics_dir, sim_dist, tgt_dist)
        logger.info("Plotting PROSAIL parameters, reference vs prediction")
        plot_lat_hist2D(tgt_dist, sim_pdfs, sim_supports, res_dir, nbin=50)
        plot_pred_vs_tgt(metrics_dir, sim_dist, tgt_dist)
        ssimulator = PROSAIL_VAE.decoder.ssimulator
        refl_dist = loader.dataset[:][0]
        plot_refl_dist(rec_dist, refl_dist, res_dir, normalized=False, 
                    ssimulator=PROSAIL_VAE.decoder.ssimulator)
        
        normed_rec_dist =  (rec_dist.to(device) - ssimulator.norm_mean.to(device)) / ssimulator.norm_std.to(device) 
        normed_refl_dist =  (refl_dist.to(device) - ssimulator.norm_mean.to(device)) / ssimulator.norm_std.to(device) 
        logger.info("Plotting reflectance distribution")
        plot_refl_dist(normed_rec_dist, normed_refl_dist, metrics_dir, normalized=True, ssimulator=PROSAIL_VAE.decoder.ssimulator, bands_name=bands_name)
        logger.info("Plotting reconstructed reflectance components pair plots")
        pair_plot(normed_rec_dist, tensor_2=None, features = BANDS, 
                res_dir=metrics_dir, filename='normed_rec_pair_plot.png')
        logger.info("Plotting reference reflectance components pair plots")
        pair_plot(normed_refl_dist, tensor_2=None, features = BANDS, 
                res_dir=metrics_dir, filename='normed_s2bands_pair_plot.png')
        logger.info("Plotting inferred PROSAIL parameters pair plots")
        pair_plot(sim_dist.squeeze(), tensor_2=None, features = PROSAILVARS, 
                res_dir=metrics_dir, filename='sim_prosail_pair_plot.png')
        logger.info("Plotting reference PROSAIL parameters pair plots")
        pair_plot(tgt_dist.squeeze(), tensor_2=None, features = PROSAILVARS, 
                res_dir=metrics_dir, filename='ref_prosail_pair_plot.png')
        logger.info("Plotting reconstruction error against angles")
        plot_rec_error_vs_angles(s2_r_dist, rec_dist, angles_dist,  res_dir=metrics_dir)
    
    logger.info("Program completed.")
    return



def get_encoded_image(image_tensor, PROSAIL_VAE, patch_size=32, bands=torch.tensor([0,1,2,3,4,5,6,7,8,9]), mode='lat_mode'):
    hw = PROSAIL_VAE.encoder.nb_enc_cropped_hw
    patched_tensor = patchify(image_tensor, patch_size=patch_size, margin=hw)
    patched_sim_image = torch.zeros((patched_tensor.size(0), patched_tensor.size(1), 11, patch_size, patch_size)).to(PROSAIL_VAE.device)
    patched_rec_image = torch.zeros((patched_tensor.size(0), patched_tensor.size(1), len(bands), patch_size, patch_size)).to(PROSAIL_VAE.device)
    for i in range(patched_tensor.size(0)):
        for j in range(patched_tensor.size(1)):
            x = patched_tensor[i,j, bands, :, :]
            angles = torch.zeros(3, patch_size + 2 * hw, patch_size + 2 * hw).to(PROSAIL_VAE.device)
            angles[0,:,:] = patched_tensor[i, j, 11,:,:]
            angles[1,:,:] = patched_tensor[i, j, 13, :,:]
            angles[2,:,:] = patched_tensor[i, j, 12, :,:] - patched_tensor[i,j,14, :,:]
            with torch.no_grad():
                dist_params, z, sim, rec = PROSAIL_VAE.point_estimate_rec(x.unsqueeze(0), 
                                                                          angles.unsqueeze(0), mode=mode)
            patched_rec_image[i,j,:,:,:] = rec
            patched_sim_image[i,j,:,:,:] = sim
    sim_image = unpatchify(patched_sim_image)[:,:image_tensor.size(1),:image_tensor.size(2)][:,hw:-hw,hw:-hw]
    rec_image = unpatchify(patched_rec_image)[:,:image_tensor.size(1),:image_tensor.size(2)][:,hw:-hw,hw:-hw]
    cropped_image = image_tensor[:,hw:-hw,hw:-hw]
    return rec_image, sim_image, cropped_image


def check_fold_res_dir(fold_dir, n_xp, params):
    same_fold = ""
    all_dirs = os.listdir(fold_dir)
    for d in all_dirs:
        if d.startswith(f"{n_xp}_kfold_{params['k_fold']}_n_{params['n_fold']}") :
            same_fold = d
    return same_fold

def get_res_dir_path(root_results_dir, params, n_xp=None, overwrite_xp=False):
    
    if not os.path.exists(root_results_dir):
        os.makedirs(root_results_dir)
    if not os.path.exists(root_results_dir+"n_xp.json"):    
        save_dict({"xp":0}, root_results_dir+"n_xp.json")
    if n_xp is None:
        n_xp = load_dict(root_results_dir+"n_xp.json")['xp']+1
    save_dict({"xp":n_xp}, root_results_dir+"n_xp.json")
    date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    if params['k_fold']>1:
        k_fold_dir = f"{root_results_dir}/{n_xp}_kfold_{params['k_fold']}_supervised_{params['supervised']}_{params['dataset_file_prefix']}"
        if not params['supervised']:
            k_fold_dir + f"kl_{params['beta_kl']}"
        if not os.path.exists(k_fold_dir):
            os.makedirs(k_fold_dir)    
        res_dir = f"{k_fold_dir}/{n_xp}_kfold_{params['k_fold']}_n_{params['n_fold']}_d{date}_supervised_{params['supervised']}_{params['dataset_file_prefix']}"
        same_fold_dir = check_fold_res_dir(k_fold_dir, n_xp, params)
        if len(same_fold_dir)>0:
            if overwrite_xp:
                warnings.warn("WARNING: Overwriting existing fold experiment in 5s")
                sleep(5)
                shutil.rmtree(k_fold_dir + "/"+ same_fold_dir)
            else:
                raise ValueError(f"The same experiment (fold) has already been carried out at {same_fold_dir}.\n Please change the number of fold or allow overwrite")
    else:
        res_dir = f"{root_results_dir}/{n_xp}_d{date}_supervised_{params['supervised']}_{params['dataset_file_prefix']}"
    if not os.path.isdir(res_dir):
        os.makedirs(res_dir)    
    return res_dir

# def setupResults():
#     if socket.gethostname()=='CELL200973':
#         args=["-d", "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/",
#               "-r", "",
#               "-rsr", '/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/',
#               "-t", "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/real_data/torchfiles/"]
        
#         parser = get_prosailvae_results_parser().parse_args(args)    
#     else:
#         parser = get_prosailvae_results_parser().parse_args()
#     root_dir = os.path.join(os.path.dirname(prosailvae.__file__), os.pardir)

#     if len(parser.data_dir)==0:
#         data_dir = os.path.join(root_dir,"data/")
#     else:
#         data_dir = parser.data_dir

#     if len(parser.root_results_dir)==0:
#         res_dir = os.path.join(os.path.join(os.path.dirname(prosailvae.__file__),
#                                                      os.pardir),"results/")
#     else:
#         res_dir = parser.root_results_dir    
#     params = load_dict(res_dir + "/config.json")
#     if params["supervised"]:
#         params["simulated_dataset"]=True
#     params["n_fold"] = parser.n_fold if params["k_fold"] > 1 else None

#     params_sup_kl_model = None
#     if params["supervised_kl"]:
#         params_sup_kl_model = load_dict(res_dir+"/sup_kl_model_config.json")
#         params_sup_kl_model['sup_model_weights_path'] = res_dir+"/sup_kl_model_weights.tar"
    
#     logging.basicConfig(filename=res_dir+'/results_log.log', 
#                               level=logging.INFO, force=True)
#     logger_name = LOGGER_NAME
#     # create logger
#     logger = logging.getLogger(logger_name)
#     logger.info('Starting computation of results of PROSAIL-VAE.')
#     logger.info('========================================================================')
#     logger.info('Parameters are : ')
#     for _, key in enumerate(params):
#         logger.info(f'{key} : {params[key]}')
#     logger.info('========================================================================')

#     return params, parser, res_dir, data_dir, params_sup_kl_model
    
def configureEmissionTracker(parser):
    logger = logging.getLogger(LOGGER_NAME)
    try:
        from codecarbon import OfflineEmissionsTracker
        tracker = OfflineEmissionsTracker(country_iso_code="FRA", output_dir=parser.root_results_dir)
        tracker.start()
        useEmissionTracker = True
    except:
        logger.error("Couldn't start codecarbon ! Emissions not tracked for this execution.")
        useEmissionTracker = False
        tracker = None
    return tracker, useEmissionTracker

# def main():
#     params, parser, res_dir, data_dir, params_sup_kl_model = setupResults()
#     tracker, useEmissionTracker = configureEmissionTracker(parser)
#     try:
#         vae_file_path = res_dir + '/prosailvae_weights.tar'
#         PROSAIL_VAE = load_prosail_vae_with_hyperprior(params, parser.rsr_dir, data_dir, 
#                                 logger_name=LOGGER_NAME, vae_file_path=vae_file_path, params_sup_kl_model=params_sup_kl_model)
#         save_results(PROSAIL_VAE, res_dir, data_dir, LOGGER_NAME=LOGGER_NAME, plot_results=parser.plot_results)
#     except Exception as e:
#         traceback.print_exc()
#         print(e)
#     if useEmissionTracker:
#         tracker.stop()
#     pass
#     pass

# if __name__ == "__main__":
#     main()