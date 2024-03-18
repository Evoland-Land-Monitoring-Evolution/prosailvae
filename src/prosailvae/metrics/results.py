import argparse
import logging
import os
import shutil
import socket
import warnings
from datetime import datetime
from pathlib import Path
from time import sleep

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tikzplotlib
import torch
from torchutils.patches import patchify, unpatchify

from article_plots.belsar_plots import (
    get_belsar_lai_vs_hspot,
    get_belsar_sites_time_series,
)

from ..bvnet_regression.bvnet_utils import get_bvnet_biophyiscal_from_batch
from ..dataset.loaders import get_simloader
from ..ProsailSimus import BANDS, PROSAILVARS
from ..utils.image_utils import crop_s2_input, get_encoded_image_from_batch
from ..utils.utils import load_dict, save_dict
from ..validation.validation import (
    get_all_campaign_CCC_results_BVNET,
    get_all_campaign_lai_results,
    get_all_campaign_lai_results_BVNET,
    get_belsar_x_frm4veg_lai_results,
    get_frm4veg_ccc_results,
    get_validation_global_metrics,
)
from .metrics_utils import (
    get_bvnet_validation_metrics,
    get_juan_validation_metrics,
    get_metrics,
    save_metrics,
)
from .prosail_plots import (
    PROSAIL_2D_aggregated_results,
    PROSAIL_2D_article_plots,
    PROSAIL_2D_res_plots,
    all_loss_curve,
    article_2D_aggregated_results,
    loss_curve,
    pair_plot,
    plot_lai_preds,
    plot_lai_vs_ndvi,
    plot_lat_hist2D,
    plot_metric_boxplot,
    plot_metrics,
    plot_param_dist,
    plot_pred_vs_tgt,
    plot_rec_and_latent,
    plot_rec_error_vs_angles,
    plot_rec_hist2D,
    plot_refl_dist,
    plot_single_lat_hist_2D,
    regression_plot,
    regression_plot_2hues,
)

LOGGER_NAME = "PROSAIL-VAE results logger"
PC_SOCKET_NAME = "CELL200973"  # toggle options for dev and debug on PC
TOP_PATH = ""


def tikzplotlib_fix_ncols(obj):
    """workaround for matplotlib 3.6 renamed legend's _ncol to _ncols,
    which breaks tikzplotlib

    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)


def get_prosailvae_results_parser():
    """
    Creates a new argument parser.
    """
    parser = argparse.ArgumentParser(description="Parser for data generation")

    parser.add_argument(
        "-c",
        dest="config_file",
        help="name of config json file on config directory.",
        type=str,
        default="config.json",
    )

    parser.add_argument(
        "-d",
        dest="data_dir",
        help="path to data direcotry",
        type=str,
        default="/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/",
    )

    parser.add_argument(
        "-r",
        dest="root_results_dir",
        help="path to root results direcotry",
        type=str,
        default="",
    )

    parser.add_argument(
        "-rsr",
        dest="rsr_dir",
        help="directory of rsr_file",
        type=str,
        default="/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/",
    )

    parser.add_argument(
        "-t",
        dest="tensor_dir",
        help="directory of mmdc tensor files",
        type=str,
        default="/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/"
        "data/real_data/torchfiles/",
    )

    parser.add_argument(
        "-p",
        dest="plot_results",
        help="toggle results plotting",
        type=bool,
        default=False,
    )
    return parser


def save_validation_results(
    model,
    res_dir,
    frm4veg_data_dir,
    frm4veg_2021_data_dir,
    belsar_data_dir,
    model_name="pvae",
    method="simple_interpolate",
    mode="sim_tg_mean",
    save_reconstruction=True,
    remove_files=False,
    plot_results=False,
):
    (
        barrax_results,
        barrax_2021_results,
        wytham_results,
        belsar_results,
        all_belsar,
    ) = get_all_campaign_lai_results(
        model,
        frm4veg_data_dir,
        frm4veg_2021_data_dir,
        belsar_data_dir,
        res_dir,
        mode=mode,
        method=method,
        model_name=model_name,
        save_reconstruction=save_reconstruction,
        get_all_belsar=plot_results,
        remove_files=remove_files,
    )
    results = {}
    results["lai"] = get_belsar_x_frm4veg_lai_results(
        belsar_results,
        barrax_results,
        barrax_2021_results,
        wytham_results,
        frm4veg_lai="lai",
        get_reconstruction_error=save_reconstruction,
        bands_idx=model.encoder.bands,
    )
    hue_elem = pd.unique(results["lai"]["Land cover"])
    hue2_elem = pd.unique(results["lai"]["Campaign"])
    hue_color_dict = {}
    for j, h_e in enumerate(hue_elem):
        hue_color_dict[h_e] = f"C{j}"
    default_markers = ["o", "v", "D", "s", "+", ".", "^", "1"]
    hue2_markers_dict = {}
    for j, h2_e in enumerate(hue2_elem):
        hue2_markers_dict[h2_e] = default_markers[j]

    results["ccc"] = get_frm4veg_ccc_results(
        barrax_results,
        barrax_2021_results,
        wytham_results,
        frm4veg_ccc="ccc",
        get_reconstruction_error=save_reconstruction,
        bands_idx=model.encoder.bands,
    )
    lai_dir = os.path.join(res_dir, "lai_scatter")
    if not Path(lai_dir).exists():
        os.makedirs(lai_dir)
    ccc_dir = os.path.join(res_dir, "ccc_scatter")
    if not Path(ccc_dir).exists:
        os.makedirs(ccc_dir)
    scatter_dir = {"lai": lai_dir, "ccc": ccc_dir}
    global_metrics = {}

    for variable in ["lai", "ccc"]:
        (
            global_rmse_dict,
            global_picp_dict,
            global_mpiw_dict,
            global_mestdr_dict,
        ) = get_validation_global_metrics(
            results[variable],
            decompose_along_columns=["Campaign"],  # ["Site", "Land cover", "Campaign"],
            n_sigma=3,
            variable=variable,
        )
        global_metrics[variable] = {
            "rmse": global_rmse_dict["Campaign"],
            "picp": global_picp_dict["Campaign"],
            "mpiw": global_mpiw_dict["Campaign"],
            "mestdr": global_mestdr_dict["Campaign"],
        }
        if not os.path.exists(scatter_dir[variable]):
            os.mkdir(scatter_dir[variable])
        for key, rmse_df in global_rmse_dict.items():
            rmse_df.to_csv(
                os.path.join(
                    scatter_dir[variable],
                    f"{model_name}_{key}_{variable}_validation_rmse.csv",
                )
            )
        for key, pcip_df in global_picp_dict.items():
            pcip_df.to_csv(
                os.path.join(
                    scatter_dir[variable],
                    f"{model_name}_{key}_{variable}_validation_picp.csv",
                )
            )
        for key, mpiw_df in global_mpiw_dict.items():
            mpiw_df.to_csv(
                os.path.join(
                    scatter_dir[variable],
                    f"{model_name}_{key}_{variable}_validation_mpiw.csv",
                )
            )
        for key, mestdr_df in global_mestdr_dict.items():
            mestdr_df.to_csv(
                os.path.join(
                    scatter_dir[variable],
                    f"{model_name}_{key}_{variable}_validation_mestdr.csv",
                )
            )

    df_results_bvnet = {}
    if plot_results:
        (
            barrax_results_bvnet,
            barrax_2021_results_bvnet,
            wytham_results_bvnet,
            belsar_results_bvnet,
            all_belsar_bvnet,
        ) = get_all_campaign_lai_results_BVNET(
            frm4veg_data_dir,
            frm4veg_2021_data_dir,
            belsar_data_dir,
            res_dir,
            method=method,
            get_all_belsar=True,
            remove_files=remove_files,
        )

        df_results_bvnet["lai"] = get_belsar_x_frm4veg_lai_results(
            belsar_results_bvnet,
            barrax_results_bvnet,
            barrax_2021_results_bvnet,
            wytham_results_bvnet,
            frm4veg_lai="lai",
            get_reconstruction_error=False,
        )

        (
            barrax_results_bvnet,
            barrax_2021_results_bvnet,
            wytham_results_bvnet,
        ) = get_all_campaign_CCC_results_BVNET(
            frm4veg_data_dir, frm4veg_2021_data_dir, ccc_bvnet=None, cab_mode=False
        )
        df_results_bvnet["ccc"] = get_frm4veg_ccc_results(
            barrax_results_bvnet,
            barrax_2021_results_bvnet,
            wytham_results_bvnet,
            frm4veg_ccc="ccc",
            get_reconstruction_error=False,
        )

        time_series_dir = os.path.join(res_dir, "time_series")
        if not Path(time_series_dir).exists():
            os.makedirs(time_series_dir)
        fig, axs = plt.subplots(
            10, 1, dpi=150, sharex=True, tight_layout=True, figsize=(10, 2 * 10)
        )
        for i in range(0, 10):
            site = "W" + str(i + 1)
            fig, ax = get_belsar_sites_time_series(
                all_belsar,
                belsar_data_dir,
                site=site,
                fig=fig,
                ax=axs[i],
                label="PROSAIL-VAE",
                use_ref_metrics=True,
            )
            fig, ax = get_belsar_sites_time_series(
                all_belsar_bvnet,
                belsar_data_dir,
                site=site,
                fig=fig,
                ax=axs[i],
                label="SNAP",
            )
            ax.legend()
        axs[-1].set_ylabel("Date")
        fig.savefig(
            os.path.join(
                time_series_dir, f"{model_name}_belSAR_LAI_time_series_Wheat.png"
            )
        )
        for i in range(0, 10):
            fig, ax = plt.subplots(dpi=150, tight_layout=True, figsize=(1, 2))
            site = "W" + str(i + 1)
            fig, ax = get_belsar_sites_time_series(
                all_belsar,
                belsar_data_dir,
                site=site,
                fig=fig,
                ax=ax,
                label="PROSAIL-VAE",
                use_ref_metrics=True,
            )
            fig, ax = get_belsar_sites_time_series(
                all_belsar_bvnet,
                belsar_data_dir,
                site=site,
                fig=fig,
                ax=ax,
                label="SNAP",
            )
            ax.legend()

            tikzplotlib_fix_ncols(fig)
            tikzplotlib.save(
                os.path.join(time_series_dir, f"belSAR_LAI_time_series_Wheat_{i}.tex")
            )
            fig, ax = plt.subplots(dpi=150, tight_layout=True, figsize=(1, 2))
            site = "M" + str(i + 1)
            fig, ax = get_belsar_sites_time_series(
                all_belsar,
                belsar_data_dir,
                site=site,
                fig=fig,
                ax=ax,
                label="PROSAIL-VAE",
                use_ref_metrics=True,
            )
            fig, ax = get_belsar_sites_time_series(
                all_belsar_bvnet,
                belsar_data_dir,
                site=site,
                fig=fig,
                ax=ax,
                label="SNAP",
            )
            ax.legend()

            tikzplotlib_fix_ncols(fig)
            tikzplotlib.save(
                os.path.join(time_series_dir, f"belSAR_LAI_time_series_Maize_{i}.tex")
            )
            plt.close("all")

        fig, axs = plt.subplots(
            10, 1, dpi=150, sharex=True, tight_layout=True, figsize=(10, 2 * 10)
        )
        for i in range(0, 10):
            site = "M" + str(i + 1)
            fig, ax = get_belsar_sites_time_series(
                all_belsar,
                belsar_data_dir,
                site=site,
                fig=fig,
                ax=axs[i],
                label="PROSAIL-VAE",
                use_ref_metrics=True,
            )
            fig, ax = get_belsar_sites_time_series(
                all_belsar_bvnet,
                belsar_data_dir,
                site=site,
                fig=fig,
                ax=axs[i],
                label="SNAP",
            )

        fig.savefig(
            os.path.join(
                time_series_dir, f"{model_name}_belSAR_LAI_time_series_Maize.png"
            )
        )

        hspot_dir = os.path.join(res_dir, "hspot_vs_lai")
        os.makedirs(hspot_dir)

        fig, ax = get_belsar_lai_vs_hspot(
            all_belsar,
            belsar_data_dir,
            sites=[f"W{i+1}" for i in range(10)],
            fig=None,
            ax=None,
            label="",
        )
        fig.savefig(
            os.path.join(hspot_dir, f"{model_name}_belSAR_LAI_vs_hspot_Wheat.png")
        )
        tikzplotlib_fix_ncols(fig)
        tikzplotlib.save(
            os.path.join(hspot_dir, f"{model_name}_belSAR_LAI_vs_hspot_Wheat.tex")
        )

        fig, ax = get_belsar_lai_vs_hspot(
            all_belsar,
            belsar_data_dir,
            sites=[f"M{i+1}" for i in range(10)],
            fig=None,
            ax=None,
            label="",
        )
        fig.savefig(
            os.path.join(hspot_dir, f"{model_name}_belSAR_LAI_vs_hspot_Maize.png")
        )
        tikzplotlib_fix_ncols(fig)
        tikzplotlib.save(
            os.path.join(hspot_dir, f"{model_name}_belSAR_LAI_vs_hspot_Maize.tex")
        )

        for variable in ["lai", "ccc"]:
            fig, axs = plt.subplots(1, 2, figsize=(14, 7), dpi=150)
            _, _ = regression_plot_2hues(
                df_results_bvnet[variable],
                x=f"{variable}",
                y=f"Predicted {variable}",
                hue="Land cover",
                hue2="Campaign",
                display_text=False,
                error_x=f"{variable} std",
                error_y=None,
                hue_perfs=False,
                title_hue="Land cover",
                title_hue2="\n Site",
                hue_color_dict=hue_color_dict,
                hue2_markers_dict=hue2_markers_dict,
                legend_col=0,
                fig=fig,
                ax=axs[0],
            )

            _, _ = regression_plot_2hues(
                results[variable],
                x=f"{variable}",
                y=f"Predicted {variable}",
                hue="Land cover",
                hue2="Campaign",
                display_text=False,
                legend_col=1,
                error_x=f"{variable} std",
                error_y=f"Predicted {variable} std",
                hue_perfs=False,
                title_hue="Land cover",
                title_hue2="\n Site",
                hue_color_dict=hue_color_dict,
                hue2_markers_dict=hue2_markers_dict,
                fig=fig,
                ax=axs[1],
            )
            fig.savefig(
                os.path.join(
                    scatter_dir[variable],
                    f"{model_name}_{variable}_pvae_vs_"
                    "snap_regression_campaign_land_cover.png",
                )
            )
            tikzplotlib_fix_ncols(fig)
            tikzplotlib.save(
                os.path.join(
                    scatter_dir[variable],
                    f"{model_name}_{variable}_pvae_vs_"
                    "snap_regression_campaign_land_cover.tex",
                )
            )

            results[variable][f"{variable} error"] = (
                results[variable][f"Predicted {variable}"]
                - results[variable][f"{variable}"]
            )
            results[variable].to_csv(
                os.path.join(
                    scatter_dir[variable],
                    f"{model_name}_all_campaigns_{variable}_{mode}_{method}.csv",
                )
            )

            fig, ax = regression_plot(
                results[variable],
                x=f"{variable}",
                y=f"Predicted {variable}",
                fig=None,
                ax=None,
                hue="Site",
                legend_col=3,
                error_x=f"{variable} std",
                error_y=f"Predicted {variable} std",
                hue_perfs=True,
            )
            fig.savefig(
                os.path.join(
                    scatter_dir[variable],
                    f"{model_name}_{variable}_regression_sites.png",
                )
            )
            tikzplotlib_fix_ncols(fig)
            tikzplotlib.save(
                os.path.join(
                    scatter_dir[variable],
                    f"{model_name}_{variable}_regression_sites.tex",
                )
            )

            fig, ax = regression_plot(
                results[variable],
                x=f"{variable}",
                y=f"Predicted {variable}",
                fig=None,
                ax=None,
                hue="Campaign",
                legend_col=2,
                error_x=f"{variable} std",
                error_y=f"Predicted {variable} std",
                hue_perfs=True,
            )
            fig.savefig(
                os.path.join(
                    scatter_dir[variable],
                    f"{model_name}_{variable}_regression_campaign.png",
                )
            )
            tikzplotlib_fix_ncols(fig)
            tikzplotlib.save(
                os.path.join(
                    scatter_dir[variable],
                    f"{model_name}_{variable}_regression_campaign.tex",
                )
            )

            fig, ax = regression_plot(
                results[variable],
                x=f"{variable}",
                y=f"Predicted {variable}",
                fig=None,
                ax=None,
                hue="Land cover",
                legend_col=3,
                error_x=f"{variable} std",
                error_y=f"Predicted {variable} std",
                hue_perfs=False,
            )
            fig.savefig(
                os.path.join(
                    scatter_dir[variable],
                    f"{model_name}_{variable}_regression_land_cover.png",
                )
            )
            tikzplotlib_fix_ncols(fig)
            tikzplotlib.save(
                os.path.join(
                    scatter_dir[variable],
                    f"{model_name}_{variable}_regression_land_cover.tex",
                )
            )

            fig, ax = plt.subplots(dpi=150)
            sns.scatterplot(
                data=results[variable],
                x=f"{variable} error",
                y="Reconstruction error",
                hue="Site",
                ax=ax,
            )
            fig.savefig(
                os.path.join(
                    scatter_dir[variable],
                    f"{model_name}_{variable}_error_vs_reconstruction_error.png",
                )
            )
            tikzplotlib_fix_ncols(fig)
            tikzplotlib.save(
                os.path.join(
                    scatter_dir[variable],
                    f"{model_name}_{variable}_error_vs_reconstruction_error.tex",
                )
            )

            n_cols = len(np.array(BANDS)[model.encoder.bands.cpu()]) // 2
            fig, axs = plt.subplots(2, n_cols, dpi=150, figsize=(n_cols * 3, 2 * 3))
            for i, band in enumerate(
                np.array(BANDS)[model.encoder.bands.cpu()].tolist()
            ):
                row = i // n_cols
                col = i % n_cols
                sns.scatterplot(
                    data=results[variable],
                    x=f"{variable} error",
                    y=f"{band} error",
                    hue="Campaign",
                    ax=axs[row, col],
                    s=5,
                )
            fig.savefig(
                os.path.join(
                    scatter_dir[variable],
                    f"{model_name}_{variable}_error_vs_band_rec_error_Campaign.png",
                )
            )

            fig, axs = plt.subplots(2, n_cols, dpi=150, figsize=(n_cols * 3, 2 * 3))
            for i, band in enumerate(
                np.array(BANDS)[model.encoder.bands.cpu()].tolist()
            ):
                row = i // n_cols
                col = i % n_cols
                sns.scatterplot(
                    data=results[variable],
                    x=f"{variable} error",
                    y=f"{band} error",
                    ax=axs[row, col],
                    s=5,
                )
            fig.savefig(
                os.path.join(
                    scatter_dir[variable],
                    f"{model_name}_{variable}_error_vs_band_rec_error.png",
                )
            )
            fig, ax = plt.subplots(dpi=150)
            sns.scatterplot(
                data=results[variable],
                x=f"{variable} error",
                y="Time delta",
                hue="Campaign",
                ax=ax,
            )
            fig.savefig(
                os.path.join(
                    scatter_dir[variable],
                    f"{model_name}_{variable}_error_vs_dt_boxplot_campaign.png",
                )
            )

            fig, ax = plt.subplots(dpi=150)
            sns.scatterplot(
                data=results[variable],
                x=f"{variable} error",
                y=f"Predicted {variable} std",
                hue="Campaign",
                ax=ax,
            )
            fig.savefig(
                os.path.join(
                    scatter_dir[variable],
                    f"{model_name}_{variable}_error_vs_sigma_campaign.png",
                )
            )

            fig, ax = plt.subplots(dpi=150)
            sns.scatterplot(
                data=results[variable],
                x=f"{variable} error",
                y="Reconstruction error",
                hue="Land cover",
                ax=ax,
            )
            fig.savefig(
                os.path.join(
                    scatter_dir[variable],
                    f"{model_name}_{variable}_error_vs_"
                    "reconstruction_error_land_cover.png",
                )
            )

            fig, ax = plt.subplots(dpi=150)
            sns.scatterplot(
                data=results[variable],
                x=f"{variable} error",
                y="Reconstruction error",
                hue="Time delta",
                ax=ax,
            )
            fig.savefig(
                os.path.join(
                    scatter_dir[variable],
                    f"{model_name}_{variable}_error_vs_reconstruction_error_dt.png",
                )
            )

            fig, ax = plt.subplots(dpi=150)
            sns.scatterplot(
                data=results[variable],
                x=f"{variable} error",
                y="Reconstruction error",
                hue="Site",
                ax=ax,
            )
            fig.savefig(
                os.path.join(
                    scatter_dir[variable],
                    f"{model_name}_{variable}_error_vs_reconstruction_error_site.png",
                )
            )

            fig, ax = plt.subplots(dpi=150)
            sns.scatterplot(
                data=results[variable],
                x=f"{variable}",
                y=f"{variable} error",
                hue="Campaign",
                ax=ax,
            )
            fig.savefig(
                os.path.join(
                    scatter_dir[variable],
                    f"{model_name}_{variable}_error_vs_{variable}_campaign.png",
                )
            )

            fig, ax = plt.subplots(dpi=150)
            sns.scatterplot(
                data=results[variable],
                x=f"{variable} std",
                y=f"Predicted {variable} std",
                hue="Campaign",
                ax=ax,
            )
            fig.savefig(
                os.path.join(
                    scatter_dir[variable],
                    f"{model_name}_{variable}_std_vs_{variable}_pred_std_campaign.png",
                )
            )

            fig, ax = plt.subplots(dpi=150)
            sns.scatterplot(
                data=results[variable],
                x=f"{variable} error",
                y="Reconstruction error",
                hue="Campaign",
                ax=ax,
            )
            fig.savefig(
                os.path.join(
                    scatter_dir[variable],
                    f"{model_name}_{variable}_error_vs_"
                    "reconstruction_error_Campaign.png",
                )
            )
            plt.close("all")

            fig, ax = plt.subplots(dpi=150)
            ax.scatter(
                results[variable][f"Predicted {variable}"],
                df_results_bvnet[variable][f"Predicted {variable}"],
                c="k",
                s=1,
            )
            tikzplotlib_fix_ncols(fig)
            tikzplotlib.save(
                os.path.join(
                    scatter_dir[variable],
                    f"{model_name}_vs_sl2p_{variable}_scatterplot.tex",
                )
            )
    return global_metrics


def get_rec_var(
    PROSAIL_VAE,
    loader,
    max_batch=50,
    n_samples=10,
    sample_dim=1,
    bands_dim=2,
    n_bands=10,
):
    with torch.no_grad():
        all_rec_var = []
        for i, batch in enumerate(loader):
            if i == max_batch:
                break
            s2_r = patchify(batch[0].squeeze(0), patch_size=32, margin=0).to(
                PROSAIL_VAE.device
            )
            s2_r = s2_r.reshape(-1, *s2_r.shape[2:])
            s2_a = patchify(batch[1].squeeze(0), patch_size=32, margin=0).to(
                PROSAIL_VAE.device
            )
            s2_a = s2_a.reshape(-1, *s2_a.shape[2:])
            for j in range(s2_a.size(0)):
                _, _, _, rec = PROSAIL_VAE.forward(
                    s2_r[j, ...].unsqueeze(0),
                    n_samples=n_samples,
                    angles=s2_a[j, ...].unsqueeze(0),
                )
                rec_var = rec.var(sample_dim)
                rec_var = rec_var.transpose(bands_dim, 0).reshape(n_bands, -1)
            all_rec_var.append(rec_var.cpu())
    return torch.cat(all_rec_var, 1)


def plot_losses(
    res_dir,
    all_train_loss_df=None,
    all_valid_loss_df=None,
    info_df=None,
    LOGGER_NAME="PROSAIL-VAE logger",
    plot_results=False,
):
    logger = logging.getLogger(LOGGER_NAME)
    logger.info("Saving Loss")
    # Saving Loss
    loss_dir = res_dir + "/loss/"
    if not os.path.isdir(loss_dir):
        os.makedirs(loss_dir)

    if all_train_loss_df is not None:
        all_train_loss_df.to_csv(loss_dir + "train_loss.csv")
        if plot_results:
            loss_curve(all_train_loss_df, save_file=loss_dir + "train_loss.svg")
            loss_curve(
                all_train_loss_df[["epoch", "loss_sum"]],
                save_file=loss_dir + "train_loss_sum.svg",
            )
    if all_valid_loss_df is not None:
        all_valid_loss_df.to_csv(loss_dir + "valid_loss.csv")
        if plot_results:
            loss_curve(all_valid_loss_df, save_file=loss_dir + "valid_loss.svg")
            loss_curve(
                all_valid_loss_df[["epoch", "loss_sum"]],
                save_file=loss_dir + "train_loss_sum.svg",
            )
    if info_df is not None:
        if plot_results:
            loss_curve(info_df, save_file=loss_dir + "lr.svg")
            all_loss_curve(
                all_train_loss_df[["epoch", "loss_sum"]],
                all_valid_loss_df[["epoch", "loss_sum"]],
                info_df,
                save_file=loss_dir + "all_loss_sum.svg",
            )
            all_loss_curve(
                all_train_loss_df,
                all_valid_loss_df,
                info_df,
                save_file=loss_dir + "all_loss.svg",
            )


def save_results_on_s2_data(
    PROSAIL_VAE,
    loader,
    res_dir,
    LOGGER_NAME="PROSAIL-VAE logger",
    plot_results=False,
    info_test_data=None,
    max_test_patch=50,
    lai_cyclical_loader=None,
):
    rec_mode = "lat_mode"
    logger = logging.getLogger(LOGGER_NAME)
    # Computing metrics
    PROSAIL_VAE.eval()

    # logger.info("Computing inference metrics with test dataset...")
    # test_loss = PROSAIL_VAE.validate(loader, mmdc_dataset=True, n_samples=10)
    # pd.DataFrame(test_loss, index=[0]).to_csv(loss_dir + "/test_loss.csv")
    if not plot_results:
        return

    plot_dir = res_dir + "/plots/"
    if not os.path.isdir(plot_dir):
        os.makedirs(plot_dir)

    all_rec = []
    all_lai = []
    all_cab = []
    all_cw = []
    all_vars = []
    all_vars_hyper = []
    all_std_hyper = []
    all_bvnet_lai = []
    all_bvnet_cab = []
    all_bvnet_cw = []
    all_s2_r = []
    all_std = []
    cyclical_ref_lai = []
    cyclical_lai = []
    cyclical_lai_std = []
    hw = PROSAIL_VAE.encoder.nb_enc_cropped_hw

    with torch.no_grad():
        for i, batch in enumerate(loader):
            (
                rec_image,
                sim_image,
                cropped_s2_r,
                cropped_s2_a,
                std_image,
            ) = get_encoded_image_from_batch(
                batch,
                PROSAIL_VAE,
                patch_size=32,
                bands=torch.arange(10),
                mode=rec_mode,
                no_rec=False,
            )
            if PROSAIL_VAE.hyper_prior is not None:
                (
                    _,
                    sim_image_hyper,
                    _,
                    _,
                    std_image_hyper,
                ) = get_encoded_image_from_batch(
                    batch,
                    PROSAIL_VAE.hyper_prior,
                    patch_size=32,
                    bands=torch.arange(10),
                    mode=rec_mode,
                    no_rec=True,
                )
                sim_image_hyper = crop_s2_input(sim_image_hyper, hw)
                std_image_hyper = crop_s2_input(std_image_hyper, hw)
                all_vars_hyper.append(sim_image_hyper.reshape(11, -1))
                all_std_hyper.append(std_image_hyper.reshape(11, -1))
            (
                _,
                cyclical_sim_image,
                cyclical_cropped_s2_r,
                cyclical_cropped_s2_a,
                cyclical_std_image,
            ) = get_encoded_image_from_batch(
                (rec_image.unsqueeze(0), cropped_s2_a),
                PROSAIL_VAE,
                patch_size=32,
                bands=torch.arange(10),
                mode=rec_mode,
                no_rec=True,
            )

            cyclical_ref_lai.append(crop_s2_input(sim_image, hw)[6, ...].reshape(-1))
            cyclical_lai.append(cyclical_sim_image[6, ...].reshape(-1))
            cyclical_lai_std.append(cyclical_std_image[6, ...].reshape(-1))
            info = info_test_data[i, :]
            (bvnet_lai, bvnet_cab, bvnet_cw) = get_bvnet_biophyiscal_from_batch(
                (cropped_s2_r, cropped_s2_a), patch_size=32, sensor=info[0]
            )

            patch_plot_dir = plot_dir + f"/{i}_{info[1]}_{info[2]}_{info[3]}/"
            if not os.path.isdir(patch_plot_dir):
                os.makedirs(patch_plot_dir)
            PROSAIL_2D_article_plots(
                patch_plot_dir,
                sim_image,
                cropped_s2_r.squeeze(),
                rec_image,
                bvnet_lai,
                bvnet_cab,
                bvnet_cw,
                std_image,
                i,
                info=info,
            )
            PROSAIL_2D_res_plots(
                patch_plot_dir,
                sim_image,
                cropped_s2_r.squeeze(),
                rec_image,
                bvnet_lai,
                bvnet_cab,
                bvnet_cw,
                std_image,
                i,
                info=info,
                var_bounds=PROSAIL_VAE.sim_space.var_bounds,
            )
            all_rec.append(rec_image.reshape(10, -1))
            all_lai.append(sim_image[6, ...].reshape(-1))
            all_cab.append(sim_image[1, ...].reshape(-1))
            all_cw.append(sim_image[4, ...].reshape(-1))
            all_vars.append(sim_image.reshape(11, -1))
            if not socket.gethostname() == PC_SOCKET_NAME:
                pair_plot(
                    sim_image.reshape(11, -1).squeeze().permute(1, 0),
                    tensor_2=None,
                    features=PROSAILVARS,
                    res_dir=patch_plot_dir,
                    filename="sim_prosail_pair_plot.png",
                )
            all_bvnet_lai.append(bvnet_lai.reshape(-1))
            all_bvnet_cab.append(bvnet_cab.reshape(-1))
            all_bvnet_cw.append(bvnet_cw.reshape(-1))
            all_s2_r.append(cropped_s2_r.reshape(10, -1))
            all_std.append(std_image.reshape(11, -1))

            if i == max_test_patch - 1:
                break

        if PROSAIL_VAE.hyper_prior is not None:
            all_vars_hyper = torch.cat(all_vars_hyper, axis=1)
            all_std_hyper = torch.cat(all_std_hyper, axis=1)
        all_rec = torch.cat(all_rec, axis=1)
        all_lai = torch.cat(all_lai)
        all_cab = torch.cat(all_cab)
        all_ccc = all_lai * all_cab

        all_cw = torch.cat(all_cw)
        all_vars = torch.cat(all_vars, axis=1)
        all_cw_rel = 1 - all_vars[5, ...] / all_cw
        all_bvnet_lai = torch.cat(all_bvnet_lai)
        all_bvnet_cab = torch.cat(all_bvnet_cab)
        all_bvnet_cw = torch.cat(all_bvnet_cw)
        all_s2_r = torch.cat(all_s2_r, axis=1)
        all_std = torch.cat(all_std, axis=1)
        cyclical_ref_lai = torch.cat(cyclical_ref_lai)
        cyclical_lai = torch.cat(cyclical_lai)
        cyclical_lai_std = torch.cat(cyclical_lai_std)

        article_2D_aggregated_results(
            plot_dir,
            all_s2_r,
            all_rec,
            all_lai,
            all_cab,
            all_cw,
            all_vars,
            all_bvnet_lai,
            all_bvnet_cab,
            all_bvnet_cw,
            all_std,
            all_ccc,
            all_cw_rel,
            cyclical_ref_lai,
            cyclical_lai,
            cyclical_lai_std,
            var_bounds=PROSAIL_VAE.sim_space.var_bounds,
        )

        PROSAIL_2D_aggregated_results(
            plot_dir,
            all_s2_r,
            all_rec,
            all_lai,
            all_cab,
            all_cw,
            all_vars,
            all_bvnet_lai,
            all_bvnet_cab,
            all_bvnet_cw,
            all_std,
            all_ccc,
            all_cw_rel,
            cyclical_ref_lai,
            cyclical_lai,
            cyclical_lai_std,
            all_vars_hyper=all_vars_hyper,
            all_std_hyper=all_std_hyper,
            #   gdf_lai, lai_validation_pred, bvnet_validation_lai
            var_bounds=PROSAIL_VAE.sim_space.var_bounds,
        )

    logger.info("Metrics computed.")
    rec_var = get_rec_var(
        PROSAIL_VAE,
        loader,
        max_batch=10,
        n_samples=10,
        sample_dim=1,
        bands_dim=2,
        n_bands=10,
    )
    n_col = 5
    fig, ax = plt.subplots(
        10 // n_col,
        n_col,
        dpi=150,
        sharex=True,
        sharey=True,
        figsize=(5 * n_col, 2 * (10 // n_col)),
    )
    for i in range(10):
        col = i % n_col
        row = i // n_col
        ax[row, col].hist(torch.log10(rec_var)[i, :], bins=100, density=True)
        ax[row, col].set_title(BANDS[i])
        ax[-1, col].set_xlabel("log10 rec. variance")
    fig.savefig(os.path.join(plot_dir, "rec_var.png"))
    return


def save_results_on_sim_data(
    PROSAIL_VAE,
    res_dir,
    data_dir,
    all_train_loss_df=None,
    all_valid_loss_df=None,
    info_df=None,
    LOGGER_NAME="PROSAIL-VAE logger",
    plot_results=False,
    juan_validation=True,
    bvnet_mode=False,
    n_samples=1,
    lai_cyclical_loader=None,
):
    bands_name = np.array(BANDS)[PROSAIL_VAE.encoder.bands.cpu()].tolist()
    # if bvnet_mode:
    #     bands_name = ["B03", "B04", "B05", "B06", "B07", "B8A", "B11", "B12"]
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
            loss_curve(all_train_loss_df, save_file=loss_dir + "train_loss.svg")
    if all_valid_loss_df is not None:
        all_valid_loss_df.to_csv(loss_dir + "valid_loss.csv")
        if plot_results:
            loss_curve(all_valid_loss_df, save_file=loss_dir + "valid_loss.svg")
    if info_df is not None:
        if plot_results:
            loss_curve(info_df, save_file=loss_dir + "lr.svg")
            all_loss_curve(
                all_train_loss_df,
                all_valid_loss_df,
                info_df,
                save_file=loss_dir + "all_loss.svg",
            )

    logger.info("Loading test loader...")
    loader = get_simloader(file_prefix="test_", data_dir=data_dir)
    logger.info("Test loader, loaded.")

    alpha_pi = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    alpha_pi.reverse()
    PROSAIL_VAE.eval()
    logger.info("Computing inference metrics with test dataset...")
    test_loss = PROSAIL_VAE.validate(loader, n_samples=n_samples)
    pd.DataFrame(test_loss, index=[0]).to_csv(loss_dir + "/test_loss.csv")
    nlls = PROSAIL_VAE.compute_lat_nlls(loader).mean(0).squeeze()
    torch.save(nlls, res_dir + "/params_nll.pt")

    if bvnet_mode:
        bvnet_validation_dir = res_dir + "/bvnet_validation/"
        if not os.path.isdir(bvnet_validation_dir):
            os.makedirs(bvnet_validation_dir)
        bvnet_data_dir_path = os.path.join(data_dir, os.pardir) + "/bvnet/"
        prosail_ref_params = (
            torch.load(bvnet_data_dir_path + "bvnet_test_prosail_sim_vars.pt")
            .float()
            .to(PROSAIL_VAE.device)
        )
        s2_r = (
            torch.load(bvnet_data_dir_path + "bvnet_test_prosail_s2_sim_refl.pt")
            .float()
            .to(PROSAIL_VAE.device)
        )
        s2_a = prosail_ref_params[:, -3:]
        prosail_ref_params = prosail_ref_params[:, :-3]
        lai_nlls, lai_preds, sim_pdfs, sim_supports = get_bvnet_validation_metrics(
            PROSAIL_VAE, s2_r, s2_a, prosail_ref_params
        )
        torch.save(lai_nlls.cpu(), bvnet_validation_dir + "/bvnet_lai_nll.pt")
        torch.save(lai_preds.cpu(), bvnet_validation_dir + "/bvnet_lai_ref_pred.pt")
        if plot_results:
            fig, ax = plot_lai_preds(
                lai_preds[:, 1].cpu(), lai_preds[:, 0].cpu(), site="bvnet"
            )
            fig.savefig(bvnet_validation_dir + "/bvnet_lai_pred_vs_true.png")
            plot_single_lat_hist_2D(
                heatmap=None,
                extent=None,
                tgt_dist=lai_preds[:, 1].cpu(),
                sim_pdf=sim_pdfs[:, 6, :].cpu(),
                sim_support=sim_supports[:, 6, :].cpu(),
                res_dir=bvnet_validation_dir,
                fig=None,
                ax=None,
                var_name="LAI",
                nbin=100,
            )

    if juan_validation:
        juan_data_dir_path = TOP_PATH + "/field_data/processed/"
        juan_validation_dir = res_dir + "/juan_validation/"
        if not os.path.isdir(juan_validation_dir):
            os.makedirs(juan_validation_dir)
        sites = ["france", "spain1", "italy1", "italy2"]
        (
            j_list_lai_nlls,
            list_lai_preds,
            j_dt_list,
            j_ndvi_list,
        ) = get_juan_validation_metrics(
            PROSAIL_VAE,
            juan_data_dir_path,
            lai_min=0,
            dt_max=10,
            sites=sites,
            bvnet_mode=bvnet_mode,
        )
        all_lai_preds = torch.cat(list_lai_preds)
        all_dt_list = torch.cat(j_dt_list)
        all_ndvi = torch.cat(j_ndvi_list)
        for i, site in enumerate(sites):
            torch.save(
                j_list_lai_nlls[i].cpu(), juan_validation_dir + f"/{site}_lai_nll.pt"
            )
            torch.save(
                list_lai_preds[i].cpu(),
                juan_validation_dir + f"/{site}_lai_ref_pred.pt",
            )
            torch.save(j_dt_list[i].cpu(), juan_validation_dir + f"/{site}_dt.pt")
            if plot_results:
                fig, ax = plot_lai_preds(
                    list_lai_preds[i][:, 1].cpu(),
                    list_lai_preds[i][:, 0].cpu(),
                    j_dt_list[i],
                    site,
                )
                fig.savefig(juan_validation_dir + f"/{site}_lai_pred_vs_true.png")
                fig, ax = plot_lai_vs_ndvi(
                    list_lai_preds[i][:, 1].cpu(),
                    j_ndvi_list[i].cpu(),
                    j_dt_list[i],
                    site,
                )
                fig.savefig(juan_validation_dir + f"/{site}_lai_true_vs_ndvi.png")
                lai_filter = torch.logical_not(
                    torch.logical_and(
                        list_lai_preds[i][:, 1] < 0.5, j_ndvi_list[i] > 0.4
                    )
                ).cpu()
                fig, ax = plot_lai_preds(
                    list_lai_preds[i][lai_filter, 1].cpu(),
                    list_lai_preds[i][lai_filter, 0].cpu(),
                    j_dt_list[i][lai_filter],
                    site,
                )
                fig.savefig(
                    juan_validation_dir + f"/filtered_{site}_lai_pred_vs_true.png"
                )
        if plot_results:
            fig, ax = plot_lai_preds(
                all_lai_preds[:, 1].cpu(), all_lai_preds[:, 0].cpu(), all_dt_list, "all"
            )
            fig.savefig(juan_validation_dir + "/all_lai_pred_vs_true.png")
            lai_err = all_lai_preds[:, 1].cpu() - all_lai_preds[:, 0].cpu()
            fig, ax = plot_lai_vs_ndvi(
                all_lai_preds[lai_err.abs() > 1, 1].cpu(),
                all_ndvi[lai_err.abs() > 1].cpu(),
                all_dt_list[lai_err.abs() > 1],
                "all",
            )
            fig.savefig(juan_validation_dir + "/all_lai_true_vs_ndvi.png")
            lai_filter = torch.logical_not(
                torch.logical_and(all_lai_preds[:, 1] < 0.5, all_ndvi > 0.4)
            ).cpu()
            fig, ax = plot_lai_preds(
                all_lai_preds[lai_filter, 1].cpu(),
                all_lai_preds[lai_filter, 0].cpu(),
                all_dt_list[lai_filter],
                "all",
            )
            fig.savefig(juan_validation_dir + "/filtered_all_lai_pred_vs_true.png")
    if plot_results:
        plot_rec_hist2D(PROSAIL_VAE, loader, res_dir, nbin=50, bands_name=bands_name)
    (
        mae,
        mpiw,
        picp,
        mare,
        sim_dist,
        tgt_dist,
        rec_dist,
        angles_dist,
        s2_r_dist,
        sim_pdfs,
        sim_supports,
        ae_percentiles,
        are_percentiles,
        piw_percentiles,
    ) = get_metrics(PROSAIL_VAE, loader, n_pdf_sample_points=3001, alpha_conf=alpha_pi)
    logger.info("Metrics computed.")

    save_metrics(
        res_dir,
        mae,
        mpiw,
        picp,
        alpha_pi,
        ae_percentiles,
        are_percentiles,
        piw_percentiles,
        var_bounds_type=PROSAIL_VAE.sim_space.var_bounds,
    )
    maer = pd.read_csv(res_dir + "/metrics/maer.csv").drop(columns=["Unnamed: 0"])
    mpiwr = pd.read_csv(res_dir + "/metrics/mpiwr.csv").drop(columns=["Unnamed: 0"])
    if plot_results:
        # Plotting results
        metrics_dir = res_dir + "/metrics_plot/"
        if not os.path.isdir(metrics_dir):
            os.makedirs(metrics_dir)

        logger.info("Plotting metrics.")

        plot_metrics(metrics_dir, alpha_pi, maer, mpiwr, picp, mare)
        plot_metric_boxplot(ae_percentiles, res_dir, metric_name="ae", logscale=True)
        plot_metric_boxplot(are_percentiles, res_dir, metric_name="are")
        # plot_metric_boxplot(piw_percentiles, res_dir, metric_name='piw')
        rec_dir = res_dir + "/reconstruction/"
        if not os.path.isdir(rec_dir):
            os.makedirs(rec_dir)
        logger.info("Plotting reconstructions")
        plot_rec_and_latent(
            PROSAIL_VAE, loader, rec_dir, n_plots=20, bands_name=bands_name
        )

        logger.info("Plotting PROSAIL parameter distributions")
        plot_param_dist(
            metrics_dir,
            sim_dist,
            tgt_dist,
            var_bounds_type=PROSAIL_VAE.sim_space.var_bounds,
        )
        logger.info("Plotting PROSAIL parameters, reference vs prediction")
        plot_lat_hist2D(tgt_dist, sim_pdfs, sim_supports, res_dir, nbin=50)
        plot_pred_vs_tgt(
            metrics_dir,
            sim_dist,
            tgt_dist,
            var_bounds_type=PROSAIL_VAE.sim_space.var_bounds,
        )
        ssimulator = PROSAIL_VAE.decoder.ssimulator
        refl_dist = loader.dataset[:][0]
        plot_refl_dist(
            rec_dist,
            refl_dist,
            res_dir,
            normalized=False,
            ssimulator=PROSAIL_VAE.decoder.ssimulator,
        )

        normed_rec_dist = ssimulator.normalize(rec_dist.to(device))
        normed_refl_dist = ssimulator.normalize(refl_dist.to(device))
        logger.info("Plotting reflectance distribution")
        plot_refl_dist(
            normed_rec_dist,
            normed_refl_dist,
            metrics_dir,
            normalized=True,
            ssimulator=PROSAIL_VAE.decoder.ssimulator,
            bands_name=bands_name,
        )
        logger.info("Plotting reconstructed reflectance components pair plots")
        pair_plot(
            normed_rec_dist,
            tensor_2=None,
            features=BANDS,
            res_dir=metrics_dir,
            filename="normed_rec_pair_plot.png",
        )
        logger.info("Plotting reference reflectance components pair plots")
        pair_plot(
            normed_refl_dist,
            tensor_2=None,
            features=BANDS,
            res_dir=metrics_dir,
            filename="normed_s2bands_pair_plot.png",
        )
        logger.info("Plotting inferred PROSAIL parameters pair plots")
        pair_plot(
            sim_dist.squeeze(),
            tensor_2=None,
            features=PROSAILVARS,
            res_dir=metrics_dir,
            filename="sim_prosail_pair_plot.png",
        )
        logger.info("Plotting reference PROSAIL parameters pair plots")
        pair_plot(
            tgt_dist.squeeze(),
            tensor_2=None,
            features=PROSAILVARS,
            res_dir=metrics_dir,
            filename="ref_prosail_pair_plot.png",
        )
        logger.info("Plotting reconstruction error against angles")
        plot_rec_error_vs_angles(s2_r_dist, rec_dist, angles_dist, res_dir=metrics_dir)

    logger.info("Program completed.")
    return


def get_encoded_image(
    image_tensor,
    PROSAIL_VAE,
    patch_size=32,
    bands=None,
    mode="lat_mode",
):
    if bands is None:
        bands = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    hw = PROSAIL_VAE.encoder.nb_enc_cropped_hw
    patched_tensor = patchify(image_tensor, patch_size=patch_size, margin=hw)
    patched_sim_image = torch.zeros(
        (patched_tensor.size(0), patched_tensor.size(1), 11, patch_size, patch_size)
    ).to(PROSAIL_VAE.device)
    patched_rec_image = torch.zeros(
        (
            patched_tensor.size(0),
            patched_tensor.size(1),
            len(bands),
            patch_size,
            patch_size,
        )
    ).to(PROSAIL_VAE.device)
    for i in range(patched_tensor.size(0)):
        for j in range(patched_tensor.size(1)):
            x = patched_tensor[i, j, bands, :, :]
            angles = torch.zeros(3, patch_size + 2 * hw, patch_size + 2 * hw).to(
                PROSAIL_VAE.device
            )
            angles[0, :, :] = patched_tensor[i, j, 11, :, :]
            angles[1, :, :] = patched_tensor[i, j, 13, :, :]
            angles[2, :, :] = (
                patched_tensor[i, j, 12, :, :] - patched_tensor[i, j, 14, :, :]
            )
            with torch.no_grad():
                dist_params, z, sim, rec = PROSAIL_VAE.point_estimate_rec(
                    x.unsqueeze(0), angles.unsqueeze(0), mode=mode
                )
            patched_rec_image[i, j, :, :, :] = rec
            patched_sim_image[i, j, :, :, :] = sim
    sim_image = unpatchify(patched_sim_image)[
        :, : image_tensor.size(1), : image_tensor.size(2)
    ][:, hw:-hw, hw:-hw]
    rec_image = unpatchify(patched_rec_image)[
        :, : image_tensor.size(1), : image_tensor.size(2)
    ][:, hw:-hw, hw:-hw]
    cropped_image = image_tensor[:, hw:-hw, hw:-hw]
    return rec_image, sim_image, cropped_image


def check_fold_res_dir(fold_dir, n_xp, params):
    same_fold = ""
    all_dirs = os.listdir(fold_dir)
    for d in all_dirs:
        if d.startswith(f"{n_xp}_kfold_{params['k_fold']}_n_{params['n_fold']}"):
            same_fold = d
    return same_fold


def get_res_dir_path(root_results_dir, params, n_xp=None, overwrite_xp=False):
    if not os.path.exists(root_results_dir):
        os.makedirs(root_results_dir)
    if not os.path.exists(root_results_dir + "n_xp.json"):
        save_dict({"xp": 0}, root_results_dir + "n_xp.json")
    if n_xp is None:
        n_xp = load_dict(root_results_dir + "n_xp.json")["xp"] + 1
    save_dict({"xp": n_xp}, root_results_dir + "n_xp.json")
    date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    if params["k_fold"] > 1:
        k_fold_dir = (
            f"{root_results_dir}/{n_xp}_kfold_{params['k_fold']}"
            f"_supervised_{params['supervised']}_{params['dataset_file_prefix']}"
        )
        if not params["supervised"]:
            k_fold_dir + f"kl_{params['beta_kl']}"
        if not os.path.exists(k_fold_dir):
            os.makedirs(k_fold_dir)
        res_dir = (
            f"{k_fold_dir}/{n_xp}_kfold_{params['k_fold']}_n_"
            f"{params['n_fold']}_d{date}_supervised_{params['supervised']}"
            f"_{params['dataset_file_prefix']}"
        )
        same_fold_dir = check_fold_res_dir(k_fold_dir, n_xp, params)
        if len(same_fold_dir) > 0:
            if overwrite_xp:
                warnings.warn(
                    "WARNING: Overwriting existing fold experiment in 5s", stacklevel=2
                )
                sleep(5)
                shutil.rmtree(k_fold_dir + "/" + same_fold_dir)
            else:
                raise ValueError(
                    "The same experiment (fold) has already been carried out at"
                    f" {same_fold_dir}.\n Please change the number of "
                    "fold or allow overwrite"
                )
    else:
        if not socket.gethostname() == PC_SOCKET_NAME:
            res_dir = f"{root_results_dir}/{n_xp}"
        else:
            res_dir = (
                f"{root_results_dir}/{n_xp}_d{date}_supervised_"
                f"{params['supervised']}_{params['dataset_file_prefix']}"
            )
    if not os.path.isdir(res_dir):
        os.makedirs(res_dir)
    return res_dir


def configureEmissionTracker(parser):
    logger = logging.getLogger(LOGGER_NAME)
    try:
        from codecarbon import OfflineEmissionsTracker

        tracker = OfflineEmissionsTracker(
            country_iso_code="FRA", output_dir=parser.root_results_dir
        )
        tracker.start()
        useEmissionTracker = True
    except Exception:
        logger.error(
            "Couldn't start codecarbon ! Emissions not tracked for this execution."
        )
        useEmissionTracker = False
        tracker = None
    return tracker, useEmissionTracker
