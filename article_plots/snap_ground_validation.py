import torch
import os
import numpy as np
import pandas as pd

from bvnet_regression.bvnet import BVNET
import matplotlib.pyplot as plt
from validation.validation import (get_all_campaign_CCC_results_BVNET, get_frm4veg_ccc_results, 
                                   get_validation_global_metrics, get_all_campaign_lai_results_BVNET,
                                   get_belsar_x_frm4veg_lai_results)
from metrics.prosail_plots import regression_plot, regression_plot_2hues
import tikzplotlib

def tikzplotlib_fix_ncols(obj):
    """
    workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)

def main():
    frm4veg_data_dir = "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/frm4veg_validation"
    frm4veg_2021_data_dir = "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/frm4veg_2021_validation"
    belsar_data_dir = "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/belSAR_validation"
    belsar_pred_dir = "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/results/bvnet_belsar_pred/"
    if not os.path.isdir(belsar_pred_dir):
        os.makedirs(belsar_pred_dir)
    res_dir = "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/results/bvnet_ground_validation/"
    if not os.path.isdir(res_dir):
        os.makedirs(res_dir)


    (barrax_results, barrax_2021_results, wytham_results, belsar_results, 
    all_belsar) = get_all_campaign_lai_results_BVNET(frm4veg_data_dir, frm4veg_2021_data_dir, 
                                                    belsar_data_dir, belsar_pred_dir,
                                                    method="simple_interpolate", get_all_belsar=False, 
                                                    remove_files=True, lai_bvnet=None)    
    df_results_lai = get_belsar_x_frm4veg_lai_results(belsar_results, barrax_results, barrax_2021_results, wytham_results,
                                                    frm4veg_lai="lai", get_reconstruction_error=False)
    hue_elem = pd.unique(df_results_lai["Land cover"])
    hue2_elem = pd.unique(df_results_lai["Campaign"])
    hue_color_dict= {}
    for j, h_e in enumerate(hue_elem):
        hue_color_dict[h_e] = f"C{j}"
    default_markers = ["o", "v", "D", "s", "+", ".", "^", "1"]
    hue2_markers_dict= {}
    for j, h2_e in enumerate(hue2_elem):
        hue2_markers_dict[h2_e] = default_markers[j]
    variable="lai"
    fig, ax = regression_plot_2hues(df_results_lai, x=f"{variable}", y=f"Predicted {variable}", 
                                    fig=None, ax=None, hue="Land cover", hue2="Campaign", display_text=False,
                                    legend_col=1, error_x=f"{variable} std", 
                                    error_y=f"Predicted {variable} std", hue_perfs=False, 
                                    title_hue="Land cover", title_hue2="\n Site", 
                                    hue_color_dict=hue_color_dict, 
                                    hue2_markers_dict=hue2_markers_dict)
    fig.savefig(os.path.join(res_dir, f"bvnet_{variable}_regression_campaign.png"))
    tikzplotlib_fix_ncols(fig)
    tikzplotlib.save(os.path.join(res_dir, f"bvnet_{variable}_regression_campaign.tex"))
    rmse_lai, mpiw_lai, picp_lai, _ = get_validation_global_metrics(df_results_lai, decompose_along_columns=["Campaign"], variable="lai")

    barrax_results, barrax_2021_results, wytham_results = get_all_campaign_CCC_results_BVNET(frm4veg_data_dir, 
                                                                                            frm4veg_2021_data_dir,
                                                                                            ccc_bvnet=None, 
                                                                                            cab_mode=False)
    df_results_ccc = get_frm4veg_ccc_results(barrax_results, barrax_2021_results, wytham_results, frm4veg_ccc="ccc",
                                         get_reconstruction_error=False)
    variable="ccc"
    df_results_ccc['Land cover'].replace('Bare soil ', 'Bare soil', inplace=True)
    fig, ax = regression_plot_2hues(df_results_ccc, x=f"{variable}", y=f"Predicted {variable}", 
                                    fig=None, ax=None, hue="Land cover", hue2="Campaign", display_text=False,
                                    legend_col=1, error_x=f"{variable} std", 
                                    error_y=f"Predicted {variable} std", hue_perfs=False, 
                                    title_hue="Land cover", title_hue2="\n Site",
                                    hue_color_dict=hue_color_dict, 
                                    hue2_markers_dict=hue2_markers_dict)
    fig.savefig(os.path.join(res_dir, f"bvnet_{variable}_regression_campaign.png"))
    tikzplotlib_fix_ncols(fig)
    tikzplotlib.save(os.path.join(res_dir, f"bvnet_{variable}_regression_campaign.tex"))
    rmse_ccc, mpiw_ccc, picp_ccc, _ = get_validation_global_metrics(df_results_ccc, decompose_along_columns=["Campaign"], variable="ccc")
            

if __name__ =="__main__":
    main()