import os
import pandas as pd
import numpy as np
import torch
from validation.frm4veg_validation import (interpolate_frm4veg_pred, 
                                           BARRAX_FILENAMES, WYTHAM_FILENAMES, BARRAX_2021_FILENAME,
                                           get_frm4veg_results_at_date)
from validation.belsar_validation import (interpolate_belsar_metrics, save_belsar_predictions, 
                                          BELSAR_FILENAMES, ALL_BELSAR_FILENAMES, get_all_belsar_predictions,
                                          save_bvnet_belsar_predictions)
from prosailvae.ProsailSimus import BANDS

def get_all_campaign_lai_results_BVNET(frm4veg_data_dir, frm4veg2021_data_dir, belsar_data_dir, belsar_pred_dir,
                                      method="simple_interpolate", get_all_belsar=False, remove_files=False, lai_bvnet=None):
    
    all_belsar = None
    list_belsar_filenames = BELSAR_FILENAMES
    if get_all_belsar:
        list_belsar_filenames = ALL_BELSAR_FILENAMES
    save_bvnet_belsar_predictions(belsar_data_dir, belsar_pred_dir, list_belsar_filenames, lai_bvnet=lai_bvnet)
    if get_all_belsar:
        all_belsar = get_all_belsar_predictions(belsar_data_dir, belsar_pred_dir, f"_BVNET")  

    barrax_results = interpolate_frm4veg_pred(None, frm4veg_data_dir, BARRAX_FILENAMES[0], 
                                              BARRAX_FILENAMES[1],  method=method, is_BVNET=True, 
                                              get_reconstruction=False, lai_bvnet=lai_bvnet)
    barrax_2021_results = get_frm4veg_results_at_date(None, frm4veg2021_data_dir, BARRAX_2021_FILENAME,
                                                      is_BVNET=True, get_reconstruction=False, lai_bvnet=lai_bvnet)
    wytham_results = interpolate_frm4veg_pred(None, frm4veg_data_dir, WYTHAM_FILENAMES[0], 
                                              WYTHAM_FILENAMES[1],  method=method, is_BVNET=True,
                                              get_reconstruction=False, lai_bvnet=lai_bvnet)

    belsar_results = interpolate_belsar_metrics(belsar_data_dir=belsar_data_dir, belsar_pred_dir=belsar_pred_dir,
                                                file_suffix="_BVNET", method=method)
    if remove_files:
        for filename in list_belsar_filenames:
            if os.path.isfile(os.path.join(belsar_pred_dir, f"{filename}_BVNET.tif")):
                os.remove(os.path.join(belsar_pred_dir, f"{filename}_BVNET.tif"))
    return barrax_results, barrax_2021_results, wytham_results, belsar_results, all_belsar

def get_all_campaign_CCC_results_BVNET(frm4veg_data_dir, frm4veg2021_data_dir, method="simple_interpolate", 
                                      ccc_bvnet=None, lai_bvnet=None, cab_mode=False):
    
    barrax_results = interpolate_frm4veg_pred(None, frm4veg_data_dir, BARRAX_FILENAMES[0], 
                                              BARRAX_FILENAMES[1],  method=method, is_BVNET=True, 
                                              get_reconstruction=False, ccc_bvnet=ccc_bvnet, lai_bvnet=lai_bvnet, 
                                              cab_mode=cab_mode)
    barrax_2021_results = get_frm4veg_results_at_date(None, frm4veg2021_data_dir, BARRAX_2021_FILENAME,
                                                      is_BVNET=True, get_reconstruction=False, ccc_bvnet=ccc_bvnet, lai_bvnet=lai_bvnet, 
                                                        cab_mode=cab_mode)
    wytham_results = interpolate_frm4veg_pred(None, frm4veg_data_dir, WYTHAM_FILENAMES[0], 
                                              WYTHAM_FILENAMES[1],  method=method, is_BVNET=True,
                                              get_reconstruction=False, ccc_bvnet=ccc_bvnet, lai_bvnet=lai_bvnet, 
                                              cab_mode=cab_mode)
    return barrax_results, barrax_2021_results, wytham_results

def get_all_campaign_lai_results(model, frm4veg_data_dir, frm4veg2021_data_dir, belsar_data_dir, belsar_pred_dir,
                                 mode="sim_tg_mean", method="simple_interpolate", model_name="pvae",
                                 save_reconstruction=False, get_all_belsar=False, remove_files=False):

    barrax_results = interpolate_frm4veg_pred(model, frm4veg_data_dir, BARRAX_FILENAMES[0], 
                                              BARRAX_FILENAMES[1],  method=method, is_BVNET=False, 
                                              get_reconstruction=save_reconstruction, bands_idx=model.encoder.bands)
    
    all_belsar = None
    list_belsar_filenames = BELSAR_FILENAMES
    if get_all_belsar:
        list_belsar_filenames = ALL_BELSAR_FILENAMES
    save_belsar_predictions(belsar_data_dir, model, belsar_pred_dir, list_belsar_filenames, 
                            model_name=model_name, mode=mode, 
                            save_reconstruction=save_reconstruction)
    
    if get_all_belsar:
        all_belsar = get_all_belsar_predictions(belsar_data_dir, belsar_pred_dir, f"_{model_name}_{mode}", 
                                                bands_idx=model.encoder.bands)   
    

    barrax_2021_results = get_frm4veg_results_at_date(model, frm4veg2021_data_dir, BARRAX_2021_FILENAME,
                                                      is_BVNET=False, get_reconstruction=save_reconstruction)
    wytham_results = interpolate_frm4veg_pred(model, frm4veg_data_dir, WYTHAM_FILENAMES[0], 
                                              WYTHAM_FILENAMES[1],  method=method, is_BVNET=False,
                                              get_reconstruction=save_reconstruction, 
                                              bands_idx=model.encoder.bands)

    belsar_results = interpolate_belsar_metrics(belsar_data_dir=belsar_data_dir, belsar_pred_dir=belsar_pred_dir,
                                                file_suffix=f"_{model_name}_{mode}", method=method, 
                                                bands_idx=model.encoder.bands, get_error=save_reconstruction)
    if remove_files:
        for filename in list_belsar_filenames:
            if os.path.isfile(os.path.join(belsar_pred_dir, f"{filename}_{model_name}_{mode}.tif")):
                os.remove(os.path.join(belsar_pred_dir, f"{filename}_{model_name}_{mode}.tif"))
    return barrax_results, barrax_2021_results, wytham_results, belsar_results, all_belsar

def get_belsar_x_frm4veg_lai_results(belsar_results, barrax_results, barrax_2021_results, wytham_results,
                                     frm4veg_lai="lai", get_reconstruction_error=False, bands_idx=torch.arange(10)):


    date_list = [belsar_results['date'].values.reshape(-1),
                 barrax_results[f'{frm4veg_lai}_date'].reshape(-1),
                 barrax_2021_results[f'{frm4veg_lai}_date'].reshape(-1),
                 wytham_results[f"{frm4veg_lai}_date"].reshape(-1)]
    
    ref_lai = np.concatenate([belsar_results['ref_lai'].values.reshape(-1),
                    barrax_results[f'ref_{frm4veg_lai}'].reshape(-1),
                    barrax_2021_results[f'ref_{frm4veg_lai}'].reshape(-1),
                    wytham_results[f'ref_{frm4veg_lai}'].reshape(-1)])
    
    ref_lai_std_list = [belsar_results['ref_lai_std'].values.reshape(-1),
                        barrax_results[f'ref_{frm4veg_lai}_std'].reshape(-1),
                        barrax_2021_results[f'ref_{frm4veg_lai}_std'].reshape(-1),
                        wytham_results[f'ref_{frm4veg_lai}_std'].reshape(-1)]
    
    pred_lai_list = [belsar_results['lai_mean'].values.reshape(-1),
                     barrax_results[frm4veg_lai].reshape(-1),
                     barrax_2021_results[frm4veg_lai].reshape(-1),
                     wytham_results[frm4veg_lai].reshape(-1)]
    
    pred_lai_std_list = [belsar_results['lai_sigma_mean'].values.reshape(-1),
                         barrax_results[f"{frm4veg_lai}_std"].reshape(-1),
                         barrax_2021_results[f"{frm4veg_lai}_std"].reshape(-1),
                         wytham_results[f"{frm4veg_lai}_std"].reshape(-1)]
    
    site_list = (['Belgium'] * len(ref_lai_std_list[0]) 
                 + ['Spain'] * len(ref_lai_std_list[1]) 
                 + ['Spain'] * len(ref_lai_std_list[2]) 
                 + ['England'] * len(ref_lai_std_list[3]))
    campaign_list = (['BelSAR (2018)'] * len(ref_lai_std_list[0]) 
                 + ['FRM4VEG (Barrax - 2018)'] * len(ref_lai_std_list[1]) 
                 + ['FRM4VEG (Barrax - 2021)'] * len(ref_lai_std_list[2]) 
                 + ['FRM4VEG (Wytham - 2018)'] * len(ref_lai_std_list[3]))    
    land_cover_list = [belsar_results['land_cover'].values.reshape(-1),
                       barrax_results[f'{frm4veg_lai}_land_cover'].reshape(-1),
                       barrax_2021_results[f'{frm4veg_lai}_land_cover'].reshape(-1),
                       wytham_results[f'{frm4veg_lai}_land_cover'].reshape(-1)]
    if get_reconstruction_error:
        rec_err = np.concatenate([belsar_results['rec_err_mean'].values.reshape(-1),
                                    barrax_results[f'{frm4veg_lai}_rec_err'].reshape(-1),
                                    barrax_2021_results[f'{frm4veg_lai}_rec_err'].reshape(-1),
                                    wytham_results[f'{frm4veg_lai}_rec_err'].reshape(-1)])
    else:
        rec_err = np.zeros_like(ref_lai)

    bands_rec_err = {}
    for band in np.array(BANDS)[bands_idx.cpu()].tolist():
        if get_reconstruction_error:
            bands_rec_err[band] = np.concatenate([belsar_results[f'{band}_rec_err_mean'].values.reshape(-1),
                                                    barrax_results[f'{frm4veg_lai}_{band}_rec_err'].reshape(-1),
                                                    barrax_2021_results[f'{frm4veg_lai}_{band}_rec_err'].reshape(-1),
                                                    wytham_results[f'{frm4veg_lai}_{band}_rec_err'].reshape(-1)])
        else:
            bands_rec_err[band] = np.zeros_like(ref_lai)

    results = pd.DataFrame(data={f'{frm4veg_lai}':ref_lai,
                                 f'{frm4veg_lai} std':np.concatenate(ref_lai_std_list),
                                 f'Predicted {frm4veg_lai}': np.concatenate(pred_lai_list),
                                 f'Predicted {frm4veg_lai} std':np.concatenate(pred_lai_std_list),
                                 "Site": np.array(site_list),
                                 "Land cover": np.concatenate(land_cover_list),
                                 "Reconstruction error": rec_err,
                                 "Time delta": np.concatenate(date_list),
                                 "Campaign": np.array(campaign_list),
                                })
    for band in np.array(BANDS)[bands_idx.cpu()].tolist():
        results[f"{band} error"] = bands_rec_err[band]
    results['Land cover'].replace('Bare soil ', 'Bare soil', inplace=True)
    results['Land cover'].replace('Cereal reaped', 'Bare soil', inplace=True)
    results['Land cover'].replace('Onion & sunflower', 'Onion', inplace=True)
    results['Land cover'].replace('Corn', 'Maize', inplace=True)
    results['Land cover'].replace('Rappesed', 'Rapeseed', inplace=True)
    return results

def get_frm4veg_ccc_results(barrax_results, barrax_2021_results, wytham_results,
                            frm4veg_ccc="ccc", get_reconstruction_error=False, bands_idx=torch.arange(10)):


    date_list = [barrax_results[f'{frm4veg_ccc}_date'].reshape(-1),
                 barrax_2021_results[f'{frm4veg_ccc}_date'].reshape(-1),
                 wytham_results[f"{frm4veg_ccc}_date"].reshape(-1)]
    
    ref_ccc = np.concatenate([
                    barrax_results[f'ref_{frm4veg_ccc}'].reshape(-1),
                    barrax_2021_results[f'ref_{frm4veg_ccc}'].reshape(-1),
                    wytham_results[f'ref_{frm4veg_ccc}'].reshape(-1)])
    
    ref_ccc_std_list = [barrax_results[f'ref_{frm4veg_ccc}_std'].reshape(-1),
                        barrax_2021_results[f'ref_{frm4veg_ccc}_std'].reshape(-1),
                        wytham_results[f'ref_{frm4veg_ccc}_std'].reshape(-1)]
    
    pred_ccc_list = [barrax_results[frm4veg_ccc].reshape(-1),
                     barrax_2021_results[frm4veg_ccc].reshape(-1),
                     wytham_results[frm4veg_ccc].reshape(-1)]
    
    pred_ccc_std_list = [barrax_results[f"{frm4veg_ccc}_std"].reshape(-1),
                         barrax_2021_results[f"{frm4veg_ccc}_std"].reshape(-1),
                         wytham_results[f"{frm4veg_ccc}_std"].reshape(-1)]
    
    site_list = (  ['Spain'] * len(pred_ccc_std_list[0]) 
                 + ['Spain'] * len(pred_ccc_std_list[1]) 
                 + ['England'] * len(pred_ccc_std_list[2]))
    campaign_list = (
                   ['FRM4VEG (Barrax - 2018)'] * len(pred_ccc_std_list[0]) 
                 + ['FRM4VEG (Barrax - 2021)'] * len(pred_ccc_std_list[1]) 
                 + ['FRM4VEG (Wytham - 2018)'] * len(pred_ccc_std_list[2]))    
    land_cover_list = [
                       barrax_results[f'{frm4veg_ccc}_land_cover'].reshape(-1),
                       barrax_2021_results[f'{frm4veg_ccc}_land_cover'].reshape(-1),
                       wytham_results[f'{frm4veg_ccc}_land_cover'].reshape(-1)]
    if get_reconstruction_error:
        rec_err = np.concatenate([barrax_results[f'{frm4veg_ccc}_rec_err'].reshape(-1),
                                    barrax_2021_results[f'{frm4veg_ccc}_rec_err'].reshape(-1),
                                    wytham_results[f'{frm4veg_ccc}_rec_err'].reshape(-1)])
    else:
        rec_err = np.zeros_like(ref_ccc)

    bands_rec_err = {}
    for band in np.array(BANDS)[bands_idx.cpu()].tolist():
        if get_reconstruction_error:
            bands_rec_err[band] = np.concatenate([  barrax_results[f'{frm4veg_ccc}_{band}_rec_err'].reshape(-1),
                                                    barrax_2021_results[f'{frm4veg_ccc}_{band}_rec_err'].reshape(-1),
                                                    wytham_results[f'{frm4veg_ccc}_{band}_rec_err'].reshape(-1)])
        else:
            bands_rec_err[band] = np.zeros_like(ref_ccc)
    results = pd.DataFrame(data={f'{frm4veg_ccc}':ref_ccc,
                                f'{frm4veg_ccc} std':np.concatenate(ref_ccc_std_list),
                                f'Predicted {frm4veg_ccc}': np.concatenate(pred_ccc_list),
                                f'Predicted {frm4veg_ccc} std':np.concatenate(pred_ccc_std_list),
                                "Site": np.array(site_list),
                                "Land cover": np.concatenate(land_cover_list),
                                "Reconstruction error": rec_err,
                                "Time delta": np.concatenate(date_list),
                                "Campaign": np.array(campaign_list),
                                })
    for band in np.array(BANDS)[bands_idx.cpu()].tolist():
        results[f"{band} error"] = bands_rec_err[band]
    results['Land cover'].replace('Bare soil ', 'Bare soil', inplace=True)
    results['Land cover'].replace('Cereal reaped', 'Bare soil', inplace=True)
    results['Land cover'].replace('Onion & sunflower', 'Onion', inplace=True)
    results['Land cover'].replace('Corn', 'Maize', inplace=True)
    results['Land cover'].replace('Rappesed', 'Rapeseed', inplace=True)
    return results

def get_validation_global_metrics(df_results, decompose_along_columns=["Site", "Land cover"], 
                                  n_sigma=3, variable="lai"):
    global_rmse_dict = {}
    global_picp_dict = {}
    global_mpiw_dict = {}
    global_mestdr_dict = {}
    for column in decompose_along_columns:
        rmse = {}
        for _, element in enumerate(pd.unique(df_results[column])):
            results = df_results[df_results[column]==element]
            rmse[f"{variable}_rmse_{element}"] = np.sqrt((results[f'Predicted {variable}'] - results[variable]).pow(2).mean())
        rmse[f"{variable}_rmse_all"] = np.sqrt((df_results[f'Predicted {variable}'] - df_results[variable]).pow(2).mean())
        global_rmse_dict[column] = pd.DataFrame(data=rmse, index=[0])
    if n_sigma>0:
        for column in decompose_along_columns:
            picp = {}
            for _, element in enumerate(pd.unique(df_results[column])):
                results = df_results[df_results[column]==element]
                picp[f"{variable}_picp_{element}"] = np.logical_and(results[variable] < results[f'Predicted {variable}'] + n_sigma * results[f'Predicted {variable} std'],
                                               results[variable] > results[f'Predicted {variable}'] - n_sigma * results[f'Predicted {variable} std']).astype(int).mean()
            picp[f"{variable}_picp_all"] = np.logical_and(df_results[variable] < df_results[f'Predicted {variable}'] + n_sigma * df_results[f'Predicted {variable} std'],
                                         df_results[variable] > df_results[f'Predicted {variable}'] - n_sigma * df_results[f'Predicted {variable} std']).astype(int).mean()
            global_picp_dict[column] = pd.DataFrame(data=picp, index=[0])
    if n_sigma>0:
        for column in decompose_along_columns:
            mpiw = {}
            for _, element in enumerate(pd.unique(df_results[column])):
                results = df_results[df_results[column]==element]
                mpiw[f"{variable}_mpiw_{element}"] = (2 * n_sigma * results[f'Predicted {variable} std']).mean()
            mpiw[f"{variable}_mpiw_all"] = (2 * n_sigma * df_results[f'Predicted {variable} std']).mean()
            global_mpiw_dict[column] = pd.DataFrame(data=mpiw, index=[0])
    for column in decompose_along_columns:
        mestdr = {}
        for _, element in enumerate(pd.unique(df_results[column])):
            results = df_results[df_results[column]==element]
            mestdr[f"{variable}_mestdr_{element}"] = (np.abs(results[variable] - results[f'Predicted {variable}']) / results[f'Predicted {variable} std']).mean()
        mestdr[f"{variable}_mestdr_all"] = (np.abs(df_results[variable] - df_results[f'Predicted {variable}']) /  df_results[f'Predicted {variable} std']).mean()
        global_mestdr_dict[column] = pd.DataFrame(data=mestdr, index=[0])
    return global_rmse_dict, global_picp_dict, global_mpiw_dict, global_mestdr_dict