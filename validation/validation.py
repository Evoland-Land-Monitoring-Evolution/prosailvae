import pandas as pd
import numpy as np
from validation.frm4veg_validation import interpolate_frm4veg_pred, BARRAX_FILENAMES, WYTHAM_FILENAMES
from validation.belsar_validation import interpolate_belsar_metrics, save_belsar_predictions, BELSAR_FILENAMES

def get_all_campaign_lai_results(model, frm4veg_data_dir, belsar_data_dir, belsar_pred_dir,
                                 mode="sim_tg_mean", method="simple_interpolate", model_name="pvae",):
    


    barrax_results = interpolate_frm4veg_pred(model, frm4veg_data_dir, BARRAX_FILENAMES[0], 
                                              BARRAX_FILENAMES[1],  method=method, is_SNAP=False, 
                                              )

    wytham_results = interpolate_frm4veg_pred(model, frm4veg_data_dir, WYTHAM_FILENAMES[0], 
                                              WYTHAM_FILENAMES[1],  method=method, is_SNAP=False)
    
    save_belsar_predictions(belsar_data_dir, model, belsar_pred_dir, BELSAR_FILENAMES, model_name=model_name, mode=mode, 
                            save_reconstruction=True)
        
    belsar_results = interpolate_belsar_metrics(belsar_data_dir=belsar_data_dir, belsar_pred_dir=belsar_pred_dir,
                                                file_suffix=f"_{model_name}_{mode}", method=method)

    return barrax_results, wytham_results, belsar_results

def get_belsar_x_frm4veg_lai_results(belsar_results, barrax_results, wytham_results,
                                     frm4veg_lai="lai"):
    rec_err_list = [belsar_results['rec_err_mean'].values.reshape(-1),
                    barrax_results[f'{frm4veg_lai}_rec_err'].reshape(-1),
                    wytham_results[f'{frm4veg_lai}_rec_err'].reshape(-1)]
    date_list = [belsar_results['date'].values.reshape(-1),
                 barrax_results[f'{frm4veg_lai}_date'].reshape(-1),
                 wytham_results[f"{frm4veg_lai}_date"].reshape(-1)]
    
    ref_lai_list = [belsar_results['ref_lai'].values.reshape(-1),
                    barrax_results[f'ref_{frm4veg_lai}'].reshape(-1),
                    wytham_results[f'ref_{frm4veg_lai}'].reshape(-1)]
    
    ref_lai_std_list = [belsar_results['ref_lai_std'].values.reshape(-1),
                        barrax_results[f'ref_{frm4veg_lai}_std'].reshape(-1),
                        wytham_results[f'ref_{frm4veg_lai}_std'].reshape(-1)]
    
    pred_lai_list = [belsar_results['lai_mean'].values.reshape(-1),
                     barrax_results[frm4veg_lai].reshape(-1),
                     wytham_results[frm4veg_lai].reshape(-1)]
    pred_lai_std_list = [belsar_results['lai_sigma_mean'].values.reshape(-1),
                         barrax_results[f"{frm4veg_lai}_std"].reshape(-1),
                         wytham_results[f"{frm4veg_lai}_std"].reshape(-1)]
    site_list = ['Belgium'] * len(ref_lai_list[0]) + ['Spain'] * len(ref_lai_list[1]) + ['England'] * len(ref_lai_list[2])
    land_cover_list = [belsar_results['land_cover'].values.reshape(-1),
                       barrax_results[f'{frm4veg_lai}_land_cover'].reshape(-1),
                       wytham_results[f'{frm4veg_lai}_land_cover'].reshape(-1)]

    results = pd.DataFrame(data={'LAI': np.concatenate(ref_lai_list),
                                'LAI std':np.concatenate(ref_lai_std_list),
                                'Predicted LAI': np.concatenate(pred_lai_list),
                                'Predicted LAI std':np.concatenate(pred_lai_std_list),
                                "Site": np.array(site_list),
                                "Land cover": np.concatenate(land_cover_list),
                                "Reconstruction error": np.concatenate(rec_err_list),
                                "Time delta": np.concatenate(date_list)},
                                )
    return results

def get_validation_global_metrics(df_results, decompose_along_columns = ["Site", "Land cover"], n_sigma=3):
    global_rmse_dict = {}
    global_picp_dict = {}
    for column in decompose_along_columns:
        rmse = {}
        for _, element in enumerate(pd.unique(df_results[column])):
            results = df_results[df_results[column]==element]
            rmse[element] = np.sqrt((results['Predicted LAI'] - results['LAI']).pow(2).mean())
        rmse["All"] = np.sqrt((df_results['Predicted LAI'] - df_results['LAI']).pow(2).mean())
        global_rmse_dict[column] = pd.DataFrame(data=rmse, index=[0])
    if n_sigma>0:
        for column in decompose_along_columns:
            picp = {}
            for _, element in enumerate(pd.unique(df_results[column])):
                results = df_results[df_results[column]==element]
                picp[element] = np.logical_and(results['LAI'] < results['Predicted LAI'] + n_sigma * results['Predicted LAI std'],
                                               results['LAI'] > results['Predicted LAI'] - n_sigma * results['Predicted LAI std']).astype(int).mean()
            picp["All"] = np.logical_and(df_results['LAI'] < df_results['Predicted LAI'] + n_sigma * df_results['Predicted LAI std'],
                                         df_results['LAI'] > df_results['Predicted LAI'] - n_sigma * df_results['Predicted LAI std']).astype(int).mean()
            global_picp_dict[column] = pd.DataFrame(data=picp, index=[0])
    return global_rmse_dict, global_picp_dict