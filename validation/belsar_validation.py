import os
import pandas as pd
import geopandas as gpd
import rasterio as rio
import numpy as np
import socket
import argparse
import zipfile
import shutil
from tqdm import tqdm
import fiona
fiona.drvsupport.supported_drivers['KML'] = 'rw'
from rasterio.mask import mask
if not __name__=="__main__":
    from .validation_utils import simple_interpolate, read_data_from_theia
else:
    from validation_utils import simple_interpolate, read_data_from_theia
from utils.image_utils import tensor_to_raster, get_encoded_image_from_batch
from datetime import datetime
import torch

from snap_regression.snap_nn import SnapNN
from sensorsio import utils
import matplotlib.pyplot as plt

BELSAR_FILENAMES = ["2A_20180508_both_BelSAR_agriculture_database",     # OK
                    "2A_20180518_both_BelSAR_agriculture_database",     # Nuages mais + non détectés => A retirer !
                    "2B_20180526_both_BelSAR_agriculture_database",
                    "2A_20180528_both_BelSAR_agriculture_database",     # Nuages sur l'image => A retirer !
                    "2A_20180620_both_BelSAR_agriculture_database",     # Nuageuse + nuages non détectés  => A retirer !  
                    "2A_20180627_both_BelSAR_agriculture_database",     # OK
                    "2A_20180630_both_BelSAR_agriculture_database",
                    "2B_20180702_both_BelSAR_agriculture_database",
                    "2B_20180715_both_BelSAR_agriculture_database",     # OK
                    "2B_20180722_both_BelSAR_agriculture_database",     # Nuageuse Mais
                    "2B_20180725_both_BelSAR_agriculture_database",
                    "2A_20180727_both_BelSAR_agriculture_database",     # OK
                    "2B_20180804_both_BelSAR_agriculture_database"]     # OK


all_filename_dict = {"2018-05-08": "2A_20180508_both_BelSAR_agriculture_database",     # OK
                    "2018-05-18": "2A_20180518_both_BelSAR_agriculture_database",     # Nuages mais + non détectés => A retirer !
                    "2018-05-26": "2B_20180526_both_BelSAR_agriculture_database",
                    "2018-05-28": "2A_20180528_both_BelSAR_agriculture_database",     # Nuages sur l'image => A retirer !
                    "2018-06-20": "2A_20180620_both_BelSAR_agriculture_database",     # Nuageuse + nuages non détectés  => A retirer !  
                    "2018-06-27": "2A_20180627_both_BelSAR_agriculture_database",     # OK
                    "2018-06-30": "2A_20180630_both_BelSAR_agriculture_database",
                    "2018-07-02": "2B_20180702_both_BelSAR_agriculture_database",
                    "2018-07-15": "2B_20180715_both_BelSAR_agriculture_database",     # OK
                    "2018-07-22": "2B_20180722_both_BelSAR_agriculture_database",     # Nuageuse Mais
                    "2018-07-25": "2B_20180725_both_BelSAR_agriculture_database",
                    "2018-07-27": "2A_20180727_both_BelSAR_agriculture_database",     # OK
                    "2018-08-04": "2B_20180804_both_BelSAR_agriculture_database"}  



def plot_belsar_site(data_dir, filename):
    df, s2_r, s2_a, mask, xcoords, ycoords, crs = load_belsar_validation_data(data_dir, filename)

    # fig, ax = plt.subplots()
    # visu, _, _ = utils.rgb_render(s2_r)
    # ax.imshow(visu, extent = [xcoords[0], xcoords[-1], ycoords[-1], ycoords[0]])
    # wheat_sites.plot(ax=ax,  color = 'red')
    # maize_sites.plot(ax=ax,  color = 'blue')
    maize_sites = get_sites_geometry(data_dir, crs, crop="maize")
    wheat_sites = get_sites_geometry(data_dir, crs, crop="wheat")
    mask[mask==0.] = np.nan
    fig, ax = plt.subplots(dpi=200)
    visu, _, _ = utils.rgb_render(s2_r)
    ax.imshow(visu, extent = [xcoords[0], xcoords[-1], ycoords[-1], ycoords[0]])
    ax.imshow(mask.squeeze(), extent = [xcoords[0], xcoords[-1], ycoords[-1], ycoords[0]], cmap='YlOrRd')
    for i in range(len(maize_sites)):
        contour = maize_sites["geometry"].iloc[i].exterior.xy
        ax.plot(contour[0], contour[1], "blue", linewidth=0.5)
    # maize_sites.plot(ax=ax,  color = 'blue')
    # wheat_sites.plot(ax=ax,  color = 'green')
    for i in range(len(maize_sites)):
        contour = wheat_sites["geometry"].iloc[i].exterior.xy
        ax.plot(contour[0], contour[1], "red", linewidth=0.5)
    for xi, yi, text in zip(wheat_sites.centroid.x, wheat_sites.centroid.y, wheat_sites["Name"]):
        ax.annotate(text,
                xy=(xi, yi), xycoords='data',
                xytext=(1.5, 1.5), textcoords='offset points',
                color='red')
    for xi, yi, text in zip(maize_sites.centroid.x, maize_sites.centroid.y, maize_sites["Name"]):
        ax.annotate(text,
                xy=(xi, yi), xycoords='data',
                xytext=(1.5, 1.5), textcoords='offset points',
                color='blue')
    
    fig.savefig(os.path.join(data_dir, filename + "_mask.png"))

def get_data_point_bb(gdf, dataset, margin=100, res=10):
    left, right, bottom, top = (dataset.bounds.left, dataset.bounds.right, dataset.bounds.bottom, dataset.bounds.top)
    x_data_point = np.round(gdf["geometry"].x.values / res) * res
    y_data_point = np.round(gdf["geometry"].y.values / res) * res
    assert all(x_data_point > left) and all(x_data_point < right)
    assert all(y_data_point > bottom) and all(x_data_point < top)
    left = float(int(min(x_data_point) - margin))
    bottom = float(int(min(y_data_point) - margin))
    right = float(int(max(x_data_point) + margin))
    top = float(int(max(y_data_point) + margin))
    return rio.coords.BoundingBox(left, bottom, right, top)

# def get_data_idx_in_image(gdf, xmin_image_bb, ymax_image_bb, col_offset, row_offset, res=10):
#     x_data_point = (np.round(gdf["geometry"].x.values / 10) * 10 - xmin_image_bb) / res
#     y_data_point = (ymax_image_bb - np.round(gdf["geometry"].y.values / 10) * 10) / res
#     gdf["x_idx"] = x_data_point - col_offset
#     print(gdf["x_idx"])
#     gdf["y_idx"] = y_data_point - row_offset
#     print(gdf["y_idx"])

def get_prosailvae_train_parser():
    """
    Creates a new argument parser.
    """
    parser = argparse.ArgumentParser(description='Parser for data generation')

    parser.add_argument("-f", dest="data_filename",
                        help="name of data files (without extension)",
                        type=str, default="FRM_Veg_Barrax_20180605")

    parser.add_argument("-d", dest="data_dir",
                        help="name of data files (without extension)",
                        type=str, default="/home/uz/zerahy/scratch/prosailvae/data/silvia_validation/")
    
    parser.add_argument("-p", dest="product_name",
                        help="Theia product name",
                        type=str, default="")
    return parser

def get_variable_column_names(variable="lai"):
    if variable == "lai":
        return "LAI", "Uncertainty.1"
    if variable == "lai_eff":
        return "LAIeff", "Uncertainty"
    if variable == "ccc":
        return "CCC (g m-2)", "Uncertainty (g m-2).2"
    if variable == "ccc_eff":
        return "CCCeff (g m-2)", "Uncertainty (g m-2).1"
    else:
        raise NotImplementedError


def preprocess_belsar_validation_data(data_dir, filename, s2_product_name, crop ="maize", margin=100):

    output_file_name = s2_product_name[8:19] + f"_{crop}_" + filename
    data_file = filename + ".xlsx"
    if crop != "both":
        data_df = pd.read_excel(os.path.join(data_dir, data_file), sheet_name=f"BelSAR_{crop}", skiprows=[0])
        data_df.rename(columns={"Date": "date", "PAI": "lai", "Sample dry weight (g)": "cm"}, inplace=True)
    else:
        wheat_df = pd.read_excel(os.path.join(data_dir, data_file), sheet_name=f"BelSAR_wheat", skiprows=[0])
        wheat_df.rename(columns={"Date": "date", "PAI": "lai", "Sample dry weight (g)": "cm"}, inplace=True)
        maize_df = pd.read_excel(os.path.join(data_dir, data_file), sheet_name=f"BelSAR_maize", skiprows=[0])
        maize_df.rename(columns={"Date": "date", "GAI": "lai", "Sample dry weight (g)": "cm"}, inplace=True)
        data_df = pd.concat((wheat_df, maize_df))
    data_df = data_df.drop(columns=['Flight N°', 'BBCH', 'Line 1-1',
                                    'Line 1-2', 'Line 1-3', 'Line 2-1', 'Line 2-2', 'Line 2-3', 'Line 3-1',
                                    'Line 3-2', 'Line 3-3', 'Mean','FCOVER',
                                    'Total Fresh weight (g)', 'Sample fresh wieght (g)', 'Dry matter content (%)',
                                    'Interline distance mean (cm)', 'Interplant distance (cm)',
                                    'Note/comment']).dropna()
    data_df.to_csv(os.path.join(data_dir, output_file_name + ".csv"))

    if crop == "maize":
        left, bottom, right, top = (526023, 6552201, 535123, 6558589)
    elif crop== "wheat":
        left, bottom, right, top = (522857, 6544318, 531075, 6553949)
    elif crop== "both":
        left, bottom, right, top = (522857, 6544318, 535123, 6558589)
    else:
        raise ValueError
    src_epsg = "epsg:3857"
    path_to_theia_product = os.path.join(data_dir, s2_product_name)
    (s2_r, s2_a, validity_mask, xcoords, ycoords, 
     crs) = read_data_from_theia(left, bottom, right, top, src_epsg, path_to_theia_product, margin=margin)
    np.save(os.path.join(data_dir, output_file_name + "_angles.npy"), s2_a)
    np.save(os.path.join(data_dir, output_file_name + "_refl.npy"), s2_r)
    np.save(os.path.join(data_dir, output_file_name + "_xcoords.npy"), xcoords)
    np.save(os.path.join(data_dir, output_file_name + "_ycoords.npy"), ycoords)
    np.save(os.path.join(data_dir, output_file_name + "_mask.npy"), validity_mask)
    
    return output_file_name


def load_belsar_validation_data(data_dir, filename):
    df = pd.read_csv(os.path.join(data_dir, filename + ".csv"))
    s2_r = np.load(os.path.join(data_dir, filename + "_refl.npy"))
    s2_a = np.load(os.path.join(data_dir, filename + "_angles.npy"))
    mask = np.load(os.path.join(data_dir, filename + "_mask.npy"))
    xcoords = np.load(os.path.join(data_dir, filename + "_xcoords.npy"))
    ycoords = np.load(os.path.join(data_dir, filename + "_ycoords.npy"))
    return df, s2_r, s2_a, mask, xcoords, ycoords, rio.CRS.from_epsg(32631)

def get_sites_geometry(data_dir, crs, crop:str|None=None):
    sites_geometry = gpd.GeoDataFrame()
    for layer in fiona.listlayers(os.path.join(data_dir,"BelSAR"+'.kml')):
        s = gpd.read_file(os.path.join(data_dir,"BelSAR"+'.kml'), driver='KML', layer=layer)
        sites_geometry = pd.concat([sites_geometry, s], ignore_index=True)
    sites_geometry.to_crs(crs, inplace=True)
    sites_geometry = sites_geometry[sites_geometry["Name"].apply(lambda x: len(x) > 0)]
    sites_geometry = sites_geometry[sites_geometry['geometry'].apply(lambda x : x.geom_type=='Polygon')]
    sites_geometry.reset_index(inplace=True, drop=True)
    if crop is None:
        return sites_geometry
    if crop == "maize":
        return sites_geometry[sites_geometry["Name"].apply(lambda x: x[0]=="M")]    
    if crop == "wheat":
        return sites_geometry[sites_geometry["Name"].apply(lambda x: x[0]=="W")]
    else:
        raise ValueError


def get_delta_dict(filename_dict):
    delta_dict = {}
    for date, filename in filename_dict.items():
        filename_date_str = filename[3:11]
        delta = (datetime.strptime(date, "%Y-%m-%d")
                 - datetime.strptime(filename_date_str, "%Y%m%d")).days
        delta_dict[date] = delta
    return delta_dict

closest_filename_dict = {"2018-05-17" : "2A_20180518_both_BelSAR_agriculture_database",
                         '2018-05-18' : "2A_20180518_both_BelSAR_agriculture_database",
                         '2018-05-31' : "2A_20180528_both_BelSAR_agriculture_database",
                         '2018-06-01' : "2A_20180528_both_BelSAR_agriculture_database",
                         '2018-06-05' : "2A_20180528_both_BelSAR_agriculture_database",
                         '2018-06-21' : "2A_20180620_both_BelSAR_agriculture_database",
                         '2018-06-22' : "2A_20180620_both_BelSAR_agriculture_database",
                         '2018-07-19' : "2B_20180715_both_BelSAR_agriculture_database",
                         '2018-08-02' : "2B_20180804_both_BelSAR_agriculture_database",
                                #  '2018-08-29'
                                 }


before_filename_dict =  {"2018-05-17" : "2A_20180508_both_BelSAR_agriculture_database",
                         '2018-05-18' : "2A_20180518_both_BelSAR_agriculture_database",
                         '2018-05-31' : "2A_20180528_both_BelSAR_agriculture_database",
                         '2018-06-01' : "2A_20180528_both_BelSAR_agriculture_database",
                         '2018-06-05' : "2A_20180528_both_BelSAR_agriculture_database",
                         '2018-06-21' : "2A_20180620_both_BelSAR_agriculture_database",
                         '2018-06-22' : "2A_20180620_both_BelSAR_agriculture_database",
                         '2018-07-19' : "2B_20180715_both_BelSAR_agriculture_database",
                         '2018-08-02' : "2A_20180727_both_BelSAR_agriculture_database",
                                #  '2018-08-29'
                                 }

after_filename_dict =  {"2018-05-17" : "2A_20180518_both_BelSAR_agriculture_database",
                        '2018-05-18' : "2A_20180518_both_BelSAR_agriculture_database",
                        '2018-05-31' : "2A_20180620_both_BelSAR_agriculture_database",
                        '2018-06-01' : "2A_20180620_both_BelSAR_agriculture_database",
                        '2018-06-05' : "2A_20180620_both_BelSAR_agriculture_database",
                        '2018-06-21' : "2A_20180627_both_BelSAR_agriculture_database",
                        '2018-06-22' : "2A_20180627_both_BelSAR_agriculture_database",
                        '2018-07-19' : "2B_20180722_both_BelSAR_agriculture_database",
                        '2018-08-02' : "2B_20180804_both_BelSAR_agriculture_database",
                                #  '2018-08-29'
                         }
def get_belsar_image_metrics(sites_geometry, validation_df, belsar_pred_dir, belsar_pred_filename, 
                             belsar_pred_file_suffix, date, delta_t, NO_DATA=-10000, get_error=True):
    """
    Get metrics df for single image prediction for all sites
    """
    pred_array_idx = {"lai":{"mean":0, "sigma":2}, "cm":{"mean":1, "sigma":3}}
    metrics = pd.DataFrame()
    for i in range(len(sites_geometry)):
        line = sites_geometry.iloc[i]
        site_name  = line['Name']
        polygon = line['geometry']
        with rio.open(os.path.join(belsar_pred_dir, f"{belsar_pred_filename}{belsar_pred_file_suffix}.tif"), 
                      mode = 'r') as src:
            masked_array, _ = mask(src, [polygon], invert=False)
            masked_array[masked_array==NO_DATA] = np.nan
            if get_error:
                masked_err = masked_array[-1,...] 

        site_samples = validation_df[validation_df["Field ID"]==site_name]

        d = {"name" : site_name,
             "land_cover":"Wheat" if site_name[0]=="W" else "Maize",
             "date" : date,
             "delta": delta_t}
        for variable in ["lai", "cm"]:
            site_ref = site_samples[variable]   
            d[f"ref_{variable}"] = np.mean(site_ref)
            d[f"ref_{variable}_std"] = np.std(site_ref)
            d[f"{variable}_mean"] = np.nan
            d[f"{variable}_std"] = np.nan
            d[f"{variable}_sigma_mean"] = np.nan
            d[f"{variable}_sigma_std"] = np.nan

            if not np.isnan(masked_array[pred_array_idx[variable]['mean'],...]).all():
                d[f"{variable}_mean"] = np.nanmean(masked_array[pred_array_idx[variable]['mean'],...])
                d[f"{variable}_std"] = np.nanstd(masked_array[pred_array_idx[variable]['mean'],...])
            else:
                continue
            if not np.isnan(masked_array[pred_array_idx[variable]['sigma'],...]).all():
                d[f"{variable}_sigma_mean"] = np.nanmean(masked_array[pred_array_idx[variable]['sigma'],...])
                d[f"{variable}_sigma_std"] = np.nanstd(masked_array[pred_array_idx[variable]['sigma'],...])
            if get_error:
                if not np.isnan(masked_err).all():
                    d[f"rec_err_mean"] = np.nanmean(masked_err)
                    d[f"rec_err_std"] = np.nanstd(masked_err)

        metrics = pd.concat((metrics, pd.DataFrame(d, index=[0])))
    return metrics.reset_index(drop=True)

def get_belsar_campaign_metrics_df(belsar_data_dir, filename_dict, belsar_pred_dir, file_suffix, NO_DATA=-10000, 
                                   get_error=True):
    """
    Get metrics for all sites at all dates (all images)
    """
    metrics = pd.DataFrame()
    delta_dict = get_delta_dict(filename_dict)
    for date, filename in filename_dict.items():
        validation_df, _, _, _, _, _, crs = load_belsar_validation_data(belsar_data_dir, filename)
        validation_df = validation_df[validation_df['date']==date]
        ids = pd.unique(validation_df["Field ID"]).tolist()
        sites_geometry = get_sites_geometry(belsar_data_dir, crs)
        sites_geometry = sites_geometry[sites_geometry['Name'].apply(lambda x: x in ids)]
        sites_geometry.reset_index(inplace=True, drop=True)
        delta_t = delta_dict[date]
        image_metrics = get_belsar_image_metrics(sites_geometry, validation_df, belsar_pred_dir, 
                                                 filename, file_suffix, date, delta_t,
                                                 NO_DATA=NO_DATA, get_error=get_error)
        metrics = pd.concat((metrics, image_metrics))
    return metrics.reset_index(drop=True)

def interpolate_belsar_metrics(belsar_data_dir, belsar_pred_dir, method="closest", file_suffix=""):
    
    if method == "simple_interpolate":
        before_metrics = get_belsar_campaign_metrics_df(belsar_data_dir, before_filename_dict, belsar_pred_dir, file_suffix)
        after_metrics = get_belsar_campaign_metrics_df(belsar_data_dir, after_filename_dict, belsar_pred_dir, file_suffix)
        metrics = before_metrics.copy()
        metrics.drop(columns=["delta"], inplace=True)
        for variable in ['lai_mean', 'cm_mean', 'lai_std', 'cm_std',
                         'lai_sigma_mean', 'cm_sigma_mean', 'lai_sigma_std', 'cm_sigma_std']:
            is_std = variable[-3:] == 'std'
            metrics[variable] = simple_interpolate(after_metrics[variable], before_metrics[variable], 
                                                        after_metrics['delta'], before_metrics['delta'], is_std=is_std)
        metrics['delta_before'] = before_metrics['delta']
        metrics['delta_after'] = after_metrics['delta']
        metrics["date"] = (abs(metrics['delta_after']) + abs(metrics['delta_before'])) / 2
    elif method == "closest":
        metrics = get_belsar_campaign_metrics_df(belsar_data_dir, closest_filename_dict, belsar_pred_dir, file_suffix)
    elif method == 'best':
        before_metrics = get_belsar_campaign_metrics_df(belsar_data_dir, before_filename_dict, belsar_pred_dir, file_suffix)
        after_metrics = get_belsar_campaign_metrics_df(belsar_data_dir, after_filename_dict, belsar_pred_dir, file_suffix)
        metrics = before_metrics.copy()
        metrics["date"] = abs(before_metrics['delta'])
        for i in range(len(metrics)):
            err_before = np.abs(metrics['lai_mean'].iloc[i] - metrics['ref_lai'].iloc[i])
            err_after = np.abs(after_metrics['lai_mean'].iloc[i] - after_metrics['lai_mean'].iloc[i])
            if err_after < err_before:
                metrics.iloc[i] = after_metrics.iloc[i]
                metrics["date"].iloc[i] = abs(metrics['delta'].iloc[i])
    elif method =="worst":
        before_metrics = get_belsar_campaign_metrics_df(belsar_data_dir, before_filename_dict, belsar_pred_dir, file_suffix)
        after_metrics = get_belsar_campaign_metrics_df(belsar_data_dir, after_filename_dict, belsar_pred_dir, file_suffix)
        metrics = before_metrics.copy()
        metrics["date"] = abs(before_metrics['delta'])
        for i in range(len(metrics)):
            err_before = np.abs(metrics['lai_mean'].iloc[i] - metrics['ref_lai'].iloc[i])
            err_after = np.abs(after_metrics['lai_mean'].iloc[i] - after_metrics['ref_lai'].iloc[i])
            if err_after > err_before:
                metrics.iloc[i] = after_metrics.iloc[i]
                metrics["date"].iloc[i] = abs(metrics['delta'].iloc[i])
    else:
        raise NotImplementedError
    return metrics


def save_belsar_predictions(belsar_dir, model, res_dir, list_filenames, model_name="pvae", mode="lat_mode",
                            save_reconstruction=False):
    NO_DATA = -10000
    for filename in tqdm(list_filenames):
        df, s2_r, s2_a, mask, xcoords, ycoords, crs = load_belsar_validation_data(belsar_dir, filename)
        s2_r = torch.from_numpy(s2_r).float()
        mask[mask==1.] = np.nan
        mask[mask==0.] = 1.
        if np.isnan(mask).all():
            print(f"No valid pixels in {filename}!")
        s2_r = (s2_r * torch.from_numpy(mask).float()).unsqueeze(0)
        s2_a = torch.from_numpy(s2_a).float().unsqueeze(0)
        
        with torch.no_grad():
            
            (rec, sim_image, s2_r, _, sigma_image) = get_encoded_image_from_batch((s2_r, s2_a), model,
                                                        patch_size=32, bands=torch.arange(10),
                                                        mode=mode, padding=True, no_rec=not save_reconstruction)
        

                # err_tensor[err_tensor.isnan()] = NO_DATA
                # tensor_to_raster(err_tensor, res_dir + f"/error_{filename}_{model_name}_{mode}.tif",
                #          crs=crs, resolution=10, dtype=np.float32, bounds=None,
                #          xcoords=xcoords, ycoords=ycoords, nodata=NO_DATA,
                #          hw = 0, half_res_coords=True)
        # lai_validation_pred = sim_image[6,...].unsqueeze(0)
        # cm_validation_pred = sim_image[5,...].unsqueeze(0)
        tensor = torch.cat((sim_image[6,...].unsqueeze(0),
                            sim_image[5,...].unsqueeze(0),
                            sigma_image[6,...].unsqueeze(0), 
                            sigma_image[5,...].unsqueeze(0)), 0)
        if save_reconstruction:
            err_tensor = (rec - s2_r.squeeze(0)).abs().mean(0, keepdim=True)
            tensor = torch.cat((tensor, err_tensor), 0)
        tensor[tensor.isnan()] = NO_DATA
        tensor_to_raster(tensor, res_dir + f"/{filename}_{model_name}_{mode}.tif",
                         crs=crs, resolution=10, dtype=np.float32, bounds=None,
                         xcoords=xcoords, ycoords=ycoords, nodata=NO_DATA,
                         hw = 0, half_res_coords=True)
    return


def save_snap_belsar_predictions(belsar_dir, res_dir, list_belsar_filename):
    NO_DATA = -10000
    # filename = "2A_20180613_FRM_Veg_Barrax_20180605"
    for filename in list_belsar_filename: 
        ver = "3A" if filename[:2] == "2A" else "3B"
        model_lai = SnapNN(ver=ver, variable="lai")
        model_lai.set_weiss_weights()

        df, s2_r_image, s2_a, mask, xcoords, ycoords, crs = load_belsar_validation_data(belsar_dir, filename)
        s2_r = torch.from_numpy(s2_r_image)[torch.tensor([1, 2, 3, 4, 5, 7, 8, 9]), ...].float()
        mask[mask==1.] = np.nan
        mask[mask==0.] = 1.
        if np.isnan(mask).all():
            print(f"No valid pixels in {filename}!")
        s2_r = s2_r * torch.from_numpy(mask).float()
        s2_a_permutated = torch.cos(torch.deg2rad(torch.from_numpy(np.concatenate((s2_a[1:2,...], 
                                                                                    s2_a[0:1,...], 
                                                                                    s2_a[2:,...]),0)))).float()
        
        s2_data = torch.concat((s2_r, s2_a_permutated), 0)
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

def get_all_belsar_predictions(belsar_data_dir, belsar_pred_dir, file_suffix, NO_DATA=-10000):
    metrics = pd.DataFrame()
    for date, filename in all_filename_dict.items():
        validation_df, _, _, _, _, _, crs = load_belsar_validation_data(belsar_data_dir, filename)
        ids = pd.unique(validation_df["Field ID"]).tolist()
        sites_geometry = get_sites_geometry(belsar_data_dir, crs)
        sites_geometry = sites_geometry[sites_geometry['Name'].apply(lambda x: x in ids)]
        sites_geometry.reset_index(inplace=True, drop=True)
        delta_t = 0
        image_metrics = get_belsar_image_metrics(sites_geometry, validation_df, belsar_pred_dir, 
                                                 filename, file_suffix, date, delta_t,
                                                 NO_DATA=NO_DATA, get_error=False)
        metrics = pd.concat((metrics, image_metrics))
    return metrics.reset_index(drop=True)

def main():
    """
    Preprocesses belsar data for valdiation
    """
    if socket.gethostname()=='CELL200973':
        args=["-f", "BelSAR_agriculture_database",
              "-d", "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/belSAR_validation/",
            #   "-p", "SENTINEL2B_20180516-105351-101_L2A_T30SWJ_D_V1-7"]
              "-p", "SENTINEL2A_20180518-104024-461_L2A_T31UFS_C_V2-2"]
        # "SENTINEL2B_20180516-105351-101_L2A_T30SWJ_D_V1-7"
        parser = get_prosailvae_train_parser().parse_args(args)
    else:
        parser = get_prosailvae_train_parser().parse_args()
    # list_s2_products = ["SENTINEL2A_20180508-104025-460_L2A_T31UFS_D_V2-2.zip",
    #                     'SENTINEL2A_20180528-104613-414_L2A_T31UFS_D_V2-2.zip',
    #                     'SENTINEL2A_20180620-105211-086_L2A_T31UFS_D_V2-2.zip',
    #                     'SENTINEL2B_20180801-104018-457_L2A_T31UFS_D_V2-2.zip',
    #                     'SENTINEL2A_20180518-104024-461_L2A_T31UFS_D_V2-2.zip',
    #                     'SENTINEL2B_20180804-105022-459_L2A_T31UFS_D_V2-2.zip',
    #                     "SENTINEL2B_20180722-104020-458_L2A_T31UFS_D_V2-2.zip",
    #                     "SENTINEL2A_20180627-104023-457_L2A_T31UFS_D_V2-2.zip",
    #                     "SENTINEL2B_20180715-105300-591_L2A_T31UFS_D_V1-8.zip",
    #                     "SENTINEL2A_20180727-104023-458_L2A_T31UFS_D_V2-2.zip"]
    list_s2_products = ["SENTINEL2B_20180526-105635-059_L2A_T31UFS_D.zip",
                        "SENTINEL2A_20180521-105416-754_L2A_T31UFS_D.zip",
                        "SENTINEL2A_20180630-105440-000_L2A_T31UFS_D.zip",
                        "SENTINEL2A_20180528-104613-414_L2A_T31UFS_D.zip",
                        "SENTINEL2B_20180725-105415-357_L2A_T31UFS_D.zip",
                        "SENTINEL2B_20180702-104021-464_L2A_T31UFS_D.zip"]
    for s2_product in list_s2_products:
        if s2_product[-4:]=='.zip':
            with zipfile.ZipFile(os.path.join(parser.data_dir, s2_product), 'r') as zip_ref:
                zip_ref.extractall(parser.data_dir)
                s2_product = os.path.normpath(str(list(zipfile.Path(os.path.join(parser.data_dir, s2_product)).iterdir())[0])).split(os.sep)[-1]
            output_file_name = preprocess_belsar_validation_data(parser.data_dir, parser.data_filename, s2_product, crop="both")
            plot_belsar_site(parser.data_dir, output_file_name)
            shutil.rmtree(os.path.join(parser.data_dir, s2_product))
        else:
            output_file_name = preprocess_belsar_validation_data(parser.data_dir, parser.data_filename, s2_product, crop="both")
        print(output_file_name)
if __name__ == "__main__":
    main()