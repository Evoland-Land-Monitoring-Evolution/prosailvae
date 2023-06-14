import numpy as np
import torch
import os
from dataset.belsar_validation import load_belsar_validation_data, get_sites_geometry
import rasterio
from rasterio.mask import mask
import pandas as pd
import datetime

def get_delta_dict(filename_dict):
    delta_dict = {}
    for date, filename in filename_dict.items():
        filename_date_str = filename[3:11]
        delta = (datetime.datetime.strptime(date, "%Y-%m-%d")
                 - datetime.datetime.strptime(filename_date_str, "%Y%m%d")).days
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
# closest_delta_dict = {"2018-05-17" : 1,
#                       '2018-05-18' : 0,
#                       '2018-05-31' : -3,
#                       '2018-06-01' : -4,
#                       '2018-06-05' : -8,
#                       '2018-06-21' : -1,
#                       '2018-06-22' : -2,
#                       '2018-07-19' : -4,
#                       '2018-08-02' : 2,
                                #  '2018-08-29'
                                #  }

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

# before_delta_dict = {"2018-05-17" : -9,
#                      '2018-05-18' : 0,
#                      '2018-05-31' : -3,
#                      '2018-06-01' : -4,
#                      '2018-06-05' : -8,
#                      '2018-06-21' : -1,
#                      '2018-06-22' : -2,
#                      '2018-07-19' : -4,
#                      '2018-08-02' : -6,
#                                 #  '2018-08-29'
#                                  }

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

# after_delta_dict = {"2018-05-17" : 1,
#                     '2018-05-18' : 0,
#                     '2018-05-31' : 20,
#                     '2018-06-01' : 19,
#                     '2018-06-05' : 15,
#                     '2018-06-21' : 6,
#                     '2018-06-22' : 5,
#                     '2018-07-19' : 3,
#                     '2018-08-02' : 2,
#                                 #  '2018-08-29'
#                     }

def compute_metrics_at_date(belsar_dir, res_dir, method="closest", file_suffix=""):
    
    NO_DATA=-10000
    metrics = pd.DataFrame()
    before_delta_dict = get_delta_dict(before_filename_dict)
    after_delta_dict = get_delta_dict(after_filename_dict)
    if method == "interpolate":
        for date, filename_before in before_filename_dict.items():
            
            filename_after = after_filename_dict[date]
            df_before, _, _, _, _, _, crs_before = load_belsar_validation_data(belsar_dir, filename_before)
            df_after, _, _, _, _, _, crs_after = load_belsar_validation_data(belsar_dir, filename_after)
            df_before = df_before[df_before['date']==date]
            df_after = df_after[df_after['date']==date]
            ids = pd.unique(df_after["Field ID"]).tolist()
            sites_geometry = get_sites_geometry(belsar_dir, crs_before)
            sites_geometry = sites_geometry[sites_geometry['Name'].apply(lambda x: x in ids)]
            sites_geometry.reset_index(inplace=True, drop=True)
            delta_before = before_delta_dict[date]
            delta_after = after_delta_dict[date]
            for i in range(len(sites_geometry)):
                line = sites_geometry.iloc[i]
                site_name  = line['Name']
                polygon = line['geometry']
                with rasterio.open(os.path.join(res_dir, filename_before + f"{file_suffix}.tif"), mode = 'r') as src:
                    masked_array_before, _ = mask(src, [polygon], invert=False)
                    masked_array_before[masked_array_before==NO_DATA] = np.nan
                with rasterio.open(os.path.join(res_dir, filename_after + f"{file_suffix}.tif"), mode = 'r') as src:
                    masked_array_after, _ = mask(src, [polygon], invert=False)
                    masked_array_after[masked_array_after==NO_DATA] = np.nan
                if delta_after != 0:
                    m = (masked_array_after - masked_array_before) / (-delta_after - delta_before)
                    b = masked_array_after - m * (-delta_after -  delta_before)
                    masked_array = m * delta_before + b
                else:
                    masked_array = masked_array_after
                site_samples = df_before[df_before["Field ID"]==site_name]
                site_lai = site_samples["lai"]
                site_cm = site_samples["cm"]
                lai_mean = np.mean(site_lai)
                lai_std = np.std(site_lai)
                cm_mean = np.mean(site_cm)
                cm_std = np.std(site_cm)

                parcel_lai_mean = np.nan
                parcel_lai_std = np.nan
                parcel_cm_mean = np.nan
                parcel_cm_std = np.nan
                parcel_lai_sigma_mean = np.nan
                parcel_lai_sigma_std = np.nan
                parcel_cm_sigma_mean = np.nan
                parcel_cm_sigma_std = np.nan
                if not np.isnan(masked_array[0,...]).all():
                    parcel_lai_mean = np.nanmean(masked_array[0,...])
                if not np.isnan(masked_array[0,...]).all():
                    parcel_lai_std = np.nanstd(masked_array[0,...])
                if not np.isnan(masked_array[1,...]).all():
                    parcel_cm_mean = np.nanmean(masked_array[1,...])
                if not np.isnan(masked_array[1,...]).all():
                    parcel_cm_std = np.nanstd(masked_array[1,...])
                if not np.isnan(masked_array[2,...]).all():
                    parcel_lai_sigma_mean = np.nanmean(masked_array[2,...])
                if not np.isnan(masked_array[2,...]).all():
                    parcel_lai_sigma_std = np.nanstd(masked_array[2,...])
                if not np.isnan(masked_array[3,...]).all():
                    parcel_cm_sigma_mean = np.nanmean(masked_array[3,...])
                if not np.isnan(masked_array[3,...]).all():
                    parcel_cm_sigma_std = np.nanstd(masked_array[3,...])

                d = {"name" : site_name,
                    "date" : date,
                    "delta_before": delta_before,
                    "delta_after": delta_after,
                    "lai_mean" : lai_mean,
                    "lai_std" : lai_std, 
                    "cm_mean" : cm_mean,
                    "cm_std" : cm_std, 
                    "parcel_lai_mean" : parcel_lai_mean,
                    "parcel_lai_std" : parcel_lai_std,
                    "parcel_cm_mean" : parcel_cm_mean,
                    "parcel_cm_std" : parcel_cm_std,
                    "parcel_lai_sigma_mean" : parcel_lai_sigma_mean,
                    "parcel_lai_sigma_std" : parcel_lai_sigma_std,
                    "parcel_cm_sigma_mean" : parcel_cm_sigma_mean,
                    "parcel_cm_sigma_std" : parcel_cm_sigma_std
                        }
                metrics = pd.concat((metrics, pd.DataFrame(d, index=[0])))

    elif method == "closest":
        closest_delta_dict = get_delta_dict(closest_filename_dict)
        for date, filename in closest_filename_dict.items():
            df, s2_r_image, s2_a, valid_mask, xcoords, ycoords, crs = load_belsar_validation_data(belsar_dir, filename)
            df = df[df['date']==date]
            ids = pd.unique(df["Field ID"]).tolist()
            sites_geometry = get_sites_geometry(belsar_dir, crs)
            sites_geometry = sites_geometry[sites_geometry['Name'].apply(lambda x: x in ids)]
            sites_geometry.reset_index(inplace=True, drop=True)
            delta = closest_delta_dict[date]
            for i in range(len(sites_geometry)):
                line = sites_geometry.iloc[i]
                site_name  = line['Name']
                polygon = line['geometry']
                with rasterio.open(os.path.join(res_dir, filename + f"{file_suffix}.tif"), mode = 'r') as src:
                    masked_array, _ = mask(src, [polygon], invert=False)
                    masked_array[masked_array==NO_DATA] = np.nan
                if np.isnan(masked_array).all():
                    print(f"No prediction available for site {site_name} at date {date} (file {filename})!")
                    continue

                site_samples = df[df["Field ID"]==site_name]
                site_lai = site_samples["lai"]
                site_cm = site_samples["cm"]
                lai_mean = np.mean(site_lai)
                lai_std = np.std(site_lai)
                cm_mean = np.mean(site_cm)
                cm_std = np.std(site_cm)
                parcel_lai_mean = np.nan
                parcel_lai_std = np.nan
                parcel_cm_mean = np.nan
                parcel_cm_std = np.nan
                parcel_lai_sigma_mean = np.nan
                parcel_lai_sigma_std = np.nan
                parcel_cm_sigma_mean = np.nan
                parcel_cm_sigma_std = np.nan
                if not np.isnan(masked_array[0,...]).all():
                    parcel_lai_mean = np.nanmean(masked_array[0,...])
                if not np.isnan(masked_array[0,...]).all():
                    parcel_lai_std = np.nanstd(masked_array[0,...])
                if not np.isnan(masked_array[1,...]).all():
                    parcel_cm_mean = np.nanmean(masked_array[1,...])
                if not np.isnan(masked_array[1,...]).all():
                    parcel_cm_std = np.nanstd(masked_array[1,...])
                if not np.isnan(masked_array[2,...]).all():
                    parcel_lai_sigma_mean = np.nanmean(masked_array[2,...])
                if not np.isnan(masked_array[2,...]).all():
                    parcel_lai_sigma_std = np.nanstd(masked_array[2,...])
                if not np.isnan(masked_array[3,...]).all():
                    parcel_cm_sigma_mean = np.nanmean(masked_array[3,...])
                if not np.isnan(masked_array[3,...]).all():
                    parcel_cm_sigma_std = np.nanstd(masked_array[3,...])

                d = {"name" : site_name,
                    "date" : date,
                    "delta": delta,
                    "lai_mean" : lai_mean,
                    "lai_std" : lai_std, 
                    "cm_mean" : cm_mean,
                    "cm_std" : cm_std, 
                    "parcel_lai_mean" : parcel_lai_mean,
                    "parcel_lai_std" : parcel_lai_std,
                    "parcel_cm_mean" : parcel_cm_mean,
                    "parcel_cm_std" : parcel_cm_std,
                    "parcel_lai_sigma_mean" : parcel_lai_sigma_mean,
                    "parcel_lai_sigma_std" : parcel_lai_sigma_std,
                    "parcel_cm_sigma_mean" : parcel_cm_sigma_mean,
                    "parcel_cm_sigma_std" : parcel_cm_sigma_std
                        }
                metrics = pd.concat((metrics, pd.DataFrame(d, index=[0]).loc[:])).reset_index(drop=True)
    else:
        raise NotImplementedError
    return metrics