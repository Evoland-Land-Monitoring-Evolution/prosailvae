#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 17:21:00 2022

@author: yoel
"""
import time
from datetime import datetime
import sqlite3
import pandas as pd
import numpy as np
import torch
MASK_VALID = 0
MASK_INVALID = 1
MASK_NODATA = 2
MASK_NEG_REFL = 3
MASK_UPSAMPLED = 4
import phenovae
import os
import argparse
from tqdm import tqdm
import time

def save_feat_columns(database_file, out_file, bands=['b4', 'b8']):
    """
    Save features (specified spectral bands) related columns names of the table of the sql dataset.

    Parameters
    ----------
    database_file : TYPE
        DESCRIPTION.
    out_file : TYPE
        DESCRIPTION.
    bands : TYPE, optional
        DESCRIPTION. The default is ['b4', 'b8'].

    Returns
    -------
    None.

    """
    conn = sqlite3.connect(database_file)
    cursor = conn.execute('select * from output')
    names = [description[0] for description in cursor.description]
    feats_raw = []
    ext_bands = [b+"_" for b in bands]
    for i in range(len(names)):
        if names[i][:11]=='sentinel2_b':
            if (names[i][10:13] in ext_bands) or (names[i][10:14] in ext_bands):
                feats_raw.append(names[i])
    file = open(out_file, 'w+')
    for i in range(len(feats_raw)):
        file.write(feats_raw[i]+"\n")
    file.close()
    
def load_feat_columns(feat_file, max_try=5):
    """
    Load features (specified spectral bands) related columns names of the table of the sql dataset.

    """
    file = open(feat_file, "r")
    lines = file.read().splitlines()
    i=0
    while len(lines)==0 and i < max_try:
        file.close()
        time.sleep(0.1)
        file = open(feat_file, "r")
        lines = file.read().splitlines()
        i+=1
    feats = []
    for line in lines:
        feats.append(line)
    file.close()
    return feats

def save_mask_columns(database_file, out_file):
    """
    Save mask-related columns names of the table of the sql dataset
    Parameters
    ----------
    database_file : TYPE
        DESCRIPTION.
    out_file : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    conn = sqlite3.connect(database_file)
    cursor = conn.execute('select * from output')
    names = [description[0] for description in cursor.description]
    feats_raw = []
    for i in range(len(names)):
        if names[i][9:15]=='_mask_':
            feats_raw.append(names[i])
    file = open(out_file, 'w+')
    for i in range(len(feats_raw)):
        file.write(feats_raw[i]+"\n")
    file.close()
    
def load_masks_columns(mask_file, max_try=5):
    """
    Load mask-related columns names of the table of the sql dataset

    Parameters
    ----------
    mask_file : TYPE
        DESCRIPTION.
    max_try : TYPE, optional
        DESCRIPTION. The default is 5.

    Returns
    -------
    feats : TYPE
        DESCRIPTION.

    """
    file = open(mask_file, "r")
    lines = file.read().splitlines()
    i=0
    while len(lines)==0 and i < max_try:
        file.close()
        time.sleep(0.1)
        file = open(mask_file, "r")
        lines = file.read().splitlines()
        i+=1
    feats = []
    for line in lines:
        feats.append(line)
    return feats

def doy_by_sensors(features_labels, logger):
    """for each sensor's features return the number of days from the first feature

    Example
    -------
    features_labels = ["sentinel2_b1_20150101", "sentinel2_b1_20150110"]
    doy = doy_by_sensors(features_labels)
    >>> doy["doy"] = [0, 9]
    >>> doy["features_per_date"] = 2
    """
    sensors_dates = {}
    sensors_doy = {}
    for feature_labels in features_labels:
        try:
            sensor_name, feat, date_time_str = feature_labels.split("_")
            if sensor_name not in sensors_dates:
                sensors_dates[sensor_name] = {
                    "dates": [],
                    "features_per_date": []
                }
            date_time_obj = datetime.strptime(date_time_str, "%Y%m%d")
            if date_time_obj not in sensors_dates[sensor_name]["dates"]:
                sensors_dates[sensor_name]["dates"].append(date_time_obj)
            nb_component = len([
                elem for elem in features_labels
                if sensor_name in elem and date_time_str in elem
            ])
            sensors_dates[sensor_name]["features_per_date"].append(
                nb_component)
        except ValueError:
            logger.warning(f"{feature_labels} cannot be converted to DOY")
    sensors_dates_sorted = {}
    for sensor_name, _ in sensors_dates.items():
        sensors_dates_sorted[sensor_name] = {}
        sensors_dates_sorted[sensor_name]["dates"] = sorted(
            sensors_dates[sensor_name]["dates"])
        nb_components = sensors_dates[sensor_name]["features_per_date"]
        if nb_components.count(nb_components[0]) != len(nb_components):
            raise ValueError(
                f"There is not the same number of features per date in the sensor {sensor_name}"
            )
        sensors_dates_sorted[sensor_name]["features_per_date"] = sensors_dates[
            sensor_name]["features_per_date"][0]

    for sensor_name, dic in sensors_dates_sorted.items():
        first_date = dic["dates"][0]
        delta_days = [(date - first_date).days for date in dic["dates"]]
        sensors_doy[sensor_name] = {
            "doy":
            delta_days,
            "features_per_date":
            sensors_dates_sorted[sensor_name]["features_per_date"]
        }
    return sensors_doy

def get_only_mask(masks_columns, table_name, labels, conn, n_samples, 
                  field='point_id', field_value_list=None):
    if field_value_list is not None:
        df_masks = pd.read_sql_query(
            "SELECT {} from {} WHERE {} IN ({}) AND i2label IN ({}) LIMIT {}".format(
                ",".join(masks_columns), table_name, field,
                ','.join(map(str, field_value_list)), 
                ','.join(map(str,labels)),
                str(n_samples)), conn)
        df_masks = df_masks.fillna(MASK_NODATA)
    else:
        df_masks = pd.read_sql_query(
            "SELECT {} from {} WHERE i2label IN ({}) LIMIT {}".format(
                ",".join(masks_columns), table_name, 
                ','.join(map(str,labels)),
                str(n_samples)), conn)
        df_masks = df_masks.fillna(MASK_NODATA)
    return df_masks

def get_feats_from_sql(features_columns, table_name, fids, labels_column, 
                       labels, conn, n_samples):
    list_df_feats = []
    if fids is not None:
        for chunk in tqdm(pd.read_sql(
            f"SELECT {','.join(features_columns)} FROM {table_name} "
            f"WHERE originfid IN ({','.join(map(str,fids))}) "
            f"AND i2label IN ({','.join(map(str,labels))})"
            f" LIMIT {str(n_samples)}", con=conn, chunksize=10000)):
            list_df_feats.append(chunk)

        df_labs = pd.read_sql_query(
            f"SELECT {labels_column} FROM {table_name} "
            f"WHERE originfid IN ({','.join(map(str,fids))}) "
            f"AND i2label IN ({','.join(map(str,labels))})"
            f" LIMIT {str(n_samples)}", conn)
    else:
        for chunk in tqdm(pd.read_sql(
            f"select {','.join(features_columns)} from {table_name} "
            f"WHERE i2label IN ({','.join(map(str,labels))})"
            f" LIMIT {str(n_samples)}", con=conn, chunksize=10000)):
            list_df_feats.append(chunk)
        df_labs = pd.read_sql_query(
            f"select {labels_column} from {table_name} "
            f"WHERE i2label IN ({','.join(map(str,labels))})"
            f" LIMIT {str(n_samples)}", conn)
    df_feats = pd.concat(list_df_feats, ignore_index=True)
    return df_feats, df_labs

def saturate_negative_reflectances(df_feats, sat=0.0001):
    arr_feats_sat = df_feats.values.copy()
    arr_feats_sat[arr_feats_sat<=-1000]=np.nan
    arr_feats_sat[arr_feats_sat<=0]=sat
    df_feats_sat = pd.DataFrame(arr_feats_sat)
    return df_feats_sat

def upsample_mask(df_mask, doys, len_signal=365, binary=True):

    n_samples = len(df_mask)
    arr_upsample_mask = np.ones((n_samples, len_signal)) * MASK_UPSAMPLED
    for i in range(len_signal):
        if i in doys:
            day_idx = doys.index(i)
            current_date_mask = df_mask[df_mask.columns[day_idx]].copy()
            if binary: # to get binary mask
                current_date_mask[current_date_mask.ne(0)] = 1.0 
            arr_upsample_mask[:,i] = current_date_mask.values

    df_upsample_mask = pd.DataFrame(arr_upsample_mask)      
    return df_upsample_mask

def comp_ndvi(rband, irband, eps=1e-3):
    return (irband - rband) / (irband + rband + eps)

def get_ndvi_bands_from_df_feats(df_feats, doys, len_signal=365):
    arr_rband = np.ones((len(df_feats),len_signal))
    arr_irband = np.ones((len(df_feats),len_signal))
    for i in range(len_signal): # for each date (in 365 days)
        if i in doys: # if date is is among dates of mask/features
            day_idx = doys.index(i)
            arr_rband[:,i] = df_feats[df_feats.columns[2*day_idx]].values # Warning : hardcoded features position
            arr_irband[:,i] = df_feats[df_feats.columns[2*day_idx+1]].values 
    return arr_rband, arr_irband


def bands2ndvi(rband, irband, mask):
    ndvi_mask = mask.copy()
    # making binary mask
    ndvi_mask.replace(to_replace=MASK_INVALID, value=np.nan, inplace=True)
    ndvi_mask.replace(to_replace=MASK_NODATA, value=np.nan, inplace=True)
    ndvi_mask.replace(to_replace=MASK_NEG_REFL, value=np.nan, inplace=True)
    ndvi_mask.replace(to_replace=MASK_UPSAMPLED, value=np.nan, inplace=True)
    ndvi_mask.replace(to_replace=MASK_VALID, value=1.0, inplace=True)
    arr_ndvi = comp_ndvi(rband, irband) * ndvi_mask.values
    df_ndvi = pd.DataFrame(data=arr_ndvi)
    df_ndvi = df_ndvi.interpolate(axis=1).bfill(axis=1)
    return df_ndvi
    

def remove_rows_from_dataset(rows, df_ndvi, df_mask, df_labs):
    df_mask = df_mask[rows==False].reset_index(drop=True)
    df_labs = df_labs[rows==False].reset_index(drop=True)
    df_ndvi = df_ndvi[rows==False].reset_index(drop=True)
    return df_ndvi, df_mask, df_labs

def get_s2_dataset_size(table_name, labels, conn):
    n_samples = pd.read_sql_query(
        f"SELECT COUNT(*) from {table_name} "
        f"WHERE i2label IN ({','.join(map(str,labels))})", conn).values[0,0]
    return n_samples

def get_S2_data_preprocess_parser():
    """
    Creates a new argument parser.
    """
    parser = argparse.ArgumentParser(description='Parser for data generation')
    
    parser.add_argument("-label_type", "-l", dest="label_type",
                        help="group of labels to be included in the dataset",
                        type=str, default="all")
    parser.add_argument("-n_samples", "-n", dest="n_samples",
                        help="Number of samples to be included in the dataset",
                        type=int, default=10000)
    parser.add_argument("-time_sampling", "-t", dest="time_sampling",
                        help="Temporal sampling step",
                        type=int, default=5)
    return parser

if __name__ == "__main__":
    t0 = time.time()
    data_dir = os.path.join(os.path.join(os.path.dirname(phenovae.__file__),
                                         os.pardir),"data/")
    parser = get_S2_data_preprocess_parser().parse_args()
    datafile = data_dir +"s2_data.sqlite"
    if parser.label_type=="all":
        labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '12', '13', 
                  '14', '15', '16', '17', '18', '19', '20', '23']
    else:
        raise NotImplementedError("Only 'all' label types supported.")
        
    feats_file = data_dir + "/sentinel2_feats.txt"
    mask_file = data_dir + "/sentinel2_mask.txt"
    labels_column='i2label'
    table_name = 'output'
    fids=None
    select_dates_delta = parser.time_sampling
    if not os.path.isfile(feats_file):
        save_feat_columns(datafile, feats_file, bands=['b4', 'b8'])
    features_columns = load_feat_columns(feats_file) # loading columns names of acquistions to compute ndvi
    if not os.path.isfile(mask_file):
        save_mask_columns(datafile, mask_file)
    masks_columns = load_masks_columns(mask_file) # loading mask columns names 
    doys = doy_by_sensors(features_columns, None)['sentinel2']['doy']
    conn = sqlite3.connect(datafile)
    
    n_samples = get_s2_dataset_size(table_name, labels, conn)
    n_samples = np.min([n_samples, parser.n_samples])
    print(f"{n_samples} samples will be loaded.")
    print("Loading masks...")
    df_masks = get_only_mask(masks_columns, table_name, labels, conn, 
                             n_samples, field='point_id', field_value_list=None)
    print("Mask Loaded !")
    # load features from dataset
    print("Loading labels and time series features...")
    df_feats, df_labs = get_feats_from_sql(features_columns, 
                table_name, fids, labels_column, labels, conn, n_samples)
    conn.close() 
    print("Labels and time series features Loaded !")
    print("Computing NDVI time series...")
    df_feats.fillna(np.nan, inplace=True) # Remove None
    df_feats = saturate_negative_reflectances(df_feats, sat=1.0) 
    df_upsampled_masks = upsample_mask(df_masks, doys)
    arr_rband, arr_irband = get_ndvi_bands_from_df_feats(df_feats, doys)
    df_ndvi = bands2ndvi(arr_rband, arr_irband, df_upsampled_masks)
    del df_feats
    print("Verifying time series...")
    nan_ndvi_rows = df_ndvi.isna().any(axis=1)
    print("Number of NaN time series : {}".format(nan_ndvi_rows.sum()))
    (df_ndvi, df_masks, 
     df_labs) = remove_rows_from_dataset(nan_ndvi_rows, df_ndvi, df_masks, df_labs) # removes nan rows
    null_rows = (df_ndvi.sum(axis=1)==0).copy() # Finds all null rows
    print("Number of null time series : {}".format(null_rows.sum()))
    (df_ndvi, df_masks, 
     df_labs) = remove_rows_from_dataset(null_rows, df_ndvi, df_masks, 
                                                         df_labs)   # removes all null rows
    n_samples = len(df_ndvi)                                               
    len_signal = df_ndvi.values.shape[1]
    print(f"Getting NDVI values for temporal grid with {select_dates_delta} days step...")
    selected_dates = np.arange(0, len_signal, select_dates_delta).reshape(1,-1)
    selected_ndvi = torch.from_numpy(df_ndvi.values[:, selected_dates]).reshape(n_samples,-1)
    df_s2_binary_mask = df_masks.copy()
    df_s2_binary_mask[df_s2_binary_mask.ne(0)] = 1.0 
    s2_binary_mask = torch.from_numpy(df_s2_binary_mask.values)
    s2labels = torch.from_numpy(df_labs.values).view(-1,1)
    print("Saving NDVI time series...")
    torch.save(selected_ndvi, data_dir + "/s2_ndvi_dataset.pt")
    torch.save(torch.tensor(doys), data_dir + "/s2_mask_doys.pt")
    torch.save(s2labels, data_dir + "/s2_labels.pt")
    torch.save(s2_binary_mask, data_dir + "/s2_binary_mask.pt")
    t1 = time.time()
    print(f"Sentinel-2 dataset preprocessing completed ({int(t1-t0)} s) !")
    