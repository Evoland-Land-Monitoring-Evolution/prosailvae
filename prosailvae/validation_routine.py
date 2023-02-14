#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales
"""

Example of how to read a raster and a vector file with the objective
get the in situ measurements for validate the results.

Inspired by :
https://geopandas.org/en/stable/gallery/geopandas_rasterio_sample.html
"""
# imports
import argparse
import json
from dataclasses import dataclass
import dateutil
from datetime import timedelta
import logging
from itertools import chain
import glob
import os
from pathlib import Path
from typing import Any, List, Union, Tuple, List

from sensorsio.sentinel2 import Sentinel2
from tqdm import tqdm
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio as rio

os.chdir("/home/uz/vinascj/src/prosailvae/prosailvae")

def get_parser() -> argparse.ArgumentParser:
    """
    Generate argument parser for CLI
    """

    arg_parser = argparse.ArgumentParser(
        os.path.basename(__file__),
        description="Create a time series validation dataset for prosailvae",
    )

    arg_parser.add_argument(
        "--loglevel",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Logger level (default: INFO. Should be one of "
        "(DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )

    arg_parser.add_argument(
        "--input_config",
        type=str,
        help="full path to the config file",
        required=True,
    )

    arg_parser.add_argument(
        "--export_path", type=str, help="path to store results", required=True
    )

    return arg_parser


@dataclass
class Config:
    """
    Dataclass for hold the dataset
    filenames
    """
    site: str
    raster: str
    vector: str

    def __post_init__(self) -> None:
        if not Path(self.raster).exists:
            raise Exception(f"The dataset {self.raster} do not exist!")
        if not Path(self.vector).exists:
            raise Exception(f"The dataset {self.vector} do not exist!")


def read_config_file(config_file : str) -> Config:
    """
    Func for retrieve the config file
    """
    # current_path = Path.cwd()
    # parent_path = current_path.parent
    # config_file = parent_path / "config" / "validation.json"
    config_dict = json.loads(Path(config_file).read_text())
    return Config(**config_dict)


def raster_time_serie(raster_path : Union[str, Path]
                      ) -> pd.DataFrame:
    """
    Scan the raster folder
    and create a dataframe with the filenames and
    the acquisitions date
    """
    raster_filenames = [filename for filename in
                        glob.glob(f"{raster_path}/SENTINEL2*.tif")
                        if not filename.endswith("old.tif")]
    ts_df = pd.DataFrame(raster_filenames, columns=["s2_filenames"])
    # get the acquisition date
    get_acquisition_date = lambda x : dateutil.parser.parse(
        x.split("/")[-1].split("_")[1].split("-")[0]
    )
    ts_df['s2_date'] = ts_df['s2_filenames'].apply(
        get_acquisition_date).apply(
            pd.to_datetime)
    return ts_df


def read_vector(vector_filename: str) -> gpd.GeoDataFrame:
    """
    Read the the vector data
    """
    # read geo data
    logging.info("reading vector file : %s ", vector_filename)
    vector_data = gpd.read_file(vector_filename)

    return vector_data


def get_pixel_value(raster_filename: str,
                    pixel_coords: gpd.GeoDataFrame,
                    colnames: List[str],
                    ) -> List[float]:
    """
    read a raster file in a given extension
    using rasterio functionallyties
    """
    # get coordinates
    coord_list = [(x,y) for
                  x,y in
                  zip(pixel_coords['geometry'].x,
                      pixel_coords['geometry'].y)]
    # extract pixel values
    with rio.open(raster_filename) as raster:
        pixel_values = [coords
                        for coords in raster.sample(coord_list)]
    # araay
    s2_array = np.array(pixel_values, dtype=np.float64)
    # set the dtypes
    s2_array[:,:10]= s2_array[:,:10].astype(np.float64)
    s2_array[:,10:14]= s2_array[:,10:14].astype(int)
    s2_array[:,:14]= s2_array[:,:14].astype(np.float64)
    # scale the reflectances
    s2_array[:,:10] = np.divide(s2_array[:,:10].astype(np.float64), 10_000)
    # masks
    s2_array[:,11] = s2_array[:,11].astype(np.uint8)
    s2_array[:,11] = np.logical_or(
        s2_array[:,11].astype(np.uint8), #s2_array_cloud,
        np.logical_or( s2_array[:,10].astype(np.uint8) ,
                       s2_array[:,12].astype(np.uint8)
        ))
    # compute the angles
    s2_array[:,14:18] = join_even_odd_s2_angles(s2_array[:,14:])
    # Form output array
    s2_data_out = np.zeros((s2_array.shape[0], 18))
    s2_data_out[:,:14] = s2_array[:,:14]
    s2_data_out[:,14:] = s2_array[:,14:18]
    # DataFrame for store the values
    out_df = pd.DataFrame(s2_data_out, columns=colnames)
    return out_df

def join_even_odd_s2_angles(s2_angles_data: np.array) -> np.array:
    """
    Compute the joining between angles
    between odd and even detectors in S2

    :param: s2_angles_data: [BxCxWxH]

    inspired on :
    https://framagit.org/jmichel-otb/sensorsio/-/blob/master/notebooks/sentinel2_angles.ipynb
    adapted from
    https://src.koda.cnrs.fr/mmdc/mmdc-singledate/-/blob/master/src/mmdc_singledate/datamodules/components/datamodule_utils.py
    """
    # s2_angles_data
    # C = [sun_zen, sun_az, even_zen, odd_zen, even_az, odd_az]

    (sun_zen, sun_az, even_zen, odd_zen, even_az, odd_az) = (
        s2_angles_data[:,0 ],
        s2_angles_data[:,1 ],
        s2_angles_data[:,2 ],
        s2_angles_data[:,3 ],
        s2_angles_data[:,4 ],
        s2_angles_data[:,5 ],
    )

    join_zen = even_zen.copy()
    join_zen[np.isnan(even_zen)] = odd_zen[np.isnan(even_zen)]
    join_az = even_az.copy()
    join_az[np.isnan(even_az)] = odd_az[np.isnan(even_az)]

    return np.concatenate(
        [
            sun_zen,
            sun_az,
            join_zen,
            join_az,
        ],
    ).reshape(s2_angles_data.shape[0],4)


def compute_time_deltas(vector_timestamp: Any,
                        raster_timestamp: List[Any],
                        ) -> List[Any]:
    """
    Compute the time difference between the field
    campaing and the satellite acquisition
    """
    time_delta = [vector_timestamp - raster_ts
                  for raster_ts in raster_timestamp.to_list()]
    return time_delta



def get_pixels(n_nearest, vector_ts, vector_ds, col_names):
    """
    Extract the pixel values and append the field values
    """
    # the criterium
    s2_pix_values = get_pixel_value(
        n_nearest[1]["s2_filenames"],
        vector_ds,
        col_names,
    )
    # append field and raster date
    s2_pix_values["s2_filenames"] = [n_nearest[1]["s2_filenames"] for
                                i in range(len(s2_pix_values))]
    s2_pix_values["time_delta"] = [n_nearest[1]["time_delta"] for
                                i in range(len(s2_pix_values))]
    s2_pix_values["s2_date"] = [n_nearest[1]["s2_date"] for
                                i in range(len(s2_pix_values))]
    values_np = np.concatenate((np.array(s2_pix_values), np.array(vector_ds)),
                            axis=1)
    values_df = pd.DataFrame(
        values_np,
        columns = list(
            *chain([list(s2_pix_values.columns) + (list(vector_ds.columns))])
        )
    )
    return values_df

def check_pixel_validity(values_df : gpd.GeoDataFrame,
                         ) -> gpd.GeoDataFrame:
    """
    Check if the pixel are valids
    """
    values_df['sat_mask'] = values_df['sat_mask'].apply(
        lambda x : int(x)).to_list()
    values_df['cloud_mask'] = values_df['cloud_mask'].apply(
        lambda x : int(x)).to_list()
    values_df['edge_mask'] = values_df['edge_mask'].apply(
        lambda x : int(x)).to_list()
    values_df['geophysical_mask'] = values_df['geophysical_mask'].apply(
        lambda x : int(x)).to_list()

    return values_df


def clean_france(vector_ds: gpd.GeoDataFrame,
                ) -> gpd.GeoDataFrame:
    """
    Normalize the dates
    """
    normalize_france_date = lambda x : dateutil.parser.parse(str(x))
    vector_ds['field_date'] = vector_ds['Date DHP'].apply(
        normalize_france_date).apply(
            pd.to_datetime)

    # delete duplicate columns
    vector_ds = vector_ds.drop(['Date DHP', 'Date DHP.1'], axis=1)

    return vector_ds

def clean_spain(vector_ds: gpd.GeoDataFrame,
                ) -> gpd.GeoDataFrame:
    """
    Normalize the dates
    """
    normalize_spain_date = lambda x : dateutil.parser.parse(str(x))
    vector_ds['field_date'] = vector_ds['f_datetime'].apply(
        normalize_spain_date).apply(
            pd.to_datetime)

    # delete duplicate columns
    vector_ds = vector_ds.drop(['f_datetime', 'Time'], axis=1)

    return vector_ds

def clean_italy(vector_ds: gpd.GeoDataFrame,
                ) -> gpd.GeoDataFrame:
    """
    Normalize the dates
    """
    normalize_italy_date = lambda x : dateutil.parser.parse(x)
    vector_ds['field_date'] = vector_ds['Date'].apply(
        normalize_italy_date).apply(
            pd.to_datetime)

    # delete duplicate columns
    vector_ds = vector_ds.drop(['Date'], axis=1)

    return vector_ds



def main(args : argparse.ArgumentParser):
    """
    Entry point
    """
    # read config
    config = read_config_file(args.input_config)
    logging.info(f"Reading config : {args.input_config}")
    logging.info(f"Vector Folder : {config.vector}")
    logging.info(f"Raster Folder : {config.raster}")
    # get the raster time serie
    raster_ts = raster_time_serie(config.raster)

    if config.site == "italy":
        # read geo data
        vector_data = read_vector(*glob.glob(f"{config.vector}/*.gpkg"))
        # clean the data
        vector_data = clean_italy(vector_data)

    if config.site == "france":
        # read geo data
        vector_data = read_vector(glob.glob(f"{config.vector}/*.gpkg")[1])
        # clean the data
        vector_data = clean_france(vector_data)

    if config.site == "spain":
        # read geo data
        vector_data = read_vector(*glob.glob(f"{config.vector}/*.gpkg"))
        # clean the data
        vector_data = clean_spain(vector_data)

    # init export dataframe
    col_names =  [i.value for i in Sentinel2.GROUP_10M + Sentinel2.GROUP_20M]
    aux_names = ['sat_mask', 'cloud_mask', 'edge_mask','geophysical_mask',
                 'sun_zen','sun_az','join_zen','join_az']

                 #'cos(sun_zen)', 'sin(sun_az)', 'cos(join_zen)', 'sin(join_az)', 'cos(join_az)', 'sin(join_az)']
    col_names.extend(aux_names)
    before_values_list = []
    after_values_list = []

    # iterate over the vector dates
    window_days = 25
    for vector_ts, vector_ds in tqdm(vector_data.groupby('field_date'), desc="Extracting Pixel Values :"):
        # get the closest dates
        time_delta = compute_time_deltas(
            vector_ts,
            raster_ts["s2_date"],
        )
        raster_ts["time_delta"] = time_delta
        days_before = vector_ts - timedelta(days=window_days)
        days_after = vector_ts + timedelta(days=window_days)

        raster_ts["time_delta_after"] = raster_ts["time_delta"][raster_ts["time_delta"] >= timedelta(days=0)]
        raster_ts["time_delta_before"] = raster_ts["time_delta"][raster_ts["time_delta"] < timedelta(days=0)]
        raster_ts["time_delta_before"] = raster_ts["time_delta_before"].abs()
        # select using a 20 days window
        n_closest = raster_ts.set_index(raster_ts["s2_date"])
        n_closest.index.names = ["s2_date_idx"]
        n_closest_before = n_closest.loc[
            (n_closest["s2_date"] >= days_before.strftime("%Y-%m-%d")) &
            (n_closest["s2_date"] < vector_ts.strftime("%Y-%m-%d"))
        ]
        n_closest_before.index.names = ["s2_date_idx"]
        n_closest_before = n_closest_before.sort_index(ascending=False)
        #
        #
        n_closest_after = n_closest.loc[
            (n_closest["s2_date"] >= vector_ts.strftime("%Y-%m-%d")) &
            (n_closest["s2_date"]  < days_after.strftime("%Y-%m-%d"))
        ]
        n_closest_after.index.names = ["s2_date_idx"]
        n_closest_after = n_closest_after.sort_index(ascending=True)

        # iterate over
        for idx_b, n_nearest_b in enumerate(n_closest_before.iterrows()):
            # get pixel values
            values_df_b = get_pixels(n_nearest_b, vector_ts, vector_ds, col_names)
            # check pixel validity
            val_check_b = check_pixel_validity(values_df_b)
            # iterate over the dataframe
            for cloud_idx_b, cloud_presence_b in values_df_b["cloud_mask"].iteritems():
                # print(f"cloud presence before : {cloud_presence_b}")
                if not bool(cloud_presence_b):
                    # print(f"cloud presence before : {cloud_presence_b} <<<<<------- ")
                    before_values_list.append(values_df_b.iloc[cloud_idx_b,:])
                    break
                else:
                    continue

        # iterate over
        for idx_a, n_nearest_a in enumerate(n_closest_after.iterrows()):
            # get pixel values
            values_df_a = get_pixels(n_nearest_a, vector_ts, vector_ds, col_names)
            # check pixel validity
            values_df_a = check_pixel_validity(values_df_a)

            for cloud_idx_a, cloud_presence_a in values_df_a["cloud_mask"].iteritems():
                # print(f"cloud presence after : {cloud_presence_a}")
                if not bool(cloud_presence_a):
                    # print(f"cloud presence after : {cloud_presence_a} ------>>>>")
                    after_values_list.append(values_df_a.iloc[cloud_idx_a,:])
                    break
                else:
                    continue


    # get the final value
    if len(after_values_list) :
        final_df_af = pd.concat(after_values_list, axis=1).T
        # remove special character
        final_df_af.columns = final_df_af.columns.str.replace(' ', '')
        final_df_af.columns = final_df_af.columns.str.replace(',', '')
        # reorder columns
        column_list_af = final_df_af.columns
        new_order_columns_af= list(column_list_af[-2:]) + list(column_list_af[:-2])
        final_df_af = final_df_af[new_order_columns_af]
        final_df_af.to_csv(
            f"{args.export_path}/after_{config.site}_{config.raster.split('/')[-2]}.csv",
                           index=False)
        logging.info(f"Exported to : {args.export_path}/after_{config.site}_{config.raster.split('/')[-2]}.csv")

    if len(before_values_list) :
        final_df_be = pd.concat(before_values_list, axis=1).T
        final_df_be.columns = final_df_be.columns.str.replace(' ', '')
        final_df_be.columns = final_df_be.columns.str.replace(',', '')
        # reorder columns
        column_list_be = final_df_be.columns
        new_order_columns_be = list(column_list_be[-2:]) + list(column_list_be[:-2])
        final_df_be = final_df_be[new_order_columns_be]

        final_df_be.to_csv(
            f"{args.export_path}/before_{config.site}_{config.raster.split('/')[-2]}.csv",
                           index=False)
        logging.info(f"Exported to : {args.export_path}/before_{config.site}.csv")


if __name__ == "__main__":
    # Parser arguments
    parser = get_parser()
    args = parser.parse_args()

    # Configure logging
    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level:{args.loglevel}")

    logging.basicConfig(
        level=numeric_level,
        datefmt="%y-%m-%d %H:%M:%S",
        format="%(asctime)s :: %(levelname)s :: %(message)s",
    )


    ### Entry Point
    main(args)
    logging.info("Export finish!")
