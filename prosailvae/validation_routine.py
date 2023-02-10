#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright: (c) 2022 CESBIO / Centre National d'Etudes Spatiales
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
    vector_field : str

    def __post_init__(self) -> None:
        if not Path(self.raster).exists:
            raise Exception(f"The dataset {self.raster} do not exist!")
        if not Path(self.vector).exists:
            raise Exception(f"The dataset {self.vector} do not exist!")


def read_config_file() -> Config:
    """
    Func for retrieve the config file
    """
    current_path = Path.cwd()
    parent_path = current_path.parent
    config_file = parent_path / "config" / "validation.json"
    config_dict = json.loads(config_file.read_text())
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
    s2_array = np.array(pixel_values)
    # scale the reflectances
    s2_array[:,:10] = s2_array[:,:10] #/ 10_000
    # masks
    s2_array_cloud = s2_array[:,11][s2_array[:,11] > 0 ] = 1
    s2_array[:,11] = np.logical_or(
        s2_array_cloud,
        np.logical_or( s2_array[:,10],
                       s2_array[:,12]
        ))
    # compute the angles
    s2_array[:,14:] = join_even_odd_s2_angles(s2_array[:,14:])
    # DataFrame for store the values
    out_df = pd.DataFrame(s2_array, columns=colnames)

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
            np.cos(np.deg2rad(sun_zen)),
            np.cos(np.deg2rad(sun_az)),
            np.sin(np.deg2rad(sun_az)),
            np.cos(np.deg2rad(join_zen)),
            np.cos(np.deg2rad(join_az)),
            np.sin(np.deg2rad(join_az)),
        ],
    ).reshape(s2_angles_data.shape)


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


def clean_italy(vector_ds: gpd.GeoDataFrame
                ) -> gpd.GeoDataFrame:
    """
    Normalize the dates
    """
    normalize_italy_date = lambda x : dateutil.parser.parse(x)
    vector_ds['Date'] = vector_ds['Date'].apply(
        normalize_italy_date).apply(
            pd.to_datetime)

    return vector_ds

def clean_france(vector_ds: gpd.GeoDataFrame
                ) -> gpd.GeoDataFrame:
    """
    Normalize the dates
    """
    normalize_france_date = lambda x : dateutil.parser.parse(str(x))
    vector_ds['Date'] = vector_ds['Date DHP'].apply(
        normalize_france_date).apply(
            pd.to_datetime)

    return vector_ds

def clean_spain(vector_ds: gpd.GeoDataFrame
                ) -> gpd.GeoDataFrame:
    """
    Normalize the dates
    """
    normalize_spain_date = lambda x : dateutil.parser.parse(str(x))
    vector_ds['Date'] = vector_ds['Date DHP'].apply(
        normalize_france_date).apply(
            pd.to_datetime)

    return vector_ds


def main():
    """
    Entry point
    """
    # read config
    config = read_config_file()

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


    # init export dataframe
    col_names = ["B"+str(i+1) for i in range(10)]

    aux_names = ['sat_mask', 'cloud_mask', 'edge_mask','geophysical_mask',
                 'sun_zen', 'sun_az', 'even_zen', 'odd_zen', 'even_az', 'odd_az']
    col_names.extend(aux_names)
    values_list = []

    # iterate over the vector dates
    #  vector_ts, vector_ds = next(iter(vector_data.groupby("Date")))
    for vector_ts, vector_ds in vector_data.groupby("Date"):
        # get the closest dates
        time_delta = compute_time_deltas(
            vector_ts,
            raster_ts["s2_date"]
        )
        raster_ts["time_delta"] = time_delta
        raster_ts["time_delta_after"] = raster_ts["time_delta"][raster_ts["time_delta"] >= timedelta()]
        raster_ts["time_delta_before"] = raster_ts["time_delta"][raster_ts["time_delta"] < timedelta()]

        #n_closest = raster_ts.sort_values(by="time_delta").iloc[:3, :]
        # n_nearest = next(n_closest.iterrows())
        # search for the nex valid data
        #for idx, n_nearest in enumerate(n_closest.iterrows()):
        for idx, n_nearest in enumerate(raster_ts.iterrows()):
            # get the pixel values for those how match
            # the criterium
            s2_pix_values = get_pixel_value(
                n_nearest[1]["s2_filenames"],
                vector_ds,
                col_names
            )
            # append field and raster date
            s2_pix_values["s2_filenames"] = [n_nearest[1]["s2_filenames"] for
                                        i in range(len(s2_pix_values))]
            s2_pix_values["time_delta"] = [n_nearest[1]["time_delta"] for
                                        i in range(len(s2_pix_values))]
            s2_pix_values["s2_date"] = [n_nearest[1]["s2_date"] for
                                        i in range(len(s2_pix_values))]
            s2_pix_values["field_date"] = [vector_ts for
                                           i in range(len(s2_pix_values))]
            values_np = np.concatenate((np.array(s2_pix_values), np.array(vector_ds)),
                                  axis=1)
            values_df = pd.DataFrame(values_np,
                                     columns = list(
                                         *chain([list(s2_pix_values.columns) + (list(vector_ds.columns))]))
            values_list.append(values_df)
    # get the final value
    final_df = pd.concat(values_list)
    # export file
    # final_df.to_csv(f"{args.export_path}/{config.site}.csv",
    final_df.to_csv(f"/work/scratch/vinascj/withoutfilter{config.site}.csv",
                    index=False)


if __name__ == "__main__":
    # Parser arguments
    parser = get_parser()
    args = parser.parse_args()

    # check patch selection strategy
    if not (args.patch_index or args.nb_patches):
        parser.error(
            "Not patch indexing strategy requested,"
            " add --patch_index or --nb_patches"
        )

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
    os.chdir("/home/uz/vinascj/src/prosailvae/prosailvae")
    # call main
    main(args)
    logging.info("Export finish!")
