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
import logging
import glob
import os
from pathlib import Path
from typing import Any, List, Union

import pandas as pd
import geopandas as gpd
import rasterio as rio


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
                    pixel_coords: Tuple[float,float]) -> List[float]:
    """
    read a raster file in a given extension
    using rasterio functionallyties
    """
    with rio.open(raster_filename) as raster:
        pixel_values = raster.sample(pixel_coords)
    return pixel_values


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


def compute_time_deltas(vector_timestamp: Any,
                        raster_timestamp: List[Any],
                        nb_acquisitions: int,
                        ) -> List[Any]:
    """
    Compute the time difference between the field
    campaing and the satellite acquisition
    """
    time_delta = [abs(vector_timestamp - raster_ts)
                  for raster_ts in raster_timestamp]
    return time_delta.sort()[nb_acquisitions]



def main():
    """
    Entry point
    """
    # read config
    config = read_config_file()

    # get the raster time serie
    raster_ts = raster_time_serie(config.raster)

    # read geo data
    vector_data = read_vector(*glob.glob(f"{config.vector}/*.gpkg"))

    # clean the data
    if config.site == "italy":
        vector_data = clean_italy(vector_data)

    # iterate over the vector dates
    for vector_ds in vector_data.iterrows():
        # get the n closest dates
        vector_ds_date = dict(vector_ds[1])["Date"]
        n_closest_dates = compute_time_deltas(
            vector_ds_date,
            raster_ts["s2_date"],
            3
        )
        # select the raster img with the closest
        # acquistions dates

        # get the pixel values for those how match
        # the criterium
        ajs = get_pixel_value(
            tuple()[0],
            {vector_data["geometry"].x,
             vector_data["geometry"].y}
        )

        #df.loc[df.timestamp == nearest(df.timestamp.to_list(),dt)].index[0]


    # raster_data = read_raster(raster_dataset_path)

    # # extract pixel values
    # coord_list = [(x,y) for x,y
    #               in zip(vector_data['geometry'].x,
    #                      vector_data['geometry'].y)]

    # vector_data['value'] = [x for x in raster_data.sample(coord_list)]
    # vector_data.head()


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

    # call main
    main()
    logging.info("Export finish!")




# def nearest(items, pivot):
#     return pd.to_datetime(min(
#         [i for i in items if i <= pivot],
#         key=lambda x: abs(x - pivot)
#     ))



# def clean_spain(vector_ds: gpd.GeoDataFrame
#                 ) -> gpd.GeoDataFrame:
#     """
#     Manage the clean of Spain data
#     """
#     # time columns  'f_datetime', 'Time' 'Date' -> mix f_datetime + Date

#     raise NotImplementedError




# def compute_extent(raster_extent: List[rio_coords.bounds],
#                    vector_extent: Any) -> Any:
#     """
#     Compute the common extension between
#     the raster and the vector dataset
#     for save resources
#     """
#     raise NotImplementedError
