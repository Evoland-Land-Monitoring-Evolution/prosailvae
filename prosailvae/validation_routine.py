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
from typing import Any, List, Union, Tuple, List

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
    # DataFrame for store the values
    out_df = pd.DataFrame(pixel_values, columns=colnames)

    return out_df


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
                        ) -> List[Any]:
    """
    Compute the time difference between the field
    campaing and the satellite acquisition
    """
    time_delta = [abs(vector_timestamp - raster_ts)
                  for raster_ts in raster_timestamp.to_list()]
    return time_delta


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


    # init export dataframe
    col_names = ["B"+str(i+1) for i in range(10)]

    aux_names = ['sat_mask', 'cloud_mask', 'edge_mask','geophysical_mask',
                 'sun_zen', 'sun_az', 'even_zen', 'odd_zen', 'even_az', 'odd_az']
    col_names.extend(aux_names)
    values_list = []

    # get the pandas groups by date
    #gb_date = vector_data.groupby("Date")
    #gb_date_key = next(iter(gb_date.groups.keys()))
    #gb_date.get_group(gb_date_key)

    # iterate over the vector dates
    for vector_ts, vector_ds in vector_data.groupby("Date"):
        # get the n closest dates
        #vector_ds_date = dict(vector_ds[1])["Date"]
        time_delta = compute_time_deltas(
            vector_ts,
            raster_ts["s2_date"]
        )
        raster_ts["time_delta"] = time_delta
        n_closest = raster_ts.sort_values(by="time_delta").iloc[:3, :]
        # select the raster img with the closest
        # acquistions dates
        for idx, n_nearest in enumerate(n_closest.iterrows()):
            # get the pixel values for those how match
            # the criterium
            s2_pix_values = get_pixel_value(
                n_nearest[1]["s2_filenames"],
                vector_ds,
                col_names
            )
            # append field and raster date
            s2_pix_values["s2_date"] = [n_nearest[1]["s2_date"] for
                                        i in range(len(s2_pix_values))]
            s2_pix_values["field_date"] = [vector_ts for
                                           i in range(len(s2_pix_values))]
            values_list.append(s2_pix_values)
    # get the final value
    final_df = pd.concat(values_list)
    # export file
    final_df.to_csv(f"{args.export_path}/{config.site}.csv",
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
