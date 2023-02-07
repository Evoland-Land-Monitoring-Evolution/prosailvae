#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright: (c) 2022 CESBIO / Centre National d'Etudes Spatiales
"""

Example of how to read a raster and a vector file with the objective
get the in situ measurements for validate the results.

Inspired by :
https://geopandas.org/en/stable/gallery/geopandas_rasterio_sample.html
https://github.com/ArjanCodes/2022-abtest/blob/main/from_config/main.py
"""
# imports
import argparse
import json
from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import Any

import geopandas as gpd
import rasterio as rio

# configurations

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


# TODO Read Vector Data
# TODO Read Raster Data
# TODO Get point values

@dataclass
class Config:
    """
    Dataclass for hold the dataset
    filenames
    """
    raster: str
    vector: str


def read_config_file() -> Config:
    """
    Func for retrieve the config file
    """
    current_path = Path.cwd()
    parent_path = current_path.parent
    config_file = parent_path / "config" / "validation.json"
    config_dict = json.loads(config_file.read_text())
    return Config(**config_dict)


def read_vector(vector_filename: str) -> gpd.GeoDataFrame:
    """
    Read the the vector data
    """
    # read geo data
    logging.info(f"reading vector file : {vector_filename}")
    vector_data = gpd.read_file(vector_filename)

    return vector_data

def read_raster(raster_filename : str) -> Any:
    """
    read a raster file in a given extension
    using rasterio functionallyties
    """
    raise NotImplementedError

def compute_extent():
    """
    Compute the common extension between
    the raster and the vector dataset
    for save resources
    """
    raise NotImplementedError

def main():

    # read config
    config = read_config_file()

    raster_dataset_path = config.raster
    vector_dataset_path = config.vector

    # read geo data
    vector_data = gpd.read_file(vector_dataset_path)

    with rio.open(raster_dataset_path) as raster:
        raster_data = raster.read()

    # extract pixel values
    coord_list = [(x,y) for x,y
                  in zip(vector_data['geometry'].x,
                         vector_data['geometry'].y)]

    vector_data['value'] = [x for x in raster_data.sample(coord_list)]
    vector_data.head()


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
