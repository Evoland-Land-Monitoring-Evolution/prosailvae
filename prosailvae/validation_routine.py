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

# TODO Read Vector Data
# TODO Read Raster Data
# TODO Get point values
#


import sys,os
from typing import Any
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import geopandas as gpd
import rasterio as rio
import rasterio.plot as rio_plot

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


def main():

    # read config
    config = read_config_file()

    raster_dataset_path = config.raster
    vector_dataset_path = config.vector

    # read geo data
    vector_data = gpd.read_file(vector_dataset_path)

    with rio.open(raster_dataset_path) as raster:
        raster_data = raster.read()

    # plot
    # fig, ax = plt.subplots()

    # compute the extent of the raster
    # extent = [raster_data.bounds[0],
    #           raster_data.bounds[1],
    #           raster_data.bounds[2],
    #           raster_data.bounds[3]]

    # ax = rio_plot.show(raster_data, extent=extent, ax=ax)
    # vector_data.plot(ax=ax)
    # plt.show()

    # extract pixel values
    coord_list = [(x,y) for x,y
                  in zip(vector_data['geometry'].x,
                         vector_data['geometry'].y)]

    vector_data['value'] = [x for x in raster_data.sample(coord_list)]
    vector_data.head()


if __name__ == "__main__":
    main()
