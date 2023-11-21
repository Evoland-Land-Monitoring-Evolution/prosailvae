#!/usr/bin/env python3

"""
Delete the image data not used in the sampling
"""


import glob
import os
from pathlib import Path

import pandas as pd


def s2_data_files_used():
    """
    Get the s2 products used for sampling
    """
    # list file with data
    list_data_used_ = [
        glob.glob(
            "/home/uz/vinascj/scratch/fork2/prosailvae/field_data/processed/before*csv"
        )
        + glob.glob(
            "/home/uz/vinascj/scratch/fork2/prosailvae/field_data/processed/after*csv"
        )
    ]
    # flatten list
    list_data_used = [item for sublist in list_data_used_ for item in sublist]

    # read the used dataframe and fix the interest colum
    s2_filenames_l = []
    for idx, elemnt in enumerate(list_data_used):
        # read df
        interest_df = pd.read_csv(list_data_used[idx])
        # filter
        s2_filenames_l.append(interest_df["s2_filenames"])
    s2_filenames = pd.Series(pd.concat(s2_filenames_l).unique())

    return s2_filenames


def maestria_data_files_available():
    """
    Get the universe of produtcs in the maestria folder
    """
    # list s2 products
    maestria_folder = glob.glob(
        "/work/CESBIO/projects/MAESTRIA/prosail_validation/validation_sites/T*/*.tif"
    )
    # convert to df
    maestria_folder_df = pd.Series(maestria_folder, dtype=str)

    return maestria_folder_df


def main():
    # get the products using in the sampling
    s2_data_used = set(s2_data_files_used())

    # get the products in maestria
    maestria_data_available = set(maestria_data_files_available())

    # compute the difference
    products_difference = maestria_data_available.symmetric_difference(s2_data_used)

    # check that is the correct products
    products_intersect = products_difference.intersection(s2_data_used)
    assert len(products_intersect) > 0

    # delete the products
    # (Path.unlink(item) for item in products_difference)


if __name__ == "__main__":
    main()
