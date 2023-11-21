#!/usr/bin/env python3
# Copyright: (c) 2022 CESBIO / Centre National d'Etudes Spatiales
"""
Collection of functions and utilities for process some data
related with MMDC project

"""

# imports

import os
import sys
from typing import Any

import torch
from torch.utils.data import DataLoader

thirdparties_path = "/home/yoel/Documents/Dev/PROSAIL-VAE/thirdparties/"
# thirdparties_path = "/work/scratch/zerahy/src/thirdparties/"

sys.path = [
    thirdparties_path + "/mmdc-singledate",
    thirdparties_path + "/sensorsio",
    thirdparties_path + "/torchutils/src",
] + sys.path

from src.mmdc_singledate.datamodules.components.datamodule_utils import (  # OneSetMMDCDataclass,
    MMDCDataStats,
    average_stats,
    compute_stats,
    create_tensors_path,
)
from src.mmdc_singledate.datamodules.mmdc_datamodule import (
    IterableMMDCDataset,
    destructure_batch,
    worker_init_fn,
)


def main():
    # parameters
    tensors_dir = "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/real_data/torchfiles/"  #'/work/CESBIO/projects/MAESTRIA/prosail_validation/toyexample/torchfiles/'
    batch_size = None
    batch_par_epoch = 100
    max_open_files = 4
    num_workers = 1
    pin_memory = False
    nb_epochs = 5  # 00

    # prepare

    train_data_files = [
        list(i)
        for i in create_tensors_path(
            path_to_exported_files=f"{tensors_dir}/T*", datasplit="train"
        )
    ]
    print(f"{train_data_files}")
    val_data_files = [
        list(i)
        for i in create_tensors_path(
            path_to_exported_files=f"{tensors_dir}/T*", datasplit="val"
        )
    ]

    test_data_files = [
        list(i)
        for i in create_tensors_path(
            path_to_exported_files=f"{tensors_dir}/T*", datasplit="test"
        )
    ]

    # define iterable dataset
    data_train = IterableMMDCDataset(
        zip_files=train_data_files, batch_size=1, max_open_files=max_open_files
    )
    print(f"{data_train}")
    data_val = IterableMMDCDataset(
        zip_files=val_data_files, batch_size=1, max_open_files=max_open_files
    )

    data_test = IterableMMDCDataset(
        zip_files=test_data_files, batch_size=1, max_open_files=max_open_files
    )
    # Make a DataLoader

    train_dataloader = DataLoader(
        dataset=data_train,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
    )

    val_dataloader = DataLoader(
        dataset=data_val,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
    )

    test_dataloader = DataLoader(
        dataset=data_test,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
    )

    # Make the training loop
    for e in range(nb_epochs):
        for idx, batch in enumerate(test_dataloader):
            # for idx, batch in zip(range(batch_par_epoch), test_dataloader):
            (s2_x, s2_a, s1_x, s1_a_asc, s1_a_desc, wc_x, srtm_x) = destructure_batch(
                batch
            )
            print(f"s2_x = {s2_x.shape}"),
            print(f"s2_a = {s2_a.shape}"),
            print(f"s1_x = {s1_x.shape}"),
            print(f"s1_a_asc = {s1_a_asc.shape}"),
            print(f"s1_a_desc = {s1_a_desc.shape}"),
            print(f"wc_x = {wc_x.shape}"),
            print(f"srtm_x = {srtm_x.shape}")
            print(f"*-*-*-*-*-*-*-*-*-*-*")

    print("done!")


if __name__ == "__main__":
    main()


#
