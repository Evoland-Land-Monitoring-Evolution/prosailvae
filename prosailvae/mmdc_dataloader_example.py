#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright: (c) 2022 CESBIO / Centre National d'Etudes Spatiales
"""
Collection of functions and utilities for process some data
related with MMDC project


"""

# imports

from typing import Any
from mmdc-singledate import IterableMMDCDataset

from src.datamodules.components.datamodule_utils import (MMDCDataStats,
                                                         OneSetMMDCDataclass,
                                                         average_stats,
                                                         compute_stats,
                                                         create_tensors_path,
                                                         create_tensors_stats_path)


# prepare

train_data_files = [
            list(i) for i in create_tensors_path(
                path_to_exported_files=f"{self.tensors_dir}/T*",
                datasplit="train")
        ]

val_data_files = [
            list(i) for i in create_tensors_path(
                path_to_exported_files=f"{self.tensors_dir}/T*",
                datasplit="val")
        ]


# define iterable dataset
data_train = IterableMMDCDataset(
            zip_files=self.train_data_files,
            batch_size=self.batch_size,
            batch_par_epoch=self.batch_par_epoch,
            max_open_files=self.max_open_files)

data_val = IterableMMDCDataset(
            zip_files=self.val_data_files,
            batch_size=self.batch_size,
            batch_par_epoch=self.batch_par_epoch,
            max_open_files=self.max_open_files)



# Make a DataLoader

train_dataloader =  DataLoader(dataset=self.data_train,
                          batch_size=None,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          worker_init_fn=worker_init_fn)

val_dataloader = DataLoader(dataset=self.data_val,
                          batch_size=None,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          worker_init_fn=worker_init_fn)


def test_dataloader():
        test_data_files = [
            list(i) for i in create_tensors_path(
                path_to_exported_files=f"{self.tensors_dir}/T*",
                datasplit="test")
        ]

        self.data_test = IterableMMDCDataset(
            zip_files=test_data_files,
            batch_size=self.batch_size,
            batch_par_epoch=self.batch_par_epoch,
            max_open_files=self.max_open_files)

        return DataLoader(dataset=self.data_test,
                          batch_size=None,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          worker_init_fn=worker_init_fn)

# Make the training loop
















#
