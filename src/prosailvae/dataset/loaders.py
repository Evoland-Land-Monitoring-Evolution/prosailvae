#!/usr/bin/env python3
"""
Created on Tue Oct 25 13:39:40 2022

@author: yoel
"""
import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from mmdc_singledate.datamodules.components.datamodule_utils import (
    create_tensors_path_set,
)
from mmdc_singledate.datamodules.mmdc_datamodule import (
    IterableMMDCDataset,
    worker_init_fn,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

import prosailvae

# Configure logging
NUMERIC_LEVEL = getattr(logging, "INFO", None)
logging.basicConfig(
    level=NUMERIC_LEVEL, format="%(asctime)-15s %(levelname)s: %(message)s"
)

logger = logging.getLogger(__name__)


def split_train_valid_by_fid(
    labels: torch.Tensor, ts_ids: torch.Tensor, valid_size: float = 0.1, seed: int = 42
) -> tuple[list[str], list[str]]:
    """split database in train and valid subset according to a column name"""

    label_to_fids = dict(
        pd.DataFrame(
            {"label": labels.squeeze().numpy(), "id": ts_ids.squeeze().numpy()}
        )
        .groupby("label")["id"]
        .apply(set)
    )
    fids_train = []
    fids_valid = []
    for _, fids in label_to_fids.items():
        list_fids = list(fids)
        try:
            (_, _, train_fids, valid_fids) = train_test_split(
                list_fids, list_fids, test_size=valid_size, random_state=seed
            )
            # in order to manage unique label case value
        except ValueError:
            train_fids = valid_fids = fids

        fids_train += train_fids
        fids_valid += valid_fids
    return fids_train, fids_valid


def save_ids_for_k_fold(
    k: int = 2,
    test_ratio: float = 0.01,
    file_prefix: str = "s2_",
    data_dir: str | None = None,
) -> None:
    if data_dir is None:
        data_dir = os.path.join(
            os.path.join(os.path.dirname(prosailvae.__file__), os.pardir), "data/"
        )
    labels = torch.load(data_dir + "/s2_labels.pt")
    ts_ids = torch.arange(0, labels.size(0))
    rest_of_ids, test_ids = split_train_valid_by_fid(
        labels, ts_ids, valid_size=test_ratio, seed=42
    )
    torch.save(
        torch.tensor(test_ids).reshape(-1, 1), data_dir + f"/{file_prefix}test_ids.pt"
    )
    if k > 1:
        valid_size = 1 / k
        for i in range(k - 1):
            valid_size = 1 / (k - i)
            rest_of_ids, train_valid_ids = split_train_valid_by_fid(
                labels[rest_of_ids],
                torch.tensor(rest_of_ids),
                valid_size=valid_size,
                seed=42,
            )
            torch.save(
                torch.tensor(train_valid_ids).reshape(-1, 1),
                data_dir + f"/{file_prefix}train_valid_ids_{k}_{i}.pt",
            )
        torch.save(
            torch.tensor(rest_of_ids).reshape(-1, 1),
            data_dir + f"/{file_prefix}train_valid_ids_{k}_{k-1}.pt",
        )
    elif k == 1:
        torch.save(
            torch.tensor(rest_of_ids).reshape(-1, 1),
            data_dir + f"/{file_prefix}train_valid_ids.pt",
        )
    else:
        raise NotImplementedError


def lr_finder_loader(
    sample_ids=None,
    batch_size=1024,
    num_workers=0,
    file_prefix="s2_",
    data_dir=None,
    supervised=False,
    tensors_dir=None,
):
    if tensors_dir is None:
        if data_dir is None:
            data_dir = Path(f"{Path( __file__ ).parent.absolute()}/../../data/")
        s2_refl = torch.load(data_dir + f"/{file_prefix}prosail_s2_sim_refl.pt")
        len_dataset = s2_refl.size(0)

        prosail_sim_vars = torch.load(data_dir + f"/{file_prefix}prosail_sim_vars.pt")
        angles = prosail_sim_vars[:, -3:]
        prosail_parameters = prosail_sim_vars[:, :-3]
        if sample_ids is None:
            sample_ids = torch.arange(0, len_dataset)
            sub_s2_refl = s2_refl
            sub_angles = angles
            sub_prosail_parameters = prosail_parameters
        else:
            assert (sample_ids < len_dataset).all()
            sub_s2_refl = s2_refl[sample_ids, :]
            sub_angles = angles[sample_ids, :]
            sub_prosail_parameters = prosail_parameters[sample_ids, :]

        if supervised:
            dataset = TensorDataset(
                torch.concat((sub_s2_refl.float(), sub_angles.float()), axis=1),
                sub_prosail_parameters.float(),
            )
        else:
            dataset = TensorDataset(
                torch.concat((sub_s2_refl.float(), sub_angles.float()), axis=1),
                sub_s2_refl.float(),
            )

        loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    else:
        loader, _, _ = get_train_valid_test_loader_from_patches(
            data_dir, bands=torch.arange(10), batch_size=1, num_workers=0, concat=True
        )
        # raise NotImplementedError
    return loader


def get_simloader(
    valid_ratio=None,
    sample_ids=None,
    batch_size=1024,
    num_workers=0,
    file_prefix="s2_",
    data_dir=None,
    cat_angles=False,
    lai_only=False,
):
    if data_dir is None:
        data_dir = os.path.join(
            os.path.join(os.path.dirname(prosailvae.__file__), os.pardir), "data/"
        )
    s2_refl = torch.load(data_dir + f"/{file_prefix}prosail_s2_sim_refl.pt")
    len_dataset = s2_refl.size(0)

    prosail_sim_vars = torch.load(data_dir + f"/{file_prefix}prosail_sim_vars.pt")
    prosail_params = prosail_sim_vars[:, :-3]
    angles = prosail_sim_vars[:, -3:]
    if lai_only:
        prosail_params = prosail_params[:, 6].reshape(-1, 1)
    if sample_ids is None:
        sample_ids = torch.arange(0, len_dataset)
        sub_s2_refl = s2_refl
        sub_prosail_params = prosail_params
        sub_angles = angles
    else:
        assert (sample_ids < len_dataset).all()
        sub_s2_refl = s2_refl[sample_ids, :]
        sub_prosail_params = prosail_params[sample_ids, :]
        sub_angles = angles[sample_ids, :]

    if valid_ratio is None:
        dataset = TensorDataset(
            sub_s2_refl.float(),
            sub_angles.float(),
            sub_prosail_params.float(),
        )
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
        return loader
    else:
        n_valid = int(len(sample_ids) * valid_ratio)
        ids_train = sample_ids[n_valid:]
        ids_valid = sample_ids[:n_valid]

        sub_s2_refl_train = s2_refl[ids_train, :]
        sub_prosail_params_train = prosail_params[ids_train, :]
        sub_angles_train = angles[ids_train, :]

        sub_s2_refl_valid = s2_refl[ids_valid, :]
        sub_prosail_params_valid = prosail_params[ids_valid, :]
        sub_angles_valid = angles[ids_valid, :]

        train_dataset = TensorDataset(
            sub_s2_refl_train.float(),
            sub_angles_train.float(),
            sub_prosail_params_train.float(),
        )
        valid_dataset = TensorDataset(
            sub_s2_refl_valid.float(),
            sub_angles_valid.float(),
            sub_prosail_params_valid.float(),
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, num_workers=num_workers
        )

        valid_loader = DataLoader(
            valid_dataset, batch_size=batch_size, num_workers=num_workers
        )
        return train_loader, valid_loader


def get_S2_id_split_parser():
    """
    Creates a new argument parser.
    """
    parser = argparse.ArgumentParser(description="Parser for data generation")

    parser.add_argument(
        "-k", dest="k", help="total number k of fold", type=int, default=2
    )
    parser.add_argument(
        "-t",
        dest="test_ratio",
        help="ratio of test dataset size to size of the rest of the dataset",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "-p",
        dest="file_prefix",
        help="prefix on dataset files",
        type=str,
        default="s2_",
    )
    parser.add_argument(
        "-d", dest="data_dir", help="path to data directory", type=str, default=""
    )
    return parser


def get_norm_coefs(data_dir, file_prefix=""):
    norm_mean = torch.load(data_dir + f"/{file_prefix}norm_mean.pt")
    norm_std = torch.load(data_dir + f"/{file_prefix}norm_std.pt")
    if type(norm_mean) is np.ndarray:
        norm_mean = torch.from_numpy(norm_mean)
    if type(norm_std) is np.ndarray:
        norm_std = torch.from_numpy(norm_std)
    return norm_mean, norm_std


def convert_angles(angles):
    # TODO: convert 6 S2 "angles" into sun zenith, S2 zenith and Sun/S2
    # relative Azimuth (degrees)
    c_sun_zen = angles[:, 0].unsqueeze(1)
    c_sun_azi = angles[:, 1].unsqueeze(1)
    s_sun_azi = angles[:, 2].unsqueeze(1)
    c_obs_zen = angles[:, 3].unsqueeze(1)
    c_obs_azi = angles[:, 4].unsqueeze(1)
    s_obs_azi = angles[:, 5].unsqueeze(1)

    c_rel_azi = c_obs_azi * c_sun_azi + s_obs_azi * s_sun_azi
    s_rel_azi = c_obs_azi * s_sun_azi - s_obs_azi * c_sun_azi

    sun_zen = torch.rad2deg(torch.arccos(c_sun_zen))
    obs_zen = torch.rad2deg(torch.arccos(c_obs_zen))
    rel_azi = torch.rad2deg(torch.atan2(s_rel_azi, c_rel_azi)) % 360
    return torch.concat((sun_zen, obs_zen, rel_azi), axis=1)


def flatten_patch(s2_refl, angles):
    batch_size = s2_refl.size(0)
    patch_size_x = s2_refl.size(2)
    patch_size_y = s2_refl.size(3)
    s2_refl = (
        s2_refl.transpose(1, 2)
        .transpose(2, 3)
        .reshape(batch_size * patch_size_x * patch_size_y, 10)
    )
    angles = convert_angles(
        angles.transpose(1, 2)
        .transpose(2, 3)
        .reshape(batch_size * patch_size_x * patch_size_y, 6)
    )
    return s2_refl, angles


def get_mmdc_loaders(
    tensors_dir="",
    batch_size=1,
    batch_par_epoch=100,
    max_open_files=4,
    num_workers=1,
    pin_memory=False,
):
    train_data_files = [
        list(i)
        for i in create_tensors_path_set(
            path_to_exported_files=f"{tensors_dir}/", datasplit="train"
        )
    ]
    # print(f"{train_data_files}")
    val_data_files = [
        list(i)
        for i in create_tensors_path_set(
            path_to_exported_files=f"{tensors_dir}/", datasplit="val"
        )
    ]
    test_data_files = [
        list(i)
        for i in create_tensors_path_set(
            path_to_exported_files=f"{tensors_dir}/", datasplit="test"
        )
    ]
    # define iterable dataset
    data_train = IterableMMDCDataset(
        zip_files=train_data_files, batch_size=batch_size, max_open_files=max_open_files
    )
    # print(f"{data_train}")
    data_val = IterableMMDCDataset(
        zip_files=val_data_files, batch_size=batch_size, max_open_files=max_open_files
    )

    data_test = IterableMMDCDataset(
        zip_files=test_data_files, batch_size=batch_size, max_open_files=max_open_files
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
    return train_dataloader, val_dataloader, test_dataloader


def get_train_valid_test_loader_from_patches(
    path_to_patches_dir,
    bands=None,
    batch_size=1,
    num_workers=0,
    max_valid_samples=50,
    concat=False,
):
    if bands is None:
        bands = torch.arange(10)
    path_to_train_patches = os.path.join(path_to_patches_dir, "train_patches.pth")
    path_to_valid_patches = os.path.join(path_to_patches_dir, "valid_patches.pth")
    path_to_test_patches = os.path.join(path_to_patches_dir, "test_patches.pth")
    train_loader = get_loader_from_patches(
        path_to_train_patches,
        bands=bands,
        batch_size=batch_size,
        num_workers=num_workers,
        concat=concat,
    )
    valid_loader = get_loader_from_patches(
        path_to_valid_patches,
        bands=bands,
        batch_size=batch_size,
        num_workers=num_workers,
        max_samples=max_valid_samples,
        concat=concat,
        shuffle=False,
    )
    test_loader = get_loader_from_patches(
        path_to_test_patches,
        bands=bands,
        batch_size=batch_size,
        num_workers=num_workers,
        concat=concat,
        shuffle=False,
    )
    return train_loader, valid_loader, test_loader


def get_loader_from_patches(
    path_to_patches,
    bands=None,
    batch_size=1,
    num_workers=0,
    concat=False,
    max_samples=None,
    shuffle=True,
    plot_distribution=False,
):
    if bands is None:
        bands = torch.tensor([0, 1, 2, 4, 5, 6, 3, 7, 8, 9])
    patches = torch.load(path_to_patches)
    s2_a_patches = torch.zeros(patches.size(0), 3, patches.size(2), patches.size(3))
    s2_a_patches[:, 0, ...] = patches[:, 11, ...]  # sun zenith
    s2_a_patches[:, 1, ...] = patches[:, 13, ...]  # joint zenith
    s2_a_patches[:, 2, ...] = (
        patches[:, 12, ...] - patches[:, 14, ...]
    )  # relative azimuth : sun azimuth - joint azimuth
    s2_r_patches = patches[:, bands, ...]

    logger.info(
        "sample mean of bands values in loader: "
        f"{patches[0,:10,:,:].reshape(10,-1).mean(1).cpu()}"
    )
    logger.info(f"Loading from {path_to_patches}")
    if max_samples is not None:
        max_samples = min(max_samples, s2_r_patches.size(0))
        s2_r_patches = s2_r_patches[:max_samples, ...]
        s2_a_patches = s2_a_patches[:max_samples, ...]
        logger.info(f"Limit to {max_samples} samples")
    if concat:
        dataset = TensorDataset(
            torch.cat((s2_r_patches.float(), s2_a_patches.float()), axis=1)
        )
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
        )
    else:
        dataset = TensorDataset(s2_r_patches.float(), s2_a_patches.float())
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
        )
    return loader


if __name__ == "__main__":
    parser = get_S2_id_split_parser().parse_args()
    if len(parser.data_dir) == 0:
        data_dir = os.path.join(
            os.path.join(os.path.dirname(prosailvae.__file__), os.pardir), "data/"
        )
    save_ids_for_k_fold(k=parser.k, test_ratio=parser.test_ratio)
