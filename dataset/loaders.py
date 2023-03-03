
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 13:39:40 2022

@author: yoel
"""
import os 
import pandas as pd
import prosailvae
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import argparse      

from mmdc_singledate.datamodules.mmdc_datamodule import (IterableMMDCDataset,
                                                         worker_init_fn,
                                                         destructure_batch)    
from mmdc_singledate.datamodules.components.datamodule_utils import (MMDCDataStats,
                                                                    #OneSetMMDCDataclass,
                                                                    average_stats,
                                                                    compute_stats,
                                                                    create_tensors_path)                                          


def split_train_valid_by_fid(labels, ts_ids,
                             valid_size=0.1,
                             seed=42):
    """split database in train and valid subset according to a column name

    Parameters
    ----------

    database_file
        sqlite database file
    labels_column
        labels column
    origin_fid_column
        column containing original fids
    test_size
        percentage of samples use to valid the model [0;1]
    seed
        random seed to split the database in train/validation sample-set

    Note
    ----

    Return a tuple of two list, the first list is dedicated to learn model
    and the second to validated it
    """
   
    label_to_fids = dict(pd.DataFrame({"label":labels.squeeze().numpy(), 
                                       "id":ts_ids.squeeze().numpy()}).groupby("label")["id"].apply(
                                           set))
    fids_train = []
    fids_valid = []
    for _, fids in label_to_fids.items():
        list_fids = list(fids)
        try:
            (_, _, train_fids,
             valid_fids) = train_test_split(list_fids,
                                            list_fids,
                                            test_size=valid_size,
                                            random_state=seed)
            # in order to manage unique label case value
        except ValueError:
            train_fids = valid_fids = fids

        fids_train += train_fids
        fids_valid += valid_fids
    return fids_train, fids_valid

def save_ids_for_k_fold(k=2, test_ratio=0.01, file_prefix="s2_", data_dir=None):
    if data_dir is None:
        data_dir = os.path.join(os.path.join(os.path.dirname(prosailvae.__file__),
                                             os.pardir),"data/")
    labels = torch.load(data_dir+"/s2_labels.pt")
    ts_ids = torch.arange(0,labels.size(0))
    rest_of_ids, test_ids = split_train_valid_by_fid(labels, ts_ids,
                                 valid_size=test_ratio,
                                 seed=42)
    torch.save(torch.tensor(test_ids).reshape(-1,1), data_dir + f"/{file_prefix}test_ids.pt")
    if k>1:
        valid_size = 1/k
        for i in range(k-1):
            valid_size = 1/(k-i)
            rest_of_ids, train_valid_ids = split_train_valid_by_fid(labels[rest_of_ids], 
                                                                    torch.tensor(rest_of_ids),
                                                                    valid_size=valid_size,
                                                                    seed=42)
            torch.save(torch.tensor(train_valid_ids).reshape(-1,1), 
                       data_dir + f"/{file_prefix}train_valid_ids_{k}_{i}.pt") 
        torch.save(torch.tensor(rest_of_ids).reshape(-1,1), 
                   data_dir + f"/{file_prefix}train_valid_ids_{k}_{k-1}.pt") 
    elif k==1:
        torch.save(torch.tensor(rest_of_ids).reshape(-1,1), 
                   data_dir + f"/{file_prefix}train_valid_ids.pt")
    else:
        raise NotImplementedError

def get_s2loader(valid_ratio=None, ts_ids=None, 
                 batch_size=1024, num_workers=0, file_prefix="s2_", data_dir=None):
    if data_dir is None:
        data_dir = os.path.join(os.path.join(os.path.dirname(prosailvae.__file__),
                                             os.pardir),"data/")
    ndvi = torch.load(data_dir + f"/{file_prefix}ndvi_dataset.pt")
    labels = torch.load(data_dir + f"/{file_prefix}labels.pt")
    if ts_ids is None:
        sub_ndvi = ndvi
        sub_labels = labels
    else:
        sub_ndvi = ndvi[ts_ids,:]
        sub_labels = labels[ts_ids,:]
        
    if valid_ratio is None:
        dataset = TensorDataset(sub_ndvi.float(), sub_labels.float())
        loader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=num_workers)
        return loader
    else:
        ids_train, ids_valid = split_train_valid_by_fid(sub_labels, ts_ids,
                                                          valid_size=0.1)
        sub_ndvi_train = ndvi[ids_train,:]
        sub_labels_train = labels[ids_train,:]
        sub_ndvi_valid = ndvi[ids_valid,:]
        sub_labels_valid = labels[ids_valid,:]
        train_dataset = TensorDataset(sub_ndvi_train.float(), sub_labels_train.float())
        train_loader = DataLoader(train_dataset,
                            batch_size=batch_size,
                            num_workers=num_workers)
        valid_dataset = TensorDataset(sub_ndvi_valid.float(), sub_labels_valid.float())
        valid_loader = DataLoader(valid_dataset,
                            batch_size=batch_size,
                            num_workers=num_workers)
        return train_loader, valid_loader

def load_train_valid_ids(k=None, n=None, file_prefix="s2_", data_dir=None):
    if data_dir is None:
        data_dir = os.path.join(os.path.join(os.path.dirname(prosailvae.__file__),
                                             os.pardir),"data/")
    if k is None or n is None:
        train_valid_ids = torch.load(data_dir + f"/{file_prefix}train_valid_ids.pt")
    else:
        train_valid_ids = torch.load(data_dir + f"/{file_prefix}train_valid_ids_{k}_{n}.pt")
    return train_valid_ids

def load_test_ids(file_prefix="s2_", data_dir=None):
    data_dir = os.path.join(os.path.join(os.path.dirname(prosailvae.__file__),
                                         os.pardir),"data/")
    return torch.load(data_dir + f"/{file_prefix}test_ids.pt")

def lr_finder_loader(sample_ids=None, 
                     batch_size=1024, num_workers=0, file_prefix="s2_", 
                     data_dir=None, supervised=False):
    if data_dir is None:
        data_dir = os.path.join(os.path.join(os.path.dirname(prosailvae.__file__),
                                             os.pardir),"data/")
    s2_refl = torch.load(data_dir + f"/{file_prefix}prosail_s2_sim_refl.pt")
    len_dataset = s2_refl.size(0)
    
    prosail_sim_vars = torch.load(data_dir + f"/{file_prefix}prosail_sim_vars.pt")
    angles = prosail_sim_vars[:,-3:]
    prosail_parameters = prosail_sim_vars[:,:-3]
    if sample_ids is None:
        sample_ids = torch.arange(0,len_dataset) 
        sub_s2_refl = s2_refl
        sub_angles = angles
        sub_prosail_parameters = prosail_parameters
    else:
        assert (sample_ids < len_dataset).all()
        sub_s2_refl = s2_refl[sample_ids,:]
        sub_angles = angles[sample_ids,:]
        sub_prosail_parameters = prosail_parameters[sample_ids,:]
    if supervised:
        dataset = TensorDataset(torch.concat((sub_s2_refl.float(),
                                sub_angles.float()), axis=1), 
                                sub_prosail_parameters.float(),
                                )
    else:
        dataset = TensorDataset(torch.concat((sub_s2_refl.float(),
                                sub_angles.float()), axis=1), 
                                sub_s2_refl.float(),
                                )

    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        num_workers=num_workers)
    return loader
    

def get_simloader(valid_ratio=None, sample_ids=None, 
                 batch_size=1024, num_workers=0, file_prefix="s2_", 
                 data_dir=None, cat_angles=False):
    if data_dir is None:
        data_dir = os.path.join(os.path.join(os.path.dirname(prosailvae.__file__),
                                             os.pardir),"data/")
    s2_refl = torch.load(data_dir + f"/{file_prefix}prosail_s2_sim_refl.pt")
    len_dataset = s2_refl.size(0)
    
    prosail_sim_vars = torch.load(data_dir + f"/{file_prefix}prosail_sim_vars.pt")
    prosail_params = prosail_sim_vars[:,:-3]
    angles = prosail_sim_vars[:,-3:]
    if sample_ids is None:
        sample_ids = torch.arange(0,len_dataset) 
        sub_s2_refl = s2_refl
        sub_prosail_params = prosail_params
        sub_angles = angles
    else:
        assert (sample_ids < len_dataset).all()
        sub_s2_refl = s2_refl[sample_ids,:]
        sub_prosail_params = prosail_params[sample_ids,:]
        sub_angles = angles[sample_ids,:]
        
    if valid_ratio is None:
       
        dataset = TensorDataset(sub_s2_refl.float(),
                                sub_angles.float(), 
                                sub_prosail_params.float(),
                                )
        loader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=num_workers)
        return loader
    else:
        n_valid = int(len(sample_ids) * valid_ratio)
        ids_train = sample_ids[n_valid:]
        ids_valid = sample_ids[:n_valid]
        
        sub_s2_refl_train = s2_refl[ids_train,:]
        sub_prosail_params_train = prosail_params[ids_train,:]
        sub_angles_train = angles[ids_train,:]
        
        sub_s2_refl_valid = s2_refl[ids_valid,:]
        sub_prosail_params_valid = prosail_params[ids_valid,:]
        sub_angles_valid = angles[ids_valid,:]

        train_dataset = TensorDataset(sub_s2_refl_train.float(),
                                      sub_angles_train.float(),
                                      sub_prosail_params_train.float(),)
        valid_dataset = TensorDataset(sub_s2_refl_valid.float(),
                                      sub_angles_valid.float(),
                                      sub_prosail_params_valid.float(),
                                       )
        train_loader = DataLoader(train_dataset,
                            batch_size=batch_size,
                            num_workers=num_workers)
        
        valid_loader = DataLoader(valid_dataset,
                            batch_size=batch_size,
                            num_workers=num_workers)
        return train_loader, valid_loader

def get_S2_id_split_parser():
    """
    Creates a new argument parser.
    """
    parser = argparse.ArgumentParser(description='Parser for data generation')
    
    parser.add_argument("-k", dest="k",
                        help="total number k of fold",
                        type=int, default=2)
    parser.add_argument("-t", dest="test_ratio",
                        help="ratio of test dataset size to size of the rest of the dataset",
                        type=float, default=0.01)
    parser.add_argument("-p", dest="file_prefix",
                        help="prefix on dataset files",
                        type=str, default="s2_")
    parser.add_argument("-d", dest="data_dir",
                        help="path to data directory",
                        type=str, default="")
    return parser

def get_norm_coefs(data_dir, file_prefix=''):
    norm_mean = torch.load(data_dir+f"/{file_prefix}norm_mean.pt")
    norm_std = torch.load(data_dir+f"/{file_prefix}norm_std.pt")
    if type(norm_mean) is np.ndarray:
        norm_mean = torch.from_numpy(norm_mean)
    if type(norm_std) is np.ndarray:
        norm_std = torch.from_numpy(norm_std)
    return norm_mean, norm_std


def convert_angles(angles):
    #TODO: convert 6 S2 "angles" into sun zenith, S2 zenith and Sun/S2 relative Azimuth (degrees)
    c_sun_zen = angles[:,0].unsqueeze(1)
    c_sun_azi = angles[:,1].unsqueeze(1)
    s_sun_azi = angles[:,2].unsqueeze(1)
    c_obs_zen = angles[:,3].unsqueeze(1)
    c_obs_azi = angles[:,4].unsqueeze(1)
    s_obs_azi = angles[:,5].unsqueeze(1)

    c_rel_azi = c_obs_azi * c_sun_azi + s_obs_azi * s_sun_azi
    s_rel_azi = c_obs_azi * s_sun_azi - s_obs_azi * c_sun_azi 

    sun_zen = torch.rad2deg(torch.arccos(c_sun_zen))
    obs_zen = torch.rad2deg(torch.arccos(c_obs_zen))
    rel_azi = torch.rad2deg(torch.atan2(s_rel_azi, c_rel_azi)) % 360
    sun_azi = torch.rad2deg(torch.atan2(s_sun_azi, c_sun_azi)) % 360
    obs_azi = torch.rad2deg(torch.atan2(s_obs_azi, c_obs_azi)) % 360
    return torch.concat((sun_zen, obs_zen, rel_azi), axis=1)

def flatten_patch(s2_refl, angles):
    batch_size = s2_refl.size(0)
    patch_size_x = s2_refl.size(2)
    patch_size_y = s2_refl.size(3)
    s2_refl = s2_refl.transpose(1,2).transpose(2,3).reshape(batch_size * patch_size_x * patch_size_y, 10)
    angles = convert_angles(angles.transpose(1,2).transpose(2,3).reshape(batch_size * patch_size_x * patch_size_y, 6))
    return s2_refl, angles

def patchify_2D_tensor(tensor_2D, data_size=10, patch_size=32):
    mixed_batch_size = tensor_2D.size(0) 
    batch_size = mixed_batch_size // patch_size**2 
    patchified_tensor = tensor_2D.reshape(batch_size, patch_size, patch_size, data_size).transpose(2,3).transpose(1,2)
    return patchified_tensor

def get_flattened_patch(batch, device='cpu'):
    (s2_r, s2_a, _, _, _, _, _) = destructure_batch(batch)
    s2_r, s2_a = flatten_patch(s2_r.to(device), s2_a.to(device))
    return s2_r, s2_a

def get_mmdc_loaders(tensors_dir="",
        batch_size=1,
        batch_par_epoch=100,
        max_open_files=4,
        num_workers=1,
        pin_memory=False):

    train_data_files = [
        list(i) for i in create_tensors_path(
        path_to_exported_files=f"{tensors_dir}/",
        datasplit="train")
        ]
    # print(f"{train_data_files}")
    val_data_files = [
        list(i) for i in create_tensors_path(
        path_to_exported_files=f"{tensors_dir}/",
        datasplit="val")
        ]
    test_data_files = [
        list(i) for i in create_tensors_path(
        path_to_exported_files=f"{tensors_dir}/",
        datasplit="test")
        ]
    # define iterable dataset
    data_train = IterableMMDCDataset(
            zip_files=train_data_files,
            batch_size=batch_size,
            max_open_files=max_open_files)
    # print(f"{data_train}")
    data_val = IterableMMDCDataset(
            zip_files=val_data_files,
            batch_size=batch_size,
            max_open_files=max_open_files)

    data_test = IterableMMDCDataset(
    zip_files=test_data_files,
    batch_size=batch_size,
    max_open_files=max_open_files)
    # Make a DataLoader

    train_dataloader =  DataLoader(dataset=data_train,
                            batch_size=None,
                            num_workers=num_workers,
                            pin_memory=pin_memory,
                            worker_init_fn=worker_init_fn)

    val_dataloader = DataLoader(dataset=data_val,
                            batch_size=None,
                            num_workers=num_workers,
                            pin_memory=pin_memory,
                            worker_init_fn=worker_init_fn)

    test_dataloader = DataLoader(dataset=data_test,
                    batch_size=None,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    worker_init_fn=worker_init_fn)
    return train_dataloader, val_dataloader, test_dataloader




if __name__ == "__main__":
    parser = get_S2_id_split_parser().parse_args()
    if len(parser.data_dir)==0:
        data_dir = os.path.join(os.path.join(os.path.dirname(prosailvae.__file__),
                                             os.pardir),"data/")
    save_ids_for_k_fold(k=parser.k, test_ratio=parser.test_ratio)