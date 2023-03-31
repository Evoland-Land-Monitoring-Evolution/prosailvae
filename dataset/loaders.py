
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
import socket

from mmdc_singledate.datamodules.mmdc_datamodule import (IterableMMDCDataset,
                                                         worker_init_fn,
                                                         destructure_batch)    
from mmdc_singledate.datamodules.components.datamodule_utils import (MMDCDataStats,
                                                                    #OneSetMMDCDataclass,
                                                                    average_stats,
                                                                    compute_stats,
                                                                    create_tensors_path)                                          

from torchutils.patches import patchify, unpatchify

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
                     data_dir=None, supervised=False, tensors_dir=None):
    if tensors_dir is None:
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
        
    else:
        # _,loader,_, = get_mmdc_loaders(tensors_dir=tensors_dir,
        #                                 batch_size=1,
        #                                 batch_par_epoch=100,
        #                                 max_open_files=4,
        #                                 num_workers=1,
        #                                 pin_memory=False)
        path_to_image = tensors_dir + "/after_SENTINEL2B_20171127-105827-648_L2A_T31TCJ_C_V2-2_roi_0.pth"
        _, loader, _ = get_loaders_from_image(path_to_image, patch_size=32, train_ratio=0.8, valid_ratio=0.1, 
                                               bands = torch.tensor([0,1,2,4,5,6,3,7,8,9]), n_patches_max = 100, 
                                                batch_size=1, num_workers=0, concat=True)
        # raise NotImplementedError
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


def get_loaders_from_image(path_to_image, patch_size=32, train_ratio=0.8, valid_ratio=0.1, 
                          bands = torch.tensor([0,1,2,4,5,6,3,7,8,9]), n_patches_max = 100, 
                          batch_size=1, num_workers=0, concat=False, max_im_size=1024, seed=2147483647):
    if socket.gethostname()=='CELL200973':
        patch_size = 16
    assert train_ratio + valid_ratio <=1
    image_tensor = torch.load(path_to_image)
    max_im_size = min(max_im_size, image_tensor.size(1), image_tensor.size(2))
    image_tensor = image_tensor[:,:max_im_size,:max_im_size]
    angles = torch.zeros(3, image_tensor.size(1),image_tensor.size(2))
    angles[0,...] = image_tensor[11,...]
    angles[1,...] = image_tensor[13,...]
    angles[2,...] = image_tensor[12,...] - image_tensor[14, ...]
    s2_r = image_tensor[bands,...]
    s2_r_patches = patchify(s2_r, patch_size=patch_size, margin=0).reshape(-1,len(bands), patch_size, patch_size)
    s2_a_patches = patchify(angles, patch_size=patch_size, margin=0).reshape(-1, 3,patch_size, patch_size)
    n_patches = min(s2_a_patches.size(0), n_patches_max)
    g_cpu = torch.Generator()
    g_cpu.manual_seed(seed)
    perms = torch.randperm(s2_r_patches.size(0), generator=g_cpu)
    train_idx = perms[torch.arange(int(train_ratio*n_patches))]
    valid_idx = perms[torch.arange(int(train_ratio*n_patches),int((valid_ratio + train_ratio)*n_patches))]
    test_idx = perms[torch.arange(int((valid_ratio+train_ratio)*n_patches),n_patches)]
    if concat:
        train_dataset = TensorDataset(torch.cat((s2_r_patches[train_idx,...], s2_a_patches[train_idx,...]), axis=1))
        valid_dataset = TensorDataset(torch.cat((s2_r_patches[valid_idx,...], s2_a_patches[valid_idx,...]), axis=1))
        test_dataset = TensorDataset(torch.cat((s2_r_patches[test_idx,...], s2_a_patches[test_idx,...]), axis=1))
    else:
        train_dataset = TensorDataset(s2_r_patches[train_idx,...], s2_a_patches[train_idx,...])
        valid_dataset = TensorDataset(s2_r_patches[valid_idx,...], s2_a_patches[valid_idx,...])
        test_dataset = TensorDataset(s2_r_patches[test_idx,...], s2_a_patches[test_idx,...])
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=num_workers)
    return train_loader, valid_loader, test_loader

def get_train_valid_test_loader_from_patches(path_to_patches_dir, bands = torch.tensor([0,1,2,4,5,6,3,7,8,9]), 
                             batch_size=1, num_workers=0):
    path_to_train_patches = os.path.join(path_to_patches_dir, "train_patches.pth")
    path_to_valid_patches = os.path.join(path_to_patches_dir, "valid_patches.pth")
    path_to_test_patches = os.path.join(path_to_patches_dir, "test_patches.pth")
    train_loader = get_loader_from_patches(path_to_train_patches, bands = bands, 
                             batch_size=batch_size, num_workers=num_workers)
    valid_loader = get_loader_from_patches(path_to_valid_patches, bands = bands, 
                             batch_size=batch_size, num_workers=num_workers)
    test_loader = get_loader_from_patches(path_to_test_patches, bands = bands, 
                             batch_size=batch_size, num_workers=num_workers)
    return train_loader, valid_loader, test_loader

def get_loader_from_patches(path_to_patches, bands = torch.tensor([0,1,2,4,5,6,3,7,8,9]), 
                             batch_size=1, num_workers=0):
    patches = torch.load(path_to_patches)
    s2_a_patches = torch.zeros(patches.size(0), 3, patches.size(2),patches.size(3))
    s2_a_patches[:,0,...] = patches[:,11,...]
    s2_a_patches[:,1,...] = patches[:,13,...]
    s2_a_patches[:,2,...] = patches[:,12,...] - patches[:,14, ...]
    s2_r_patches = patches[:,bands,...]
    dataset = TensorDataset(torch.cat((s2_r_patches, s2_a_patches), axis=1))
    loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    return loader

def get_bands_norm_factors_from_loaders(loader, bands_dim=1, max_samples=10000, n_bands=10):
    s2_r_samples = torch.tensor([])
    n_samples = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            s2_r = batch[0]
            assert s2_r.size(bands_dim) == n_bands
            s2_r = s2_r.transpose(0, bands_dim).reshape(n_bands, -1)
            n_samples += s2_r.size(1)
            s2_r_samples = torch.cat((s2_r_samples, s2_r), axis=1)
            if n_samples >= max_samples:
                break
        if s2_r_samples.size(1) > max_samples:
            s2_r_samples = s2_r_samples[:,:max_samples]
        norm_mean = s2_r_samples.mean(1)
        norm_std = s2_r_samples.std(1)
    return norm_mean, norm_std

if __name__ == "__main__":
    parser = get_S2_id_split_parser().parse_args()
    if len(parser.data_dir)==0:
        data_dir = os.path.join(os.path.join(os.path.dirname(prosailvae.__file__),
                                             os.pardir),"data/")
    save_ids_for_k_fold(k=parser.k, test_ratio=parser.test_ratio)


