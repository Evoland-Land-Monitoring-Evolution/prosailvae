import torch
import os
from torchutils.patches import patchify, unpatchify 
import argparse
import socket
import numpy as np

BANDS_IDX = {'B02':0, 'B03':1, 'B04':2, 'B05':4, 'B06':5, 'B07':6, 'B08':3, 'B8A':7, 'B11':8, 'B12':9}

def get_parser():
    """
    Creates a new argument parser.
    """
    parser = argparse.ArgumentParser(description='Parser for data generation')
    
    parser.add_argument("-d", dest="data_dir",
                        help="path to torchfiles directories,with tiles subdirectories",
                        type=str, default="")
    
    parser.add_argument("-o", dest="output_dir",
                        help="path to directory to save data into",
                        type=str, default="")       
    return parser



def get_images_path(data_dir, valid_tiles=None, valid_files=None):
    list_files = []
    id_list = []
    file_info = []
    tile_dirs = os.listdir(data_dir)
    
    for tile in tile_dirs:
        if valid_tiles is not None:
            if tile not in valid_tiles:
                continue
        tile_dir = os.path.join(data_dir, tile)
        if os.path.isdir(tile_dir):
            tile_files = os.listdir(tile_dir)
            for tile_file in tile_files:
                if valid_files is not None:
                    if tile_file not in valid_files:
                        continue
                tile_file_path = os.path.join(tile_dir, tile_file)
                if tile_file[-4:] == ".pth":
                    sensor, date, tile, id = get_info_from_filename(tile_file)
                    if id in id_list:
                        print(f"Already an image with : {sensor}, {date}, {tile}")
                    else:
                        id_list.append(id)
                        file_info.append([sensor, date, tile])
                        print(f"Adding : {tile_file}")
                        list_files.append(tile_file_path)
    return list_files, file_info

def get_valid_area_in_image(tile):
    if tile=="T31TCJ":
        return 0, 512, 0, 512
    elif tile=="T30TUM":
        return 0, 512, 0, 512
    elif tile=="T33TWF":
        return 1700, 2212, 0, 512
    elif tile=="T33TWG":
        return 0, 512, 1533, 2045
    # if tile=="T31TCJ":
    #     return 0, 1024, 0, 1024
    # elif tile=="T30TUM":
    #     return 0, 1024, 0, 1024
    # elif tile=="T33TWF":
    #     return 0, 1024, 0, 1024
    # elif tile=="T33TWF":
    #     return 0, 1024, 1020, 2044
    else:
        raise NotImplementedError

def get_patches(image_tensor, patch_size):
    patches = patchify(image_tensor, patch_size=patch_size, margin=0)
    if image_tensor.size(1) % patch_size != 0:
        patches = patches[:-1,...]
    if image_tensor.size(2) % patch_size != 0:
        patches = patches[:,:-1,...]
    patches = patches.reshape(-1, image_tensor.size(0), patch_size, patch_size)
    return patches

def get_clean_patch_tensor(patches, cloud_mask_idx=10, reject_mode='all'):
    clean_patches = []
    nan_flag = False
    for i in range(patches.size(0)):
        patch = patches[i,...]
        validity = patch[cloud_mask_idx,...]
        if reject_mode == 'all':
            if not validity.any():
                if not torch.isnan(patch).any():
                    clean_patches.append(patch.unsqueeze(0))
                else:
                    nan_flag=True
        else:
            raise NotImplementedError
    if len(clean_patches)>0:
        clean_patches = torch.cat(clean_patches, dim=0)
    if nan_flag:
        print("WARNING: patches with nan values were detected in this data !")
    return clean_patches

def get_train_valid_test_patch_tensors(data_dir, large_patch_size = 128, train_patch_size = 32, 
                                       valid_size = 0.05, test_size = 0.05, valid_tiles=None, valid_files=None):
    assert large_patch_size % train_patch_size == 0
    tensor_files, file_info = get_images_path(data_dir, valid_tiles=valid_tiles, valid_files=valid_files)
    train_clean_patches = []
    valid_clean_patches = []
    test_clean_patches = []
    seed = 4235910
    train_patch_info = []
    valid_patch_info = []
    test_patch_info = []
    for i, tensor_file in enumerate(tensor_files):
        info = file_info[i]
        print(tensor_file)
        image_tensor = torch.load(tensor_file)
        min_x, max_x, min_y, max_y = get_valid_area_in_image(info[2])
        image_tensor = image_tensor[:,min_x: max_x, min_y: max_y]
        patches = get_patches(image_tensor, large_patch_size)
        n_valid = max(int(patches.size(0) * valid_size), 1)
        n_test = max(int(patches.size(0) * test_size), 1)
        n_train = patches.size(0) - n_valid - n_test
        g_cpu = torch.Generator()
        g_cpu.manual_seed(seed)
        perms = torch.randperm(patches.size(0), generator=g_cpu) # For image tensor with identical sizes (i.e. the same sites) permutation will always be the same
        train_patches = get_clean_patch_tensor(patches[perms[:n_train],...], cloud_mask_idx=10, reject_mode='all')
        valid_patches = get_clean_patch_tensor(patches[perms[n_train:n_train + n_valid] ,...], cloud_mask_idx=10, reject_mode='all')
        test_patches = get_clean_patch_tensor(patches[perms[n_train + n_valid:],...], cloud_mask_idx=10, reject_mode='all')
        if len(train_patches) > 0:
            train_clean_patches.append(train_patches)
            train_patch_info += [info] * n_train * (large_patch_size // train_patch_size)**2
        if len(valid_patches) > 0:
            valid_clean_patches.append(valid_patches)
            valid_patch_info += [info] * n_valid * (large_patch_size // train_patch_size)**2
        if len(test_patches) > 0:
            test_clean_patches.append(test_patches)
            test_patch_info += [info] * n_test

    train_clean_patches = torch.cat(train_clean_patches, dim=0)
    train_clean_patches = patchify(unpatchify(train_clean_patches.unsqueeze(0)), patch_size=train_patch_size).reshape(-1,image_tensor.size(0), train_patch_size, train_patch_size)
    train_perms = torch.randperm(train_clean_patches.size(0), generator=g_cpu)
    train_clean_patches = train_clean_patches[train_perms,...]
    train_patch_info = np.array(train_patch_info)[train_perms,:]
    valid_clean_patches = torch.cat(valid_clean_patches, dim=0)
    valid_clean_patches = patchify(unpatchify(valid_clean_patches.unsqueeze(0)), patch_size=train_patch_size).reshape(-1,image_tensor.size(0), train_patch_size, train_patch_size)
    valid_perms = torch.randperm(valid_clean_patches.size(0), generator=g_cpu)
    valid_clean_patches = valid_clean_patches[valid_perms,...]
    valid_patch_info = np.array(valid_patch_info)[valid_perms,:]
    test_clean_patches = torch.cat(test_clean_patches, dim=0)
    test_perms = torch.randperm(test_clean_patches.size(0), generator=g_cpu)
    test_clean_patches = test_clean_patches[test_perms,...]
    test_patch_info = np.array(test_patch_info)[test_perms,:]
    print(f"Train patches : {train_clean_patches.size()}")
    print(f"Validation patches : {valid_clean_patches.size()}")
    print(f"Test patches : {test_clean_patches.size()}")
    swap_bands(train_clean_patches)
    swap_bands(valid_clean_patches)
    swap_bands(test_clean_patches)
    return train_clean_patches, valid_clean_patches, test_clean_patches, train_patch_info, valid_patch_info, test_patch_info

def swap_bands(patches):
    idx = torch.LongTensor([ v for _, (_,v) in enumerate(BANDS_IDX.items()) ])
    patches[:,torch.arange(10),...] = patches[:,idx,...]
    return

def get_bands_norm_factors_from_patches(patches, n_bands=10, mode='mean'):
    with torch.no_grad():
        s2_r_samples = patches.permute(1,0,2,3)[:n_bands,...].reshape(n_bands, -1)
        if mode=='mean':
            norm_mean = s2_r_samples.mean(1)
            norm_std = s2_r_samples.std(1)
        elif mode=='quantile':
            max_samples=int(1e6)
            norm_mean = torch.quantile(s2_r_samples[:, :max_samples], q=torch.tensor(0.5), dim=1)
            norm_std = torch.quantile(s2_r_samples[:, :max_samples], q=torch.tensor(0.95), dim=1) - torch.quantile(s2_r_samples[:, :max_samples], q=torch.tensor(0.05), dim=1)
    return norm_mean, norm_std

def get_info_from_filename(filename):
    filename_comp = filename.split("_")
    if filename_comp[1] == "SENTINEL2A":
        sensor = "2A"
    elif filename_comp[1] == "SENTINEL2B":
        sensor = "2B"
    else:
        raise ValueError("Sensor name not found!")
    date = filename_comp[2].split("-")[0]
    tile = filename_comp[4]
    return sensor, date, tile, sensor + date + tile

def main():
    if socket.gethostname()=='CELL200973':
        args=["-d", "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/torch_files/",
              "-o", "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/patches/"]
        
        parser = get_parser().parse_args(args)    
    else:
        parser = get_parser().parse_args()


    if not os.path.isdir(parser.output_dir):
        os.makedirs(parser.output_dir)
    large_patch_size = 128
    train_patch_size = 16
    valid_size = 0.05
    test_size = 0.05
    valid_tiles = ["T31TCJ", "T30TUM", "T33TWF", "T33TWG"]
    valid_files = [
                    "after_SENTINEL2B_20171127-105827-648_L2A_T31TCJ_C_V2-2_roi_0.pth",
                   "before_SENTINEL2A_20180620-105211-086_L2A_T31TCJ_C_V2-2_roi_0.pth",
                   "after_SENTINEL2A_20170711-111223-375_L2A_T30TUM_D_V1-7_roi_0.pth",
                   "after_SENTINEL2A_20180417-110822-655_L2A_T30TUM_C_V2-2_roi_0.pth",
                   "before_SENTINEL2A_20170518-095716-529_L2A_T33TWF_D_V1-4_roi_0.pth",
                   "before_SENTINEL2A_20170408-095711-526_L2A_T33TWF_D_V1-4_roi_0.pth",
                   "before_SENTINEL2A_20170518-095716-529_L2A_T33TWG_D_V1-4_roi_0.pth"
                ]
    valid_files = None
    (train_patches, valid_patches, test_patches,
     train_patch_info, valid_patch_info,
     test_patch_info) = get_train_valid_test_patch_tensors(data_dir=parser.data_dir, large_patch_size = large_patch_size, 
                                                        train_patch_size = train_patch_size, 
                                                        valid_size = valid_size, test_size = test_size,
                                                        valid_tiles=valid_tiles, valid_files=valid_files)

    norm_mean, norm_std = get_bands_norm_factors_from_patches(train_patches, mode='mean')
    print(f"mean {norm_std}, std {norm_std}")
    norm_mean, norm_std = get_bands_norm_factors_from_patches(train_patches, mode='quantile')
    print(f"median {norm_mean}, quantiles difference {norm_std}")

    
    torch.save(norm_mean, os.path.join(parser.output_dir, "norm_mean.pt"))
    torch.save(norm_std, os.path.join(parser.output_dir, "norm_std.pt"))
    torch.save(train_patches, os.path.join(parser.output_dir, "train_patches.pth"))
    torch.save(valid_patches, os.path.join(parser.output_dir, "valid_patches.pth"))
    torch.save(test_patches, os.path.join(parser.output_dir, "test_patches.pth"))
    np.save(os.path.join(parser.output_dir, "train_info.npy"), train_patch_info)
    np.save(os.path.join(parser.output_dir, "valid_info.npy"), valid_patch_info)
    np.save(os.path.join(parser.output_dir, "test_info.npy"), test_patch_info)
    return 

if __name__=='__main__':
    main()