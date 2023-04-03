import torch
import os
from torchutils.patches import patchify, unpatchify 
import argparse
import socket

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
    tile_dirs = os.listdir(data_dir)
    for tile_dir in tile_dirs:
        if valid_tiles is not None:
            if tile_dir not in valid_tiles:
                continue
        tile_dir = os.path.join(data_dir, tile_dir)
        if os.path.isdir(tile_dir):
            tile_files = os.listdir(tile_dir)
            for tile_file in tile_files:
                if valid_files is not None:
                    if tile_file not in valid_files:
                        continue
                tile_file = os.path.join(tile_dir, tile_file)
                if tile_file[-4:] == ".pth":
                    print(tile_file)
                    list_files.append(tile_file)
    return list_files

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
    for i in range(patches.size(0)):
        patch = patches[i,...]
        validity = patch[cloud_mask_idx,...]
        if reject_mode == 'all':
            if not validity.any():
                clean_patches.append(patch.unsqueeze(0))
        else:
            raise NotImplementedError
    if len(clean_patches)>0:
        clean_patches = torch.cat(clean_patches, dim=0)
    return clean_patches

def get_train_valid_test_patch_tensors(data_dir, large_patch_size = 128, train_patch_size = 32, 
                                       valid_size = 0.05, test_size = 0.05, valid_tiles=None, valid_files=None):
    assert large_patch_size % train_patch_size == 0
    tensor_files = get_images_path(data_dir, valid_tiles=valid_tiles, valid_files=valid_files)
    train_clean_patches = []
    valid_clean_patches = []
    test_clean_patches = []
    seed = 4235910
    for tensor_file in tensor_files:
        image_tensor = torch.load(tensor_file)
        patches = get_patches(image_tensor, large_patch_size)
        n_valid = int(patches.size(0) * valid_size)
        n_test = int(patches.size(0) * test_size)
        n_train = patches.size(0) - n_valid - n_test
        g_cpu = torch.Generator()
        g_cpu.manual_seed(seed)
        perms = torch.randperm(patches.size(0), generator=g_cpu) # For image tensor with identical sizes (i.e. the same sites) permutation will always be the same
        train_patches = get_clean_patch_tensor(patches[perms[:n_train],...], cloud_mask_idx=10, reject_mode='all')
        valid_patches = get_clean_patch_tensor(patches[perms[n_train:n_train + n_valid] ,...], cloud_mask_idx=10, reject_mode='all')
        test_patches = get_clean_patch_tensor(patches[perms[n_train + n_valid:],...], cloud_mask_idx=10, reject_mode='all')
        if len(train_patches) > 0:
            train_clean_patches.append(train_patches)
        if len(valid_patches) > 0:
            valid_clean_patches.append(valid_patches)
        if len(test_patches) > 0:
            test_clean_patches.append(test_patches)

    train_clean_patches = torch.cat(train_clean_patches, dim=0)
    train_clean_patches = patchify(unpatchify(train_clean_patches.unsqueeze(0)), patch_size=train_patch_size).reshape(-1,image_tensor.size(0), train_patch_size, train_patch_size)
    train_perms = torch.randperm(train_clean_patches.size(0), generator=g_cpu) 
    train_clean_patches = train_clean_patches[train_perms,...]
    valid_clean_patches = torch.cat(valid_clean_patches, dim=0)
    valid_clean_patches = patchify(unpatchify(valid_clean_patches.unsqueeze(0)), patch_size=train_patch_size).reshape(-1,image_tensor.size(0), train_patch_size, train_patch_size)
    valid_perms = torch.randperm(valid_clean_patches.size(0), generator=g_cpu) 
    valid_clean_patches = valid_clean_patches[valid_perms,...]
    test_clean_patches = torch.cat(test_clean_patches, dim=0)
    test_perms = torch.randperm(test_clean_patches.size(0), generator=g_cpu) 
    test_clean_patches = test_clean_patches[test_perms,...]
    print(f"Train patches : {train_clean_patches.size()}")
    print(f"Validation patches : {valid_clean_patches.size()}")
    print(f"Test patches : {test_clean_patches.size()}")
    return train_clean_patches, valid_clean_patches, test_clean_patches

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
                    # "after_SENTINEL2B_20171127-105827-648_L2A_T31TCJ_C_V2-2_roi_0.pth",
                   "before_SENTINEL2A_20180620-105211-086_L2A_T31TCJ_C_V2-2_roi_0.pth",
                #    "after_SENTINEL2A_20170711-111223-375_L2A_T30TUM_D_V1-7_roi_0.pth",
                #    "after_SENTINEL2A_20180417-110822-655_L2A_T30TUM_C_V2-2_roi_0.pth",
                #    "before_SENTINEL2A_20170518-095716-529_L2A_T33TWF_D_V1-4_roi_0.pth",
                #    "before_SENTINEL2A_20170408-095711-526_L2A_T33TWF_D_V1-4_roi_0.pth",
                #    "before_SENTINEL2A_20170518-095716-529_L2A_T33TWG_D_V1-4_roi_0.pth"
                ]

    (train_patches, 
     valid_patches, 
     test_patches) = get_train_valid_test_patch_tensors(data_dir=parser.data_dir, large_patch_size = large_patch_size, 
                                                        train_patch_size = train_patch_size, 
                                                        valid_size = valid_size, test_size = test_size,
                                                        valid_tiles=valid_tiles, valid_files=valid_files)
    torch.save(train_patches, os.path.join(parser.output_dir, "train_patches.pth"))
    torch.save(valid_patches, os.path.join(parser.output_dir, "valid_patches.pth"))
    torch.save(test_patches, os.path.join(parser.output_dir, "test_patches.pth"))
    return 

if __name__=='__main__':
    main()