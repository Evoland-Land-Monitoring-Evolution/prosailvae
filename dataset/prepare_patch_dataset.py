import torch
import os
from torchutils.patches import patchify, unpatchify 
import argparse
import socket
import numpy as np
from sensorsio import sentinel2
from utils.image_utils import rgb_render
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from rasterio.coords import BoundingBox
import datetime
from prosailvae.spectral_indices import get_spectral_idx
from metrics.prosail_plots import pair_plot

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
    
    parser.add_argument("-t", dest="theia",
                        help="toggle option for theia product as input",
                        type=bool, default=False)
    return parser



def get_images_path(data_dir, valid_tiles=None, valid_files=None, invalid_files=None):
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
                if invalid_files is not None:
                    if tile_file in invalid_files:
                        print(f"Excluding {tile_file}")
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
        # return 5000, None, 5000, None
        return 0, None, 0, None
        # raise NotImplementedError

def get_patches(image_tensor, patch_size):
    try:
        patches = patchify(image_tensor, patch_size=patch_size, margin=0)
    except Exception as exc:
        print(exc)
        print(image_tensor.size())
        raise ValueError
    if image_tensor.size(1) % patch_size != 0:
        patches = patches[:-1,...]
    if image_tensor.size(2) % patch_size != 0:
        patches = patches[:,:-1,...]
    patches = patches.reshape(-1, image_tensor.size(0), patch_size, patch_size)
    return patches

def get_rectangle_bounds_from_patch_info(patch_idx, patch_size=128, image_size = (512, 512)):
    # n_row = image_size[0] // patch_size
    n_col = image_size[1] // patch_size
    col_idx = patch_idx % n_col
    line_idx = patch_idx // n_col
    ymin = line_idx * patch_size
    ymax = (line_idx + 1) * patch_size
    xmin = col_idx * patch_size
    xmax = (col_idx + 1) * patch_size
    return xmin, xmax, ymin, ymax

def plot_patch_rectangles(ax, patch_idxs, patch_size, image_size, color='red'):

    for idx in patch_idxs:
        xmin, xmax, ymin, ymax = get_rectangle_bounds_from_patch_info(idx,
                                                                      patch_size=patch_size, 
                                                                      image_size=image_size)
        ax.add_patch(Rectangle((xmin,ymin),xmax-xmin,ymax-ymin,
                        edgecolor=color,
                        facecolor='none',
                        lw=1))
    return

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
                print(f"Non-valid pixels in patch {i}: {validity.sum()}")
        else:
            raise NotImplementedError
    if len(clean_patches) > 0:
        clean_patches = torch.cat(clean_patches, dim=0)
    if nan_flag:
        print("WARNING: patches with nan values were detected in this data !")
    return clean_patches, nan_flag

def get_all_images_norm_factor(tensor_files):
    all_tensors = []
    for _, tensor_file in enumerate(tensor_files):
        all_tensors.append(torch.load(tensor_file))
    all_tensors = torch.cat(all_tensors, 2)
    _, dmin, dmax= rgb_render(all_tensors)
    return dmin, dmax

def get_train_valid_test_patch_tensors(data_dir, large_patch_size = 128, train_patch_size = 32, 
                                       valid_size = 0.05, test_size = 0.05, valid_tiles=None, 
                                       valid_files=None, invalid_files=None,res_dir=None):
    assert large_patch_size % train_patch_size == 0
    tensor_files, file_info = get_images_path(data_dir, valid_tiles=valid_tiles, valid_files=valid_files, invalid_files=invalid_files)
    train_clean_patches = []
    valid_clean_patches = []
    test_clean_patches = []
    seed = 4235910
    train_patch_info = []
    valid_patch_info = []
    test_patch_info = []
    list_valid_image_files = []
    list_invalid_image_files = []
    if res_dir is not None:
        for i, tensor_file in enumerate(tensor_files):
            date = datetime.datetime.strptime(file_info[i][1], '%Y%m%d').date()
    # if res_dir is not None:
    #     dmin, dmax = get_all_images_norm_factor(tensor_files)
    for i, tensor_file in enumerate(tensor_files):
        info = file_info[i]
        print(tensor_file)
        image_tensor = torch.load(tensor_file)
        print(image_tensor.size())

        min_x, max_x, min_y, max_y = get_valid_area_in_image(info[2])
        # if max_x is not None and max_y is not None:
        #     image_tensor = image_tensor[:,min_x: max_x, min_y: max_y]
        patches = get_patches(image_tensor, large_patch_size)
        n_valid = max(int(patches.size(0) * valid_size), 1)
        n_test = max(int(patches.size(0) * test_size), 1)
        n_train = patches.size(0) - n_valid - n_test
        g_cpu = torch.Generator()
        g_cpu.manual_seed(seed)
        perms = torch.randperm(patches.size(0), generator=g_cpu) # For image tensor with identical sizes (i.e. the same sites) permutation will always be the same
        
        train_patches, nan_flag_1 = get_clean_patch_tensor(patches[perms[:n_train], ...],
                                                           cloud_mask_idx=10, reject_mode='all')
        valid_patches, nan_flag_2 = get_clean_patch_tensor(patches[perms[n_train:n_train + n_valid], ...],
                                                            cloud_mask_idx=10, reject_mode='all')
        test_patches, nan_flag_3 = get_clean_patch_tensor(patches[perms[n_train + n_valid:], ...],
                                                            cloud_mask_idx=10, reject_mode='all')
        if nan_flag_1 or nan_flag_2 or nan_flag_3:
            list_invalid_image_files.append(tensor_file)
            print(f"{tensor_file} is NaN!")
        else:
            list_valid_image_files.append(tensor_file)
            print(f"{tensor_file} is OK!")

        if len(train_patches) > 0:
            train_clean_patches.append(train_patches)
            train_patch_info += [info] * n_train * (large_patch_size // train_patch_size)**2

        if len(valid_patches) > 0:
            valid_clean_patches.append(valid_patches)
            valid_patch_info += [info] * n_valid * (large_patch_size // train_patch_size)**2

        if len(test_patches) > 0:
            test_clean_patches.append(test_patches)
            test_patch_info += [info] * n_test

        if res_dir is not None:
            mask = image_tensor[10].numpy()
            mask[mask==0.] = np.nan
            fig, ax = plt.subplots(dpi=150, tight_layout=True, figsize=(6, 6))
            ax.imshow(rgb_render(image_tensor)[0])
            ax.imshow(mask.squeeze(), cmap='YlOrRd')
            plot_patch_rectangles(ax, perms[:n_train], patch_size=large_patch_size,
                        image_size=((image_tensor.size(1)//large_patch_size)*large_patch_size,
                                    (image_tensor.size(2)//large_patch_size)*large_patch_size), 
                                    color='red')

            plot_patch_rectangles(ax, perms[n_train:n_train + n_valid], patch_size=large_patch_size,
                        image_size=((image_tensor.size(1)//large_patch_size)*large_patch_size,
                                    (image_tensor.size(2)//large_patch_size)*large_patch_size), 
                                    color='blue')
            plot_patch_rectangles(ax, perms[n_train + n_valid:], patch_size=large_patch_size,
                                  image_size=((image_tensor.size(1)//large_patch_size)*large_patch_size,
                                              (image_tensor.size(2)//large_patch_size)*large_patch_size), 
                                              color='green')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{info[2]} {info[1]}")
            fig.savefig(os.path.join(res_dir, f"full_roi_{info[2]}_{info[1]}.png"))
            plt.close('all')
    # raise NotImplementedError


    valid_clean_patches = torch.cat(valid_clean_patches, dim=0)
    valid_clean_patches = patchify(unpatchify(valid_clean_patches.unsqueeze(0)), 
                                   patch_size=train_patch_size).reshape(-1,image_tensor.size(0), 
                                                                        train_patch_size, train_patch_size)
    valid_perms = torch.randperm(valid_clean_patches.size(0), generator=g_cpu)

    valid_clean_patches = valid_clean_patches[valid_perms,...]
    valid_patch_info = np.array(valid_patch_info)[valid_perms,:]

    train_clean_patches = torch.cat(train_clean_patches, dim=0)
    train_clean_patches = patchify(unpatchify(train_clean_patches.unsqueeze(0)), 
                                   patch_size=train_patch_size).reshape(-1,image_tensor.size(0), 
                                                                        train_patch_size, train_patch_size)
    train_perms = torch.randperm(train_clean_patches.size(0), generator=g_cpu)
    train_clean_patches = train_clean_patches[train_perms,...]
    train_patch_info = np.array(train_patch_info)[train_perms,:]
    # if valid_size_for_small_patches:
    #     train_patches_from_valid = valid_clean_patches[(valid_perms.size(0)*train_patch_size)//large_patch_size:,...]
    #     train_clean_patches = torch.cat((train_clean_patches, train_patches_from_valid), dim=0)
    #     train_info_from_valid = valid_patch_info[(valid_perms.size(0)*train_patch_size)//large_patch_size:]
    #     train_patch_info = np.concatenate((train_patch_info, train_info_from_valid))
    #     valid_clean_patches = valid_clean_patches[:(valid_perms.size(0)*train_patch_size)//large_patch_size,...]
    #     valid_patch_info = valid_patch_info[:(valid_perms.size(0)*train_patch_size)//large_patch_size]

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
    if len(list_invalid_image_files) > 0:
        print("invalid files :")
        for file in list_invalid_image_files:
            print(file)
    return (train_clean_patches, valid_clean_patches, test_clean_patches,
            train_patch_info, valid_patch_info, test_patch_info)

def swap_bands(patches):
    """
    Put bands in reflectance order in patches, instead of resolution order
    """
    idx = torch.LongTensor([ v for _, (_,v) in enumerate(BANDS_IDX.items()) ])
    patches[:,torch.arange(10),...] = patches[:,idx,...]
    return



def min_max_to_loc_scale(minimum, maximum):
    loc = (maximum + minimum) / 2
    scale = (maximum - minimum) / 2
    return loc, scale


def get_bands_norm_factors_from_patches(patches, n_bands=10, mode='mean'):
    cos_angle_min = torch.tensor([0.342108564072183, 0.979624800125421, -1.0000]) # sun zenith, S2 senith, relative azimuth
    cos_angle_max = torch.tensor([0.9274847491748729, 1.0000, 1.0000])
    with torch.no_grad():
        # s2_a = torch.zeros(patches.size(0), 3, patches.size(2), patches.size(3))   
        # s2_a[:,0,...] = patches[:,11,0,0] # sun zenith
        # s2_a[:,1,...] = patches[:,13,0,0] # joint zenith
        # s2_a[:,2,...] = patches[:,12,0,0] 
        # s2_a[:,3,...] = patches[:,14,0,0] # Sun azimuth - joint azimuth 
        # fig, ax = plt.subplots(1,4, dpi=150)
        # for i in range(4):
        #     ax[i].hist(s2_a[:,i].squeeze().detach().cpu().numpy(), bins = 100, )
        # s2_a_rad = torch.deg2rad(s2_a)
        # s2_a_cos_sin = torch.cat((torch.cos(s2_a_rad), torch.sin(s2_a_rad)), 1)
        # s2_a_samples = s2_a_cos_sin.permute(1,0,2,3).reshape(6, -1)
        
        spectral_idx = get_spectral_idx(patches[:, :n_bands,...], bands_dim=1).permute(1,0,2,3).reshape(5, -1)
        s2_r_samples = patches.permute(1,0,2,3)[:n_bands, ...].reshape(n_bands, -1)
        if mode=='mean':
            norm_mean = s2_r_samples.mean(1)
            norm_std = s2_r_samples.std(1)
            # angles_norm_mean = s2_a_samples.mean(1)
            # angles_norm_std = s2_a_samples.std(1)
            idx_norm_mean = spectral_idx.mean(1)
            idx_norm_std = spectral_idx.std(1)
            
        elif mode=='quantile':
            max_samples=int(1e7)
            norm_mean = torch.quantile(s2_r_samples[:, :max_samples], q=torch.tensor(0.5), dim=1)
            norm_std = torch.quantile(s2_r_samples[:, :max_samples], q=torch.tensor(0.95), dim=1) - torch.quantile(s2_r_samples[:, :max_samples], q=torch.tensor(0.05), dim=1)
            # angles_norm_mean = torch.quantile(s2_a_samples[:, :max_samples], q=torch.tensor(0.5), dim=1)
            # angles_norm_std = torch.quantile(s2_a_samples[:, :max_samples], q=torch.tensor(0.95), dim=1) - torch.quantile(s2_a_samples[:, :max_samples], q=torch.tensor(0.05), dim=1)
            idx_norm_mean = torch.quantile(spectral_idx[:, :max_samples], q=torch.tensor(0.5), dim=1)
            idx_norm_std = torch.quantile(spectral_idx[:, :max_samples], q=torch.tensor(0.95), dim=1) - torch.quantile(spectral_idx[:, :max_samples], q=torch.tensor(0.05), dim=1)

        cos_angles_loc, cos_angles_scale = min_max_to_loc_scale(cos_angle_min, cos_angle_max)

    return norm_mean, norm_std, cos_angles_loc, cos_angles_scale, idx_norm_mean, idx_norm_std


def get_info_from_filename(filename, prefix=False):
    if prefix:
        filename_comp = filename.split("_")
        filename = "_".join(filename_comp[1:])
    filename_comp = filename.split("_")
    if filename_comp[0] == "SENTINEL2A":
        sensor = "2A"
    elif filename_comp[0] == "SENTINEL2B":
        sensor = "2B"
    else:
        raise ValueError("Sensor name not found!")
    date = filename_comp[1].split("-")[0]
    tile = filename_comp[3]
    return sensor, date, tile, sensor + date + tile


def theia_product_to_tensor(data_dir, s2_product_name, part_loading=1):
    path_to_theia_product = os.path.join(data_dir, s2_product_name)
    print(path_to_theia_product)
    dataset = sentinel2.Sentinel2(path_to_theia_product)
    bands = [sentinel2.Sentinel2.B2,
             sentinel2.Sentinel2.B3,
             sentinel2.Sentinel2.B4,
             sentinel2.Sentinel2.B8,
             sentinel2.Sentinel2.B5,
             sentinel2.Sentinel2.B6,
             sentinel2.Sentinel2.B7,
             sentinel2.Sentinel2.B8A,
             sentinel2.Sentinel2.B11,
             sentinel2.Sentinel2.B12]
    even_zen, odd_zen, even_az, odd_az = dataset.read_incidence_angles_as_numpy()
    joint_zen = np.array(even_zen)
    joint_zen[np.isnan(even_zen)] = odd_zen[np.isnan(even_zen)]
    del even_zen
    del odd_zen
    joint_az = np.array(even_az)
    joint_az[np.isnan(even_az)] = odd_az[np.isnan(even_az)]
    del even_az
    del odd_az
    sun_zen, sun_az = dataset.read_solar_angles_as_numpy()
    s2_a = np.stack((sun_zen, joint_zen, sun_az - joint_az), 0).data
    print(s2_a.shape)
    bb = dataset.bounds
    if part_loading > 1:
        s2_r_list = []
        masks_list = []
        top_bottom_range = (dataset.bounds.top - dataset.bounds.bottom) // part_loading
        for i in range(part_loading-1):
            bb = BoundingBox(dataset.bounds.left,
                             dataset.bounds.bottom + i * top_bottom_range,
                             dataset.bounds.right,
                             dataset.bounds.bottom + (i+1) * top_bottom_range)
            try:
                s2_r, masks, _, _, _, _ = dataset.read_as_numpy(bands, bounds=bb, crs=dataset.crs,
                                                                band_type=dataset.SRE)
                print(i)
            except Exception as exc:
                print(i, bb, top_bottom_range)
            s2_r_list.append(s2_r.data)
            masks_list.append(masks.data)
        bb = BoundingBox(dataset.bounds.left,
                            dataset.bounds.bottom + (part_loading-1) * top_bottom_range, 
                            dataset.bounds.right,
                            dataset.bounds.top)
        s2_r, masks, _, _, _, _ = dataset.read_as_numpy(bands, bounds=bb, crs=dataset.crs,
                                                        band_type=dataset.SRE)
        s2_r_list.append(s2_r.data)
        masks_list.append(masks.data)
        s2_r = np.concatenate(s2_r_list, 1)
        masks = np.concatenate(masks_list, 1)
    else:
        s2_r, masks, _, _, _, _ = dataset.read_as_numpy(bands, bounds=bb, crs=dataset.crs,
                                                        band_type=dataset.SRE)
        s2_r = s2_r.data    
        masks = masks.data
    w = s2_r.shape[1]
    h = s2_r.shape[2]
    validity_mask = np.sum(masks, axis=0, keepdims=True).astype(bool).astype(int).astype(float)
    tile_tensor = np.concatenate((s2_r, validity_mask,
                                  sun_zen.reshape((1,w,h)),
                                  sun_az.reshape((1,w,h)),
                                  joint_zen.reshape((1,w,h)),
                                  joint_az.reshape((1,w,h))))
    print("Tile Tensor completed")
    return torch.from_numpy(tile_tensor)

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
    if parser.theia:
        valid_tiles = ["barrax_theia"]
        valid_files = ["SENTINEL2B_20180516-105351-101_L2A_T30SWJ_D_V1-7",
                       "SENTINEL2A_20180613-110957-425_L2A_T30SWJ_D_V1-8"]
        for i, product in enumerate(valid_files):
            # product_tensor = theia_product_to_tensor(parser.data_dir, product, part_loading=10)
            # if not os.path.isdir(os.path.join(parser.data_dir, valid_tiles[0])):
            #     os.makedirs(os.path.join(parser.data_dir, valid_tiles[0]))
            # print("Saving tensor file at ...")
            # torch.save(product_tensor, os.path.join(os.path.join(parser.data_dir, valid_tiles[0]),
            #                                         "theia_" + product + ".pth"))
            # print("Tensor file Saved!")
            valid_files[i] = "theia_" + product + ".pth"
    else:
        valid_tiles = ["T31TCJ", "T30TUM", "T33TWF", "T33TWG", "T30SWJ"]
        valid_files = [ "after_SENTINEL2B_20171127-105827-648_L2A_T31TCJ_C_V2-2_roi_0.pth",
                        "before_SENTINEL2A_20180620-105211-086_L2A_T31TCJ_C_V2-2_roi_0.pth",
                        "after_SENTINEL2A_20170711-111223-375_L2A_T30TUM_D_V1-7_roi_0.pth",
                        "after_SENTINEL2A_20180417-110822-655_L2A_T30TUM_C_V2-2_roi_0.pth",
                        "before_SENTINEL2A_20170518-095716-529_L2A_T33TWF_D_V1-4_roi_0.pth",
                        "before_SENTINEL2A_20170408-095711-526_L2A_T33TWF_D_V1-4_roi_0.pth",
                        "before_SENTINEL2A_20170518-095716-529_L2A_T33TWG_D_V1-4_roi_0.pth",
                        # "theia_SENTINEL2B_20180516-105351-101_L2A_T30SWJ_D_V1-7.pth",
                        # "theia_SENTINEL2A_20180613-110957-425_L2A_T30SWJ_D_V1-8.pth"
                    ]
        valid_files = None
        invalid_files = ["SENTINEL2A_20161001-110717-761_L2A_T30SVJ_D_V1-4.pth",
                         "SENTINEL2B_20190904-111941-270_L2A_T30TUM_C_V2-2.pth",
                         "SENTINEL2A_20161011-110223-381_L2A_T30TXQ_D_V1-1.pth",
                         "SENTINEL2B_20190203-110848-004_L2A_T30TXQ_C_V2-2.pth",
                         "SENTINEL2A_20161103-111224-460_L2A_T30UWU_D_V1-1.pth",
                         "SENTINEL2A_20170313-111212-752_L2A_T30UWU_D_V1-4.pth",
                         "SENTINEL2A_20171109-104505-159_L2A_T31TFJ_C_V2-2.pth",
                         "SENTINEL2A_20160312-105037-460_L2A_T31UDP_D_V1-1.pth",
                         "SENTINEL2A_20161115-101301-462_L2A_T32TPQ_D_V1-4.pth",
                         "SENTINEL2A_20170219-103333-413_L2A_T32ULV_D_V1-4.pth",
                         "SENTINEL2A_20160304-095843-370_L2A_T33SVB_D_V1-4.pth",
                         "SENTINEL2A_20160622-095030-459_L2A_T33TWF_D_V1-4.pth",
                         "SENTINEL2A_20180801-095403-317_L2A_T33TWF_C_V2-2.pth",
                         "SENTINEL2A_20181215-101819-939_L2A_T32TPQ_C_V2-2.pth"]
        tiles = ["32ULV", # Vosges
                 "31UFS", # Belgique
                 "31UDP", # Ile de France
                 "30UWU", # Bretagne
                 "30TXQ", # Gironde
                 "31TFJ", # Provence
                 "31TCJ", # Toulouse
                 "33TWF", # Italie Sud
                 "32TPQ", # Italie Nord
                 "30TUM", # Espagne Nord
                 "30SVJ", # Espagne Centre (Barrax)
                 # "30SVG", # Andalousie
                #  "30STE", # Maroc
                #  "33SVB", # Sicille
                 "31UCS"] # Angleterre
        valid_tiles = ['theia_tensor']
    # valid_files = ["after_SENTINEL2A_20170621-111222-373_L2A_T30TUM_D_V1-4_roi_0.pth"]
    (train_patches, valid_patches, test_patches,
     train_patch_info, valid_patch_info,
     test_patch_info) = get_train_valid_test_patch_tensors(data_dir=parser.data_dir, large_patch_size=large_patch_size,
                                                           train_patch_size=train_patch_size,
                                                           valid_size=valid_size, test_size=test_size,
                                                           valid_tiles=valid_tiles, valid_files=valid_files, invalid_files=invalid_files,
                                                           res_dir=parser.output_dir)
    plot_test = True
    if plot_test:
        for i in range(test_patches.size(0)):
            info = test_patch_info[i]
            test_patch_i = test_patches[i,...].detach().cpu().numpy()
            fig, ax = plt.subplots(dpi=150)
            ax.imshow(rgb_render(test_patch_i)[0])
            fig.savefig(os.path.join(parser.output_dir, f"test_{info[0]}_{info[1]}_{info[2]}.png"))
            plt.close('all')
        pass
    mode = "quantile"
 
    sun_zen = train_patches[:,11,0,0] # sun zenith
    joint_zen = train_patches[:,13,0,0] # joint zenith
    sun_azi = train_patches[:,12,0,0]
    joint_azi = train_patches[:,14,0,0]
    rel_azi = train_patches[:,12,0,0] - train_patches[:,14,0,0] # Sun azimuth - joint azimuth 
    s2_a = torch.cat((joint_zen.unsqueeze(1), sun_zen.unsqueeze(1), rel_azi.unsqueeze(1)), 1)
    s2_a_rad = torch.deg2rad(s2_a)
    s2_a_cos_sin = (torch.cos(s2_a_rad))

    pair_plot(s2_a_cos_sin, tensor_2=None, features = ['Joint Zenith', "Sun Zenith", "Relative Azimuth"],
                    res_dir=parser.output_dir, filename='angles_pairplot.png')
    s2_a = torch.cat((sun_zen.unsqueeze(1), joint_zen.unsqueeze(1), sun_azi.unsqueeze(1), joint_azi.unsqueeze(1)), 1)
    pair_plot(s2_a, tensor_2=None, features = ['Sun Zenith', "S2 Zenith", "Sun Azimuth", "S2 Azimuth"],
                res_dir=parser.output_dir, filename='angles_deg_pairplot.png')
    spectral_idx = get_spectral_idx(train_patches[:, :10,...], bands_dim=1).permute(1,0,2,3).reshape(5, -1)
    perm = torch.randperm(spectral_idx.size(1))
    pair_plot(spectral_idx.permute(1,0)[perm[:1000000],:], tensor_2=None, 
              features = ["NDVI", "CRI2", "NDII", "ND_lma", "LAI_savi"],
                res_dir=parser.output_dir, filename='spectral_idx_pairplot.png')
    (norm_mean, norm_std, angles_norm_mean, angles_norm_std, idx_norm_mean, 
        idx_norm_std) = get_bands_norm_factors_from_patches(train_patches, mode=mode)
    print(f"median {norm_mean}, quantiles difference {norm_std}")

    
    torch.save(norm_mean, os.path.join(parser.output_dir, "norm_mean.pt"))
    torch.save(norm_std, os.path.join(parser.output_dir, "norm_std.pt"))
    torch.save(angles_norm_mean, os.path.join(parser.output_dir, "angles_loc.pt"))
    torch.save(angles_norm_std, os.path.join(parser.output_dir, "angles_scale.pt"))
    torch.save(idx_norm_mean, os.path.join(parser.output_dir, "idx_loc.pt"))
    torch.save(idx_norm_std, os.path.join(parser.output_dir, "idx_scale.pt"))

    print(f"Train patches : {train_patches.size(0)}")
    print(f"Valid patches : {valid_patches.size(0)}")
    print(f"Test patches : {test_patches.size(0)}")
    torch.save(train_patches, os.path.join(parser.output_dir, "train_patches.pth"))
    torch.save(valid_patches, os.path.join(parser.output_dir, "valid_patches.pth"))
    torch.save(test_patches, os.path.join(parser.output_dir, "test_patches.pth"))
    np.save(os.path.join(parser.output_dir, "train_info.npy"), train_patch_info)
    np.save(os.path.join(parser.output_dir, "valid_info.npy"), valid_patch_info)
    np.save(os.path.join(parser.output_dir, "test_info.npy"), test_patch_info)
    return 

if __name__=='__main__':
    main()