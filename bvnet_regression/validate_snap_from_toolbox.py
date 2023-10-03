import os
import pandas as pd
import torch
import geopandas as gpd
import rasterio as rio
import numpy as np
from snap_regression.snap_utils import get_weiss_biophyiscal_from_batch
from sensorsio import sentinel2
NO_DATA=-10000
from rasterio.coords import BoundingBox
from shapely import Point

def theia_product_to_tensor(data_dir, s2_product_name, part_loading=1, top_left=None, n_pixels=None, top_left_crs='epsg:3857'):
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
    # s2_a = np.stack((sun_zen, joint_zen, sun_az - joint_az), 0)
    # print(s2_a.shape)
    if top_left is not None:
        left_top_df = gpd.GeoDataFrame(data={"geometry":[Point(top_left[0], 
                                                               top_left[1])]}, 
                                       crs=top_left_crs, 
                                       geometry="geometry").to_crs(dataset.crs)
        res = 10
        top = left_top_df.geometry.values.y // res * res 
        left = left_top_df.geometry.values.x // res * res
        
        left_idx = int((left - dataset.bounds.left) // res)
        top_idx = int((dataset.bounds.top - top) // res)
        # print(top_idx, n_pixels, left_idx)
        # print( s2_a[:, top_idx:(top_idx+n_pixels),:].shape)
        # print( s2_a[:, :, left_idx:(left_idx+n_pixels)].shape)
        joint_zen = joint_zen[..., top_idx:(top_idx + n_pixels), left_idx:(left_idx + n_pixels)]
        joint_az = joint_az[..., top_idx:(top_idx + n_pixels), left_idx:(left_idx + n_pixels)]
        sun_zen = sun_zen[..., top_idx:(top_idx + n_pixels), left_idx:(left_idx + n_pixels)]
        sun_az = sun_az[..., top_idx:(top_idx + n_pixels), left_idx:(left_idx + n_pixels)]
        
        bounding_box = BoundingBox(left, top - n_pixels * res, left + n_pixels * res, top)
        assert bounding_box.left < dataset.bounds.right and bounding_box.left > dataset.bounds.left
        
        s2_r, masks, atm, xcoords, ycoords, crs = dataset.read_as_numpy(bands, bounds=bounding_box, crs=dataset.crs,
                                                                        band_type=dataset.SRE)
        s2_r = s2_r.data    
        masks = masks.data
    else: 
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
    snap_lai_path = "/home/yoel/Téléchargements/S2B_MSIL2A_20180625T105029_N0208_R051_T31TCJ_20180625T135630_resampled_biophysical.tif"
    product_path = "/home/yoel/Documents/Dev/PROSAIL-VAE/used_theia_tiles_samples/"
    product_name = "SENTINEL2B_20180625-105253-379_L2A_T31TCJ_D_V2-2"
    with rio.open(snap_lai_path, mode = 'r') as src:
        # masked_array, _ = mask(src, [polygon], invert=False)
        array = src.read()
        pass
    tile_tensor = theia_product_to_tensor(product_path, product_name, part_loading=1, top_left=None, n_pixels=None, top_left_crs='epsg:3857')


if __name__ == "__main__":
    main()