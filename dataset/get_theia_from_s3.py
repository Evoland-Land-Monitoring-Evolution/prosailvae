from sensorsio import sentinel2
from sensorsio.utils import rgb_render
import matplotlib.pyplot as plt
# from sensorsio import sentinel2
import os
import geopandas as gpd
from shapely.geometry import Polygon, Point
from rasterio.coords import BoundingBox
ROOT = "/home/uz/zerahy/"
import argparse
import socket
import pandas as pd
import datetime
import time
import shutil
import numpy as np
import torch

def get_parser():
    """
    Creates a new argument parser.
    """
    parser = argparse.ArgumentParser(description='Parser for data generation')

    parser.add_argument("-t", dest="tile",
                        help="tile to retrieve images from",
                        type=str, default="")
    parser.add_argument("-o", dest="output_dir",
                        help="directory to put files in",
                        type=str, default="")
    parser.add_argument("-m", dest="mask_max_percentage",
                        help="maximum invalid pixels in retrieved images",
                        type=float, default=0.05)
    return parser

MONTHS_TO_RETRIEVE = ["2016-02-01",
                      "2016-06-01",
                      "2016-10-01",
                      "2017-03-01",
                      "2017-07-01",
                      "2017-11-01",
                      "2018-04-01",
                      "2018-08-01",
                      "2018-10-01",
                      "2019-01-01",
                      "2019-05-01",
                      "2019-09-01"]

TILES_BB = {"32ULV": {'bb_left_top':[[ 717249, 6273008]], 'crs':"epsg:32632"},
            "31UFS": {'bb_left_top':[[ 527850, 6586729]], 'crs':"epsg:32631"},
            "31UDP": {'bb_left_top':[[ 337026, 6198482]], 'crs':"epsg:32631"},
            "30UWU": {'bb_left_top':[[-244922, 6127419]], 'crs':"epsg:32630"},
            "30TXQ": {'bb_left_top':[[-116496, 5627644]], 'crs':"epsg:32630"}, #EPSG3857
            "31TFJ": {'bb_left_top':[[ 503783, 5435881]], 'crs':"epsg:32631"},
            "33TWF": {'bb_left_top':[[1741039, 5049233], [1732470, 4977074]], 'crs':"epsg:32633"}, #[1736870, 4978840]
            "32TPQ": {'bb_left_top':[[1245580, 5625487]], 'crs':"epsg:32632"},
            "30TUM": {'bb_left_top':[[-537544, 5184355]], 'crs':"epsg:32630"},
            "30SWJ": {'bb_left_top':[[-261934, 4752698]], 'crs':"epsg:32630"},
            "30SVG": {'bb_left_top':[[-427431, 4456797]], 'crs':"epsg:32630"},
            "30STE": {'bb_left_top':[[-660596, 4232187]], 'crs':"epsg:32630"},
            "33SVB": {'bb_left_top':[[1576370, 4500670]], 'crs':"epsg:32633"},
            "31UCS": {'bb_left_top':[[  74351, 6662326]], 'crs':"epsg:32631"},
            "30SWJ": {'bb_left_top':[[-212019, 4739549]], 'crs':"epsg:32630"},
            "31TCJ": {'bb_left_top':[[ 190680, 5388953], [86908, 5441369], [109410,5462607]], 'crs':"epsg:32631"}}

def get_bb_from_left_top(left, top, size = 5120):
    bottom = top - size
    right = left + size
    return BoundingBox(left, bottom, right, top)

def get_polygon_from_left_top(left, top, size = 5120):
    bottom = top - size
    right = left + size
    y_list = [top, bottom, bottom, top, top]
    x_list = [left, left, right, right, left]
    return Polygon(zip(x_list, y_list))

def get_sites_polygons(tiles_bb, tiles=None, in_crs="epsg:3857", size=5120):
    polygons_gpd_list = []
    if tiles is None:
        tiles = tiles_bb.keys()
    for tile in tiles:
        tiles.append(tile)
        left_top = gpd.GeoDataFrame(data={"name":[tile],
                                          "geometry":[Point(tiles_bb[tile]['bb_left_top'][0][0], 
                                                            tiles_bb[tile]['bb_left_top'][0][1])]}, 
                                          crs=in_crs, geometry="geometry").to_crs(tiles_bb[tile]["crs"])
        polygons = get_polygon_from_left_top(left_top.geometry.values.x, left_top.geometry.values.y, size=size)
        polygons_gpd_list.append(gpd.GeoDataFrame(data={"name":[tile], "geometry":[polygons]},
                                                  crs=tiles_bb[tile]["crs"], geometry="geometry"))
    return tiles, polygons_gpd_list


def get_sites_bb(tiles_bb, tiles=None, in_crs="epsg:3857", size=5120):
    bb_list = []
    tiles_list = []
    if tiles is None:
        tiles = tiles_bb.keys()
    for tile in tiles:
        tiles_list.append(tile)
        left_top = gpd.GeoDataFrame(data={"name":[tile],
                                          "geometry":[Point(tiles_bb[tile]['bb_left_top'][0][0], 
                                                            tiles_bb[tile]['bb_left_top'][0][1])]}, 
                                          crs=in_crs, geometry="geometry").to_crs(tiles_bb[tile]["crs"])
        bb = get_bb_from_left_top(left_top.geometry.values.x, left_top.geometry.values.y, size=size)
        bb_list.append(bb)
    return tiles_list, bb_list


def write_s3_id_file(s3_id_file_path, s3_id):
    df = pd.DataFrame(data={"s3_id":[s3_id]})
    df.to_csv(s3_id_file_path, mode='a', index=False, header=False)

def get_checked_s3(s3_id_file_path):
    if os.path.isfile(s3_id_file_path):
        df = pd.read_csv(s3_id_file_path, header=None)
        return df.values.reshape(-1).tolist()
    return []

def check_mask(mask, max_percentage=0.05):
    mask_sum = mask.sum(0).astype(bool).astype(int)
    max_unvalid_pixels = max_percentage * mask.shape[1] * mask.shape[2]
    print(f"Unvalid_pixels on ROI : {mask_sum.sum()} / {mask.shape[1] * mask.shape[2]} ({mask_sum.sum() / (mask.shape[1] * mask.shape[2]) * 100} %)")
    if mask_sum.sum() >= max_unvalid_pixels:
        return False
    return True

def fix_product_name(path, product_name):
    all_dir = [ name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name)) ]
    if product_name in all_dir:
        return product_name
    if product_name[-6] == "C":
        if product_name.replace('_C_', '_D_') in all_dir:
            return product_name.replace('_C_', '_D_')
        else:
            raise ValueError
    if product_name[-6] == "D":
        if product_name.replace('_D_', '_C_') in all_dir:
            return product_name.replace('_D_', '_C_')
        else:
            raise ValueError(product_name, product_name[-6], all_dir)
    raise ValueError(product_name)
        

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
    print(f"Tile Tensor completed - tensor size: {tile_tensor.shape}")
    return torch.from_numpy(tile_tensor)   

def main():
    args=["-t", "30TUM",
          "-m", "0.01",
          "-o", "/home/yoel/Téléchargements/tile_s2"]
    parser = get_parser().parse_args(args)
    # parser = get_parser().parse_args()
    #for tile in ["30SVG", "30STE", "33SVB", "31UCS"]:
    list_product = ["SENTINEL2A_20170423-104254-989_L2A_T31TCJ_D",
                    "SENTINEL2A_20170526-105518-082_L2A_T31TCJ_D",
                    "SENTINEL2A_20170705-105605-592_L2A_T30SWJ_D",
                    "SENTINEL2B_20180811-104744-821_L2A_T31TCJ_D",
                    "SENTINEL2A_20180511-105804-037_L2A_T31TCJ_D",
                    "SENTINEL2B_20180821-104015-463_L2A_T31TCJ_D",
                    "SENTINEL2B_20180314-104014-461_L2A_T31TCJ_D",
                    "SENTINEL2A_20170622-104021-457_L2A_T31TCJ_D",
                    "SENTINEL2B_20180625-105253-379_L2A_T31TCJ_D",
                    "SENTINEL2B_20180426-105202-871_L2A_T30SWJ_D",
                    "SENTINEL2B_20180824-105058-149_L2A_T30SWJ_D",
                    "SENTINEL2A_20170506-105029-462_L2A_T30SWJ_D",
                    "SENTINEL2A_20180727-104023-458_L2A_T31TCJ_D",
                    "SENTINEL2B_20180725-105415-357_L2A_T30SWJ_D",
                    "SENTINEL2B_20170717-104757-036_L2A_T31TCJ_D",
                    "SENTINEL2B_20180625-105253-379_L2A_T30SWJ_D",
                    "SENTINEL2A_20170814-105517-079_L2A_T30SWJ_D",
                    "SENTINEL2A_20170605-105303-597_L2A_T30SWJ_D",
                    "SENTINEL2A_20170406-105317-631_L2A_T30SWJ_D",]
    download = True
    if not download:
        for tile in ["30SWJ", "31TCJ"]:
        #for tile in ["33TWF", "32TPQ", "30TUM", "30SVJ"]:
            tiles, bb_list = get_sites_bb(TILES_BB, tiles=[tile], in_crs="epsg:3857", size=5120)
    else:
        
        for tile in ["31TCJ", "30SWJ"]:
        #for tile in TILES_BB.keys():
            tile_dir = os.path.join(parser.output_dir,"T"+tile)
            tensor_dir = os.path.join("/home/yoel/Téléchargements/tile_s2/torch_files","T" + tile)
            list_tile_product = []
            for d in os.listdir(tile_dir):
                if os.path.isdir(os.path.join(tile_dir, d)) and d[:8]=="SENTINEL":
                    list_tile_product.append(d)
            for product_name in list_tile_product:
                for roi, top_left in enumerate(TILES_BB[tile]['bb_left_top']):
                    # product_name = s3_id.split('/')[-2]
                    # product_name = fix_product_name(tile_dir, product_name)
                    #product_name = product_name.replace("_D_V", "_C_V")
                    product_dir = os.listdir(os.path.join(tile_dir, product_name))[0]

                    product_tensor = theia_product_to_tensor(os.path.join(tile_dir, product_name), 
                                                             product_dir, part_loading=1, top_left=top_left, n_pixels=512)
                    if not product_tensor.isnan().any():
                        tensor_visu, _, _ = rgb_render(product_tensor.squeeze())
                        fig, ax = plt.subplots(dpi=150, figsize=(6,6))
                        im = ax.imshow(tensor_visu)
                        ax.set_title(product_name)
                        fig.savefig(os.path.join(tensor_dir, product_name + f'_ROI_{roi}.png'))
                        print(f"Saving tensor file at {os.path.join(tensor_dir, f'{product_name}_ROI_{roi}.pth')}")
                        torch.save(product_tensor, os.path.join(tensor_dir, f'{product_name}_ROI_{roi}.pth'))
                    else:
                        print("NaN tensor !")
                # shutils.rmtree(os.path.join(tensor_dir, product))

if __name__ == "__main__":
    main()
