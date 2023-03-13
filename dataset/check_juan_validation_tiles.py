import os
import sys
import glob
from sensorsio.utils import rgb_render
from torchutils import patches
from shapely.geometry import Point
import geopandas as gpd
import rasterio as rio
import torch
import matplotlib.pyplot as plt

# export_sample = '/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/validation_tiles/after_SENTINEL2A_20171122-105612-379_L2A_T31TCJ_C_V2-2_roi_0.pth'
# export_sample = '/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/validation_tiles/after_SENTINEL2A_20170522-110912-028_L2A_T30TUM_D_V1-4_roi_0.pth'
export_sample = '/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/validation_tiles/after_SENTINEL2B_20171127-105827-648_L2A_T31TCJ_C_V2-2_roi_0.pth'

point_data = "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/field_data/processed/after_france_T31TCJ_gai.gpkg"
# point_data = "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/field_data/processed/after_spain_T30TUM_2017.gpkg"

tensor = torch.load(export_sample) # dim : N x H x L, N = 10 bands + 1 mask + 4 angles (sun_zen, sun_azi, join_zen, join_azi)
samples = gpd.read_file(point_data)
samples_2017 = samples.iloc[(samples['field_date'].apply(lambda x: x.year)==2017).values]

samples_point = samples_2017.geometry.to_list()
gdf = gpd.GeoDataFrame(geometry=samples_point, crs=32631)

tensor_visu, minvisu, maxvisu = rgb_render(tensor)

fig, ax = plt.subplots(figsize=(5 * tensor_visu.shape[1]/tensor_visu.shape[0], 5))
plt.imshow(tensor_visu)
plt.show()

fig, ax = plt.subplots(figsize=(5 * tensor_visu.shape[1]/tensor_visu.shape[0], 5))
plt.imshow(tensor[12])
plt.colorbar()
plt.show()

ax = gdf.plot(figsize=(5 * tensor_visu.shape[1]/tensor_visu.shape[0], 5))