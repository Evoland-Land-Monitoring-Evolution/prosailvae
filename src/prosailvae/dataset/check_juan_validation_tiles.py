import glob
import os
import sys

import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio as rio
import torch
from shapely.geometry import Point
from torchutils import patches

from utils.image_utils import rgb_render

# from sensorsio.utils import rgb_render

# export_sample = '/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/validation_tiles/after_SENTINEL2A_20171122-105612-379_L2A_T31TCJ_C_V2-2_roi_0.pth'
# export_sample = '/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/validation_tiles/after_SENTINEL2A_20170522-110912-028_L2A_T30TUM_D_V1-4_roi_0.pth'
export_sample = "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/validation_tiles/after_SENTINEL2B_20171127-105827-648_L2A_T31TCJ_C_V2-2_roi_0.pth"

point_data = "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/field_data/processed/after_france_T31TCJ_gai.gpkg"
# point_data = "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/field_data/processed/after_spain_T30TUM_2017.gpkg"

tensor = torch.load(
    export_sample
)  # dim : N x H x L, N = 10 bands + 1 mask + 4 angles (sun_zen, sun_azi, join_zen, join_azi)

# fig, axs = plt.subplots(utensor.size(1),utensor.size(2), tight_layout=True)
# for i in range(utensor.size(1)):
#     for j in range(utensor.size(2)):
#         utensor_visu, minvisu, maxvisu = rgb_render(utensor[:,i,j,:,:])
#         axs[i,j].imshow(utensor_visu)

from torchutils.patches import patchify, unpatchify

N = 200
M = 100
noise = torch.rand((1, N, M))
hw = 4
pn1 = patchify(noise, patch_size=32, margin=hw)
upns = unpatchify(pn1[:, :, :, hw:-hw, hw:-hw])
fig, axs = plt.subplots(1, 3)
axs[0].imshow(noise[:, hw : N - hw, hw : M - hw].permute(1, 2, 0))
axs[1].imshow(upns[:, hw : N - hw, hw : M - hw].permute(1, 2, 0))
axs[2].imshow(
    (noise[:, hw : N - hw, hw : M - hw] - upns[:, hw : N - hw, hw : M - hw]).permute(
        1, 2, 0
    )
)

samples = gpd.read_file(point_data)
samples_2017 = samples.iloc[
    (samples["field_date"].apply(lambda x: x.year) == 2017).values
]

samples_point = samples_2017.geometry.to_list()
gdf = gpd.GeoDataFrame(geometry=samples_point, crs=32631)

tensor_visu, minvisu, maxvisu = rgb_render(tensor)

fig, ax = plt.subplots(figsize=(5 * tensor_visu.shape[1] / tensor_visu.shape[0], 5))
plt.imshow(tensor_visu)
plt.show()

fig, ax = plt.subplots(figsize=(5 * tensor_visu.shape[1] / tensor_visu.shape[0], 5))
plt.imshow(tensor[12])
plt.colorbar()
plt.show()

ax = gdf.plot(figsize=(5 * tensor_visu.shape[1] / tensor_visu.shape[0], 5))
