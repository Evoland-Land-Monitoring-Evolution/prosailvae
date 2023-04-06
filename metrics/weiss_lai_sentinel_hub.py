from math import pi 
import torch
from prosailvae.utils import torch_select_unsqueeze
    
def weiss_lai(s2_r, s2_a, band_dim = 1, bands_idx = None):
    if bands_idx is None:
        B03 = s2_r.select(band_dim, 1).unsqueeze(band_dim)
        B04 = s2_r.select(band_dim, 2).unsqueeze(band_dim)
        B05 = s2_r.select(band_dim, 3).unsqueeze(band_dim)
        B06 = s2_r.select(band_dim, 4).unsqueeze(band_dim)
        B07 = s2_r.select(band_dim, 5).unsqueeze(band_dim)
        B8A = s2_r.select(band_dim, 7).unsqueeze(band_dim)
        B11 = s2_r.select(band_dim, 8).unsqueeze(band_dim)
        B12 = s2_r.select(band_dim, 9).unsqueeze(band_dim)
    else:
        B03 = s2_r.select(band_dim, bands_idx['B03']).unsqueeze(band_dim)
        B04 = s2_r.select(band_dim, bands_idx['B04']).unsqueeze(band_dim)
        B05 = s2_r.select(band_dim, bands_idx['B05']).unsqueeze(band_dim)
        B06 = s2_r.select(band_dim, bands_idx['B06']).unsqueeze(band_dim)
        B07 = s2_r.select(band_dim, bands_idx['B07']).unsqueeze(band_dim)
        B8A = s2_r.select(band_dim, bands_idx['B8A']).unsqueeze(band_dim)
        B11 = s2_r.select(band_dim, bands_idx['B11']).unsqueeze(band_dim)
        B12 = s2_r.select(band_dim, bands_idx['B12']).unsqueeze(band_dim)
    viewZenithMean = s2_a.select(band_dim, 1).unsqueeze(band_dim)
    sunZenithAngles = s2_a.select(band_dim, 0).unsqueeze(band_dim)
    relAzim = s2_a.select(band_dim, 2).unsqueeze(band_dim)
    b03_norm = normalize(B03, 0, 0.253061520471542)
    b04_norm = normalize(B04, 0, 0.290393577911328)
    b05_norm = normalize(B05, 0, 0.305398915248555)
    b06_norm = normalize(B06, 0.006637972542253, 0.608900395797889)
    b07_norm = normalize(B07, 0.013972727018939, 0.753827384322927)
    b8a_norm = normalize(B8A, 0.026690138082061, 0.782011770669178)
    b11_norm = normalize(B11, 0.016388074192258, 0.493761397883092)
    b12_norm = normalize(B12, 0, 0.493025984460231)
    viewZen_norm = normalize(torch.cos(torch.deg2rad(viewZenithMean)), 0.918595400582046, 1)
    sunZen_norm  = normalize(torch.cos(torch.deg2rad(sunZenithAngles)), 0.342022871159208, 0.936206429175402)
    relAzim_norm = torch.cos(relAzim)

    n1 = neuron1(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm, sum_dim=band_dim)
    n2 = neuron2(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm, sum_dim=band_dim)
    n3 = neuron3(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm, sum_dim=band_dim)
    n4 = neuron4(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm, sum_dim=band_dim)
    n5 = neuron5(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm, sum_dim=band_dim)

    l2 = layer2(n1, n2, n3, n4, n5, sum_dim=band_dim)

    lai = denormalize(l2, 0.000319182538301, 14.4675094548151)
    return lai / 3


def neuron1(b03_norm, b04_norm, b05_norm, b06_norm, b07_norm, b8a_norm, b11_norm, b12_norm, 
            viewZen_norm, sunZen_norm, relAzim_norm, sum_dim=1):
    weights = torch.tensor([- 0.023406878966470, 
                            + 0.921655164636366,
                            + 0.135576544080099, 
                            - 1.938331472397950, 
                            - 3.342495816122680,
                            + 0.902277648009576,
                            + 0.205363538258614,
                            + 0.040607844721716,
                            + 0.083196409727092,
                            + 0.260029270773809,
                            + 0.284761567218845])
    weights = torch_select_unsqueeze(weights, sum_dim, len(b03_norm.size()))
    bias = torch.tensor(4.96238030555279)
    x = torch.cat((b03_norm, b04_norm, b05_norm, b06_norm,b07_norm, b8a_norm, b11_norm, b12_norm,
                   viewZen_norm,sunZen_norm,relAzim_norm), axis=sum_dim)
    sum =   bias + (weights * x).sum(sum_dim).unsqueeze(sum_dim)

    return tansig(sum)


def neuron2(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm, sum_dim=1):
    bias = torch.tensor(1.416008443981500)
    weights = torch.tensor([- 0.132555480856684,
                            - 0.139574837333540,
                            - 1.014606016898920,
                            - 1.330890038649270,
                            + 0.031730624503341,
                            - 1.433583541317050,
                            - 0.959637898574699,
                            + 1.133115706551000,
                            + 0.216603876541632,
                            + 0.410652303762839,
                            + 0.064760155543506])
    
    x = torch.cat((b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,
                   b11_norm,b12_norm,viewZen_norm,sunZen_norm,relAzim_norm), axis=sum_dim)
    weights = torch_select_unsqueeze(weights, sum_dim, len(b03_norm.size()))
    sum = bias + (weights * x).sum(sum_dim).unsqueeze(sum_dim)
    

    return tansig(sum)


def neuron3(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm, sum_dim=1): 
    bias = torch.tensor(1.075897047213310)
    weights = torch.tensor([  0.086015977724868,
                            + 0.616648776881434,
                            + 0.678003876446556,
                            + 0.141102398644968,
                            - 0.096682206883546,
                            - 1.128832638862200,
                            + 0.302189102741375,
                            + 0.434494937299725,
                            - 0.021903699490589,
                            - 0.228492476802263,
                            - 0.039460537589826,]) 
    x = torch.cat((b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,
                   b11_norm,b12_norm,viewZen_norm,sunZen_norm,relAzim_norm), axis=sum_dim)
    weights = torch_select_unsqueeze(weights, sum_dim, len(b03_norm.size()))
    sum = bias + (weights * x).sum(sum_dim).unsqueeze(sum_dim)
    return tansig(sum)


def neuron4(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm, sum_dim=1):
    bias = torch.tensor(1.533988264655420)
    weights = torch.tensor([- 0.109366593670404,
                            - 0.071046262972729,
                            + 0.064582411478320,
                            + 2.906325236823160,
                            - 0.673873108979163,
                            - 3.838051868280840,
                            + 1.695979344531530,
                            + 0.046950296081713,
                            - 0.049709652688365,
                            + 0.021829545430994,
                            + 0.057483827104091])
    x = torch.cat((b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,
                   b11_norm,b12_norm,viewZen_norm,sunZen_norm,relAzim_norm), axis=sum_dim)
    weights = torch_select_unsqueeze(weights, sum_dim, len(b03_norm.size()))
    sum = bias + (weights * x).sum(sum_dim).unsqueeze(sum_dim)
    return tansig(sum)


def neuron5(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm, sum_dim=1): 
    bias = torch.tensor(3.024115930757230)
    weights = torch.tensor([- 0.089939416159969,
                            + 0.175395483106147,
                            - 0.081847329172620,
                            + 2.219895367487790,
                            + 1.713873975136850,
                            + 0.713069186099534,
                            + 0.138970813499201,
                            - 0.060771761518025,
                            + 0.124263341255473,
                            + 0.210086140404351,
                            - 0.183878138700341,])
    x = torch.cat((b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,
                   b12_norm,viewZen_norm,sunZen_norm,relAzim_norm), axis=sum_dim)
    weights = torch_select_unsqueeze(weights, sum_dim, len(b03_norm.size()))
    sum = bias + (weights * x).sum(sum_dim).unsqueeze(sum_dim)
    return tansig(sum)


def layer2(neuron1, neuron2, neuron3, neuron4, neuron5, sum_dim=1):
    bias = torch.tensor(1.096963107077220)
    weights = torch.tensor([- 1.500135489728730,
                            - 0.096283269121503,
                            - 0.194935930577094,
                            - 0.352305895755591,
                            + 0.075107415847473,])
    x = torch.cat((neuron1,neuron2,neuron3,neuron4,neuron5), axis=sum_dim)
    weights = torch_select_unsqueeze(weights, sum_dim, len(neuron1.size()))
    sum = bias + (weights * x).sum(sum_dim).unsqueeze(sum_dim)
    return sum


def normalize(unnormalized, min_sample, max_sample): 
    return 2 * (unnormalized - min_sample) / (max_sample - min_sample) - 1

def denormalize(normalized, min_sample, max_sample): 
    return 0.5 * (normalized + 1) * (max_sample - min_sample) + min_sample

def tansig(input): 
    return 2 / (1 + torch.exp(-2 * input)) - 1 

from torchutils.patches import patchify, unpatchify
def get_weiss_lai(image_tensor, patch_size=32, bands=torch.tensor([0,1,2,3,4,5,6,7,8,9])):
    patched_tensor = patchify(image_tensor, patch_size=patch_size, margin=0)
    patched_lai_image = torch.zeros((patched_tensor.size(0), patched_tensor.size(1), 1, patch_size, patch_size)).to('cpu')
    for i in range(patched_tensor.size(0)):
        for j in range(patched_tensor.size(1)):
            x = patched_tensor[i,j, bands, :, :]
            angles = torch.zeros(3, patch_size, patch_size)
            angles[0,...] = patched_tensor[i, j, 11,...]
            angles[1,...] = patched_tensor[i, j, 13,...]
            angles[2,...] = patched_tensor[i, j, 12,...] - patched_tensor[i,j,14,...]
            with torch.no_grad():
                lai = weiss_lai(x, angles, band_dim=0)
            patched_lai_image[i,j,...] = lai
    lai_image = unpatchify(patched_lai_image)[:,:image_tensor.size(1),:image_tensor.size(2)]
    return lai_image

def compare_images():
    import matplotlib.pyplot as plt
    path_to_image = r"/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/validation_tiles/after_SENTINEL2B_20171127-105827-648_L2A_T31TCJ_C_V2-2_roi_0.pth"
    cropped_image = torch.load(path_to_image)[:,3:256-3,3:256-3]
    lai_image = get_weiss_lai(cropped_image, bands=torch.tensor([0,1,2,4,5,6,3,7,8,9]))

    import rasterio
    with rasterio.open('/home/yoel/Téléchargements/2017-11-27-00 00_2017-11-27-23 59_Sentinel-2_L2A_Custom_script.tiff', 'r') as ds:
        np_arr = ds.read()
        arr = torch.from_numpy(np_arr.astype(float))
    fig, ax = plot_patches((lai_image,arr),title_list=['NN lai', "Sentinel-Hub lai"], colorbar=True)
    fig.savefig('validation_weiss_nn_shub.png')

def main():
    import prosailvae
    import os
    import matplotlib.pyplot as plt
    from dataset.juan_datapoints import get_interpolated_validation_data
    juan_data_dir_path = os.path.join(prosailvae.__path__[0], os.pardir) + "/field_data/processed/"
    lai_min=0
    dt_max=10
    sites = ["france", "spain1", "italy1", "italy2"]
    device='cpu'

    weiss_mode=True
    list_lai_preds = []
    dt_list = []
    ndvi_list = []
    for site in sites:
        s2_r, s2_a, lais, dt = get_interpolated_validation_data(site, juan_data_dir_path, lai_min=lai_min, 
                                                                dt_max=dt_max, method="closest")
        b4 = s2_r[:,2]
        b8 = s2_r[:,6]
        ndvi = (b8-b4)/(b8+b4+1e-6)
        bands_idx = {'B02':0,
                     'B03':1,
                     'B04':2,
                     'B05':3,
                     'B06':4,
                     'B07':5,
                     'B08':6,
                     'B8A':7,
                     'B11':8,
                     'B12':9}
        if weiss_mode:
            s2_r = s2_r[:, torch.tensor([1,2,3,4,5,7,8,9])]
            bands_idx = {'B03':0,
                        'B04':1,
                        'B05':2,
                        'B06':3,
                        'B07':4,
                        'B8A':5,
                        'B11':6,
                        'B12':7}
        lai_pred = weiss_lai(s2_r, s2_a, bands_idx=bands_idx)
        list_lai_preds.append(torch.cat((lai_pred, lais), axis=1))
        dt_list.append(dt)
        ndvi_list.append(ndvi)
    list_lai_preds = torch.cat(list_lai_preds,axis=0)
    fig, ax = plt.subplots()
    from sklearn.metrics import r2_score
    import numpy as np
    fig, ax = plt.subplots()
    ax.scatter(list_lai_preds[:,1], 3*list_lai_preds[:,0],s=1)
    m, b = np.polyfit(list_lai_preds[:,1].numpy(), 3*list_lai_preds[:,0].numpy(), 1)
    r2 = r2_score(list_lai_preds[:,1].numpy(), 3*list_lai_preds[:,0].numpy())
    mse = (list_lai_preds[:,1] - 3*list_lai_preds[:,0]).pow(2).mean().numpy()
    ax.set_xlabel("LAI")
    ax.set_ylabel("LAI Predicted Sentinel-Hub")
    ax.plot([0,8],[0,8],'k--',)
    ax.plot([0,8],
            [m * 0 + b, m * 8 + b],'r', label="{:.2f} x + {:.2f}\n r2 = {:.2f}\n MSE: {:.2f}".format(m,b,r2,mse))
    ax.legend()
    ax.set_aspect('equal')
    fig.savefig('juan_shub_validation_altered.png')
    fig, ax = plt.subplots()
    ax.scatter(list_lai_preds[:,1], list_lai_preds[:,0],s=1)
    m, b = np.polyfit(list_lai_preds[:,1].numpy(), list_lai_preds[:,0].numpy(), 1)
    r2 = r2_score(list_lai_preds[:,1].numpy(), list_lai_preds[:,0].numpy())
    mse = (list_lai_preds[:,1] - list_lai_preds[:,0]).pow(2).mean().numpy()
    ax.set_xlabel("LAI")
    ax.set_ylabel("LAI Predicted Sentinel-Hub")
    ax.plot([0,8],[0,8],'k--',)
    ax.plot([0,8],
            [m * 0 + b, m * 8 + b],'r', label="{:.2f} x + {:.2f}\n r2 = {:.2f}\n MSE: {:.2f}".format(m,b,r2,mse))
    ax.legend()
    ax.set_aspect('equal')
    fig.savefig('juan_shub_validation.png')
    
    return

if __name__ == "__main__":
    from prosail_plots import plot_patches
    compare_images()
    main()
