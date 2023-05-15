import os
import torch
import numpy as np
import pandas as pd
from torchutils.patches import patchify, unpatchify
from snap_regression.snap_nn import SnapNN

def load_refl_angles(path_to_data_dir):
    path_to_file = path_to_data_dir + "/InputNoNoise_2.csv"
    assert os.path.isfile(path_to_file)
    df_validation_data = pd.read_csv(path_to_file, sep=" ", engine="python")
    n_obs = len(df_validation_data)
    s2_r = df_validation_data[['B3', 'B4', 'B5', 'B6', 'B7', 'B8A', 'B11', 'B12']].values
   
    tts = np.rad2deg(np.arccos(df_validation_data['cos(thetas)'].values))
    tto = np.rad2deg(np.arccos(df_validation_data['cos(thetav)'].values))
    psi = np.rad2deg(np.arccos(df_validation_data['cos(phiv-phis)'].values))
    # lais = torch.as_tensor(df_validation_data['lai_true'].values.reshape(-1,1))
    # lai_bv_net = torch.as_tensor(df_validation_data['lai_bvnet'].values.reshape(-1,1))
    # time_delta = torch.zeros((n_obs,1))
    return s2_r, tts, tto, psi

def load_prosail_params(path_to_data_dir):
    path_to_file = path_to_data_dir + "/Sentinel2_Laws.txt"
    prosail_params = np.loadtxt(path_to_file, skiprows=1)
    N = prosail_params[:,4].reshape(-1,1)
    cab = prosail_params[:,5].reshape(-1,1)
    car = prosail_params[:,6].reshape(-1,1)
    cbrown = prosail_params[:,9].reshape(-1,1)
    CwRel = prosail_params[:,8].reshape(-1,1)
    cm = prosail_params[:,7].reshape(-1,1)
    cw = cm / (1 - CwRel)
    lai = prosail_params[:,0].reshape(-1,1)
    lidfa = prosail_params[:,1].reshape(-1,1)
    hspot = prosail_params[:,3].reshape(-1,1)
    rsoil = prosail_params[:,10].reshape(-1,1)
    psoil = np.zeros_like(rsoil)
    return N, cab, car, cbrown, cw, cm, lai, lidfa, hspot, psoil, rsoil


def load_weiss_dataset(path_to_data_dir):
    s2_r, tts, tto, psi = load_refl_angles(path_to_data_dir)
    N, cab, car, cbrown, cw, cm, lai, lidfa, hspot, psoil, rsoil = load_prosail_params(path_to_data_dir)
    prosail_vars = np.zeros((N.shape[0], 14))
    prosail_vars[:,6] = lai.reshape(-1)
    prosail_vars[:,0] = N.reshape(-1)
    prosail_vars[:,1] = cab.reshape(-1) 
    prosail_vars[:,2] = car.reshape(-1) 
    prosail_vars[:,3] = cbrown.reshape(-1)
    prosail_vars[:,4] = cw.reshape(-1)
    prosail_vars[:,5] = cm.reshape(-1)
    prosail_vars[:,7] = lidfa.reshape(-1)
    prosail_vars[:,8] = hspot.reshape(-1)
    prosail_vars[:,9] = psoil.reshape(-1)
    prosail_vars[:,10] = rsoil.reshape(-1)
    prosail_vars[:,11] = tts.reshape(-1)
    prosail_vars[:,12] = tto.reshape(-1)
    prosail_vars[:,13] = psi.reshape(-1)
    return s2_r, prosail_vars


def get_weiss_biophyiscal_from_batch(batch, patch_size=32, sensor=None, ver=None):
    if ver is None:
        if sensor is None:
            ver = "2.1"
        elif sensor =="2A":
            ver = "3A"
        elif sensor == "2B":
            ver = "3B"
        else:
            raise ValueError
    elif ver not in ["2.1", "3A", "3B"]:
        raise ValueError
    weiss_bands = torch.tensor([1,2,3,4,5,7,8,9])
    s2_r, s2_a = batch
    patched_s2_r = patchify(s2_r.squeeze(), patch_size=patch_size, margin=0)
    patched_s2_a = patchify(s2_a.squeeze(), patch_size=patch_size, margin=0)
    patched_lai_image = torch.zeros((patched_s2_r.size(0), patched_s2_r.size(1), 1, patch_size, patch_size))
    patched_cab_image = torch.zeros((patched_s2_r.size(0), patched_s2_r.size(1), 1, patch_size, patch_size))
    patched_cw_image = torch.zeros((patched_s2_r.size(0), patched_s2_r.size(1), 1, patch_size, patch_size))
    for i in range(patched_s2_r.size(0)):
        for j in range(patched_s2_r.size(1)):
            x = patched_s2_r[i, j, weiss_bands, ...]
            angles = torch.cos(torch.deg2rad(patched_s2_a[i, j, ...]))
            s2_data = torch.cat((x, angles),0)
            with torch.no_grad():
                lai_snap = SnapNN(variable='lai', ver=ver)
                lai_snap.set_weiss_weights()
                lai = lai_snap.forward(s2_data, spatial_mode=True)
                cab_snap = SnapNN(variable='cab', ver=ver)
                cab_snap.set_weiss_weights()
                cab = torch.clip(cab_snap.forward(s2_data, spatial_mode=True), min=0) # torch.clip(cab_snap.forward(s2_data, spatial_mode=True), min=0) / torch.clip(lai, min=0.1)
                cw_snap = SnapNN(variable='cw', ver=ver)
                cw_snap.set_weiss_weights()
                cw = torch.clip(cw_snap.forward(s2_data, spatial_mode=True), min=0) # torch.clip(cw_snap.forward(s2_data, spatial_mode=True), min=0) / torch.clip(lai, min=0. 1)
                # lai = weiss_lai(x, angles, band_dim=0, ver=ver)
            patched_lai_image[i,j,...] = lai
            patched_cab_image[i,j,...] = cab
            patched_cw_image[i,j,...] = cw
    lai_image = unpatchify(patched_lai_image)[:,:s2_r.size(2),:s2_r.size(3)]
    cab_image = unpatchify(patched_cab_image)[:,:s2_r.size(2),:s2_r.size(3)]
    cw_image = unpatchify(patched_cw_image)[:,:s2_r.size(2),:s2_r.size(3)]
    return lai_image, cab_image, cw_image