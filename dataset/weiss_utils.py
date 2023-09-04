import os
# import torch
import numpy as np
import pandas as pd
# from torchutils.patches import patchify, unpatchify
# from snap_regression.snap_nn import SnapNN

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

