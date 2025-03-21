import os

import numpy as np
import pandas as pd

from prosailvae_types import GROUP_BAND_BVNET, GROUP_P5_BV, S2A, S2R, ProsailBV


def load_refl_angles(path_to_data_dir):
    path_to_file = path_to_data_dir + "/InputNoNoise_2.csv"
    assert os.path.isfile(path_to_file)
    df_validation_data = pd.read_csv(path_to_file, sep=" ", engine="python")
    s2_r = df_validation_data[
        ["B3", "B4", "B5", "B6", "B7", "B8A", "B11", "B12"]
    ].values

    tts = np.rad2deg(np.arccos(df_validation_data["cos(thetas)"].values))
    tto = np.rad2deg(np.arccos(df_validation_data["cos(thetav)"].values))
    psi = np.rad2deg(np.arccos(df_validation_data["cos(phiv-phis)"].values))
    return s2_r, tts, tto, psi


def load_prosail_params(path_to_data_dir, psoil0=0.0):
    path_to_file = path_to_data_dir + "/Sentinel2_Laws.txt"
    prosail_params = np.loadtxt(path_to_file, skiprows=1)
    N = prosail_params[:, 4].reshape(-1, 1)
    cab = prosail_params[:, 5].reshape(-1, 1)
    car = prosail_params[:, 6].reshape(-1, 1)
    cbrown = prosail_params[:, 9].reshape(-1, 1)
    CwRel = prosail_params[:, 8].reshape(-1, 1)
    cm = prosail_params[:, 7].reshape(-1, 1)
    cw = cm / (1 - CwRel)
    lai = prosail_params[:, 0].reshape(-1, 1)
    lidfa = prosail_params[:, 1].reshape(-1, 1)
    hspot = prosail_params[:, 3].reshape(-1, 1)
    rsoil = prosail_params[:, 10].reshape(-1, 1)
    psoil = psoil0 * np.ones_like(rsoil)
    return N, cab, car, cbrown, cw, cm, lai, lidfa, hspot, psoil, rsoil


def load_bvnet_dataset(path_to_data_dir, mode="pvae", psoil0=0.0):
    s2_r, sun_zen, view_zen, rel_azi = load_refl_angles(path_to_data_dir)
    N, cab, car, cbrown, cw, cm, lai, lidfa, hspot, psoil, rsoil = load_prosail_params(
        path_to_data_dir, psoil0=psoil0
    )
    prosail_vars = np.zeros((N.shape[0], 14))
    prosail_vars[:, 6] = lai.reshape(-1)
    prosail_vars[:, 0] = N.reshape(-1)
    prosail_vars[:, 1] = cab.reshape(-1)
    prosail_vars[:, 2] = car.reshape(-1)
    prosail_vars[:, 3] = cbrown.reshape(-1)
    prosail_vars[:, 4] = cw.reshape(-1)
    prosail_vars[:, 5] = cm.reshape(-1)
    prosail_vars[:, 7] = lidfa.reshape(-1)
    prosail_vars[:, 8] = hspot.reshape(-1)
    prosail_vars[:, 9] = psoil.reshape(-1)
    prosail_vars[:, 10] = rsoil.reshape(-1)
    if mode == "pvae":
        prosail_vars[:, 11] = sun_zen.reshape(-1)
        prosail_vars[:, 12] = view_zen.reshape(-1)
        prosail_vars[:, 13] = rel_azi.reshape(-1)
    elif mode == "bvnet":
        prosail_vars[:, 11] = view_zen.reshape(-1)
        prosail_vars[:, 12] = sun_zen.reshape(-1)
        prosail_vars[:, 13] = rel_azi.reshape(-1)
    else:
        raise ValueError
    return s2_r, prosail_vars
