import pandas as pd
import numpy as np
import torch
import prosailvae
import matplotlib.pyplot as plt
import os 
from prosailvae.ProsailSimus import ProsailSimulator, SensorSimulator, PROSAILVARS
from tqdm import trange
from metrics.prosail_plots import plot_param_compare_dist

PATH_TO_DATA_DIR = os.path.join(prosailvae.__path__[0], os.pardir) + "/field_data/lai/"

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

def plot_bands_scatter_plots(s2_r, s2_r_sim, vars_names = ["B3", "B4", "B5", "B6", "B7", "B8A", "B11", "B12"]):
    fig, axs = plt.subplots(2,4,dpi=150, tight_layout=True, figsize=(12,6))
    for i in range(8):
        row = i % 2
        col = i // 2
        axs[row, col].set_xlabel(f"{vars_names[i]} pred")
        axs[row, col].set_ylabel(f"{vars_names[i]} true")
        axs[row, col].scatter(s2_r_sim[:,i], s2_r[:,i], s=2)
        axs[row, col].plot([0,1], [0,1], 'k--')
        axs[row, col].set_aspect('equal', 'box')
    return fig, axs

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

def plot_lai_correlations(prosail_var_weiss,i):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.scatter(prosail_var_weiss[:,i], prosail_var_weiss[:,6], s=1)

def compare_weiss_w_simulations(prosail_var_weiss, prosail_var_simu):
    fig, ax = plot_param_compare_dist(prosail_var_simu, prosail_var_weiss, params_name=PROSAILVARS+["phi_s", "phi_o", "psi"], res_dir = None,)
    fig.savefig("/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/results/validation/sim_vs_weiss_vers.png")
    return 

def lognormal_pdf(x, mu, sigma):
    return np.exp(-0.5*(np.log(x) - mu)**2/sigma**2 )/(x * sigma * np.sqrt(2*np.pi))

def normal_pdf(x, mu, sigma):
    return np.exp(-0.5*(x - mu)**2/sigma**2 )/(sigma * np.sqrt(2*np.pi))


def main():
    s2_r, prosail_vars = load_weiss_dataset(PATH_TO_DATA_DIR)
    lai = prosail_vars[:,6]
    s2_r = torch.as_tensor(s2_r)
    data_dir = "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/"
    t_prosail_vars = torch.load(data_dir + "test_dist_prosail_sim_vars.pt")
    # t_s2_r = torch.load(data_dir + "test_dist_prosail_s2_sim_refl.pt")
    compare_weiss_w_simulations(torch.as_tensor(prosail_vars), t_prosail_vars)

    rsr_dir = '/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/'
    results_dir = "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/results/validation/"
    psimulator = ProsailSimulator()
    ssimulator = SensorSimulator(rsr_dir + "/sentinel2.rsr", bands=[2, 3, 4, 5, 6, 8, 11, 12])
    prosail_vars_lai4 = prosail_vars[(lai>0).reshape(-1),:]
    s2_r_lai4 = s2_r[(lai>0).reshape(-1),:]
    s2_r_sim = np.zeros_like(s2_r_lai4)
    for i in trange(prosail_vars_lai4.shape[0]):
        s2_r_sim[i,:] = ssimulator(psimulator(torch.from_numpy(prosail_vars_lai4[i,:]).view(1,-1).float())).numpy()
        # plt.figure()
        # plt.scatter(np.arange(8),s2_r_sim[0,:])
        # plt.scatter(np.arange(8),s2_r_lai4[0,:])
        # plt.show()
        # plt.close('all')

    fig, axs = plot_bands_scatter_plots(s2_r_lai4, s2_r_sim, vars_names = ["B3", "B4", "B5", "B6", "B7", "B8A", "B11", "B12"])
    fig.savefig(results_dir + "/weiss_refl_scatter_all_lai.png")
    return

if __name__=="__main__":
    main()