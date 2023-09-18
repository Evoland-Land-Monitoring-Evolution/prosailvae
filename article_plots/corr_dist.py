
import matplotlib.pyplot as plt
import numpy as np
from dataset.generate_dataset import correlate_with_lai, correlate_with_lai_V2, Vmin, Vmax


def plot_lai_correlation(): 
    lai = np.arange(0, 15.5, 0.5)
    V = np.arange(0, 1.1, 0.1)
    V_grid, lai_grid = np.meshgrid(V, lai)
    lai_min = 0
    lai_max = 15
    Vmin_0 = 0
    Vmax_0 = 1
    V_mean = 0.6
    lai_conv=10
    lai_thresh = 10
    Vmax_lai_max = 0.7
    Vmin_lai_max = 0.5
    lai_true_max = 5
    # V_corr_v1 = np.zeros((len(V), len(lai)))
    # V_corr_v2 = np.zeros((len(V), len(lai)))
    # lai_grid = np.zeros((len(V), len(lai)))
    V_corr_v2 = correlate_with_lai_V2(lai_grid, V_grid, Vmin_0, Vmax_0, Vmin_lai_max, 
                                      Vmax_lai_max, lai_max, lai_thresh=lai_thresh)
    V_corr_v2_no_thresh = correlate_with_lai_V2(lai_grid, V_grid, Vmin_0, Vmax_0, Vmin_lai_max, 
                                      Vmax_lai_max, lai_max, lai_thresh=None)
    V_corr_v1 = correlate_with_lai(lai_grid, V_grid, V_mean, lai_conv)
    # for i, v in enumerate(V):
    #     for j, l in enumerate(lai):
    #         V_corr_v1[i,j] = correlate_with_lai(l, v, V_mean, lai_conv)
    #         V_corr_v2[i,j] = correlate_with_lai_V2(l, v, Vmin_0, Vmax_0, Vmin_lai_max, Vmax_lai_max, lai_max)
    #         lai_grid[i,j] = l

    fig, ax = plt.subplots()

    ax.plot([lai_min, lai_max], Vmin(np.array([lai_min, lai_max]), 
                                     Vmin_0=Vmin_0, 
                                     Vmin_lai_max=Vmin_lai_max, 
                                     lai_max=lai_max), 
                                     c="r")
    
    ax.plot([lai_min, lai_max], Vmax(np.array([lai_min, lai_max]), 
                                     Vmax_0=Vmax_0, 
                                     Vmax_lai_max=Vmax_lai_max,
                                     lai_max=lai_max), 
                                     c="r")
    V_min_lai_thresh = Vmin(np.array([lai_thresh]), 
                                     Vmin_0=Vmin_0, 
                                     Vmin_lai_max=Vmin_lai_max, 
                                     lai_max=lai_max)[0]
    ax.set_xlabel("LAI")
    ax.set_ylabel('V*')
    ax.set_xticks([lai_min, lai_thresh, lai_max], ["0", "$LAI_{thresh}$", "$LAI_{max}$"])
    ax.set_yticks([Vmin_0, V_min_lai_thresh, Vmin_lai_max, Vmax_lai_max, Vmax_0], 
                  ["$V_{l,0}$", "$V_{l,T}$", "$V_{l,M}$", "$V_{u,M}$", "$V_{u,0}$"])
    ax.axhline(Vmin_lai_max, c='k', ls='--')
    ax.axhline(Vmax_lai_max, c='k', ls='--')
    ax.axhline(V_min_lai_thresh, c='k', ls='--')
    ax.axvline(lai_thresh, c='k', ls='--')
    ax.axvline(lai_max, c='k', ls='--')
    ax.scatter(lai_grid, V_corr_v2, zorder=1000)

    fig, ax = plt.subplots()
    ax.plot([lai_min, lai_max], Vmin(np.array([lai_min, lai_max]), 
                                     Vmin_0=Vmin_0, 
                                     Vmin_lai_max=Vmin_lai_max, 
                                     lai_max=lai_max), 
                                     c="r")
    
    ax.plot([lai_min, lai_max], Vmax(np.array([lai_min, lai_max]), 
                                     Vmax_0=Vmax_0, 
                                     Vmax_lai_max=Vmax_lai_max,
                                     lai_max=lai_max), 
                                     c="r")
    V_min_lai_thresh = Vmin(np.array([lai_thresh]), 
                                     Vmin_0=Vmin_0, 
                                     Vmin_lai_max=Vmin_lai_max, 
                                     lai_max=lai_max)[0]
    ax.set_xlabel("LAI")
    ax.set_ylabel('V*')
    ax.set_xticks([lai_min, lai_max], ["0", "$LAI_{max}$"])
    ax.set_yticks([Vmin_0, Vmin_lai_max, Vmax_lai_max, Vmax_0], 
                  ["$V_{l,0}$", "$V_{l,M}$", "$V_{u,M}$", "$V_{u,0}$"])
    ax.axhline(Vmin_lai_max, c='k', ls='--')
    ax.axhline(Vmax_lai_max, c='k', ls='--')
    ax.axvline(lai_thresh, c='k', ls='--')
    ax.axvline(lai_max, c='k', ls='--')
    ax.scatter(lai_grid, V_corr_v2_no_thresh, zorder=1000)

    fig, ax = plt.subplots()
    ax.scatter(lai_grid, V_corr_v1)
    ax.set_xlabel("LAI")
    ax.set_ylabel('V*')
    ax.set_xticks([lai_min, lai_true_max, lai_conv, lai_max], ["0", "$LAI_{True max}$", "$C_{LAI}$", "$LAI_{max}$"])
    ax.set_yticks([Vmin_0, V_mean, Vmax_0], 
                  ["$V_{l,0}$", "$\mu_V$", "$V_{u,0}$"])
    
    ax.axvline(lai_thresh, c='k', ls='--')
    ax.axhline(V_mean, c='k', ls='--' )
    ax.axvline(lai_true_max, c='r', ls='--' )
    return

def main():
    plot_lai_correlation()

if __name__=="__main__":
    main()