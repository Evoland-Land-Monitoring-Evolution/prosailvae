import os
import prosailvae
from prosailvae.ProsailSimus import PROSAILVARS, BANDS
from prosailvae.utils import load_dict, save_dict
from prosailvae.prosail_vae import load_PROSAIL_VAE_with_supervised_kl
from dataset.preprocess_small_validation_file import get_small_validation_data, LAI_columns
from metrics.prosail_plots import plot_lai_preds
from dataset.loaders import  get_simloader
import torch
import logging
import matplotlib.pyplot as plt
import numpy as np
from dataset.generate_dataset import simulate_prosail_samples_close_to_ref, simulate_lai_with_rec_error_hist, simulate_lai_with_rec_error_hist_with_enveloppe
import socket
from prosailvae.ProsailSimus import ProsailSimulator, SensorSimulator

from matplotlib.colors import LogNorm

LOGGER_NAME = "PROSAIL-VAE validation"

def get_model(model_dir):
    logging.basicConfig(filename=model_dir+'/pv_validation.log', 
                              level=logging.INFO, force=True)
    logger_name = LOGGER_NAME
    # create logger
    logger = logging.getLogger(logger_name)
    params = load_dict(model_dir + "/config.json")
    if params["supervised"]:
        params["simulated_dataset"]=True
    params_sup_kl_model = None
    if params["supervised_kl"]:
        params_sup_kl_model = load_dict(model_dir+"/sup_kl_model_config.json")
        params_sup_kl_model['sup_model_weights_path'] = model_dir+"/sup_kl_model_weights.tar"
    vae_file_path = model_dir + '/prosailvae_weights.tar'
    rsr_dir = '/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/'
    PROSAIL_VAE = load_PROSAIL_VAE_with_supervised_kl(params, rsr_dir, model_dir, 
                                logger_name=LOGGER_NAME, vae_file_path=vae_file_path, params_sup_kl_model=params_sup_kl_model)
    return PROSAIL_VAE

def compare_reflectances(s2_r, s2_r_sim, site):
    fig, ax = plt.subplots(dpi=200)
    dis = 0.3
    bp_real = ax.boxplot(s2_r.transpose(0,1),positions=torch.arange(0,20,2)-dis, widths=0.2,patch_artist=True, showfliers=False)
    bp_sim = ax.boxplot(s2_r_sim.transpose(0,1), positions=torch.arange(0,20,2)+dis, widths=0.2,patch_artist=True, showfliers=False)
   
    for patch in bp_real['boxes']:
        patch.set_facecolor("green")

    for patch in bp_sim['boxes']:
        patch.set_facecolor("red")
    ax.set_xticks(torch.arange(0,20,2))
    ax.set_xticklabels(BANDS)
    ax.set_ylabel("Reflectance")
    ax.legend([bp_real["boxes"][0], bp_sim["boxes"][0]], ['Validation', 'Simulated'], loc='upper left')
    ax.set_title(f'Reflectance boxplots for {site} site')
    return fig, ax

def compare_lai(lais, lai_sim, site):
    fig, ax = plt.subplots(dpi=200)
    dis = 0.3
    bp_real = ax.boxplot(lais.transpose(0,1),positions=[-dis], widths=0.2,patch_artist=True, showfliers=False)
    bp_sim = ax.boxplot(lai_sim.transpose(0,1), positions=[dis], widths=0.2,patch_artist=True, showfliers=False)
   
    for patch in bp_real['boxes']:
        patch.set_facecolor("green")

    for patch in bp_sim['boxes']:
        patch.set_facecolor("red")
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_ylabel("LAI")
    ax.legend([bp_real["boxes"][0], bp_sim["boxes"][0]], ['Validation', 'Simulated'], loc='upper left')
    ax.set_title(f'LAI boxplots for {site} site')
    return fig, ax

def lut_pred(s2_r, s2_r_sim, lai_sim):
    preds = torch.zeros((s2_r.size(0)))
    for i in range(s2_r.size(0)):
        s2_r_rmae = ((s2_r_sim - s2_r[i,:].unsqueeze(0))/s2_r_sim.max(0)[0].unsqueeze(0)).abs().mean(1)
        idx_closest = s2_r_rmae.argmin()
        preds[i] = lai_sim[idx_closest,:]
    return preds

def lut_advanced_pred(s2_r, s2_r_sim, lai_sim, s2_a, s2_a_sim):
    preds = torch.zeros((s2_r.size(0)))
    for i in range(s2_r.size(0)):
        s2_r_rmae = ((s2_r_sim - s2_r[i,:].unsqueeze(0))/s2_r_sim.max(0)[0].unsqueeze(0)).abs().mean(1)
        min_mae = s2_r_rmae.min()
        close_refl_idx = torch.where(s2_r_rmae < 1.2 * min_mae)[0]
        close_refl_lai_sim = lai_sim[close_refl_idx,:]
        close_refl_s2_a_sim = s2_a_sim[close_refl_idx,:]
        s2_a_rmae = ((close_refl_s2_a_sim - s2_a[i,:].unsqueeze(0))/s2_a_sim.max(0)[0].unsqueeze(0)).abs().mean(1)
        idx_closest_angle = torch.argmin(s2_a_rmae)
        closest_lai = close_refl_lai_sim[idx_closest_angle,:]
        preds[i] = closest_lai
    return preds



def compare_datasets():
    model_dir = "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/results/best_regression_2/"
    data_dir = "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/"
    results_dir = "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/results/validation/"
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    loader = get_simloader(file_prefix="full_", data_dir=data_dir)
    PROSAIL_VAE = get_model(model_dir)
    for site in [ "spain1", "spain2", "italy1", "italy2", "france", "spain1"]:
        print(site)
        relative_s2_time="both"
        s2_r, s2_a, lais, time_delta = get_small_validation_data(relative_s2_time=relative_s2_time, site=site, filter_if_available_positions=True)
        dist_params, z, sim, rec = PROSAIL_VAE.point_estimate_rec(s2_r, s2_a, mode='sim_mode')
        print(s2_r.size())
        sim_lai = sim[:,6,:]
        # for i in range(lais.size(1)):
        s2_r_sim = loader.dataset[:][0]
        s2_a_sim = loader.dataset[:][1]
        prosail_var_sim = loader.dataset[:][2]
        lai_sim = prosail_var_sim[:,6].unsqueeze(1)
        idx_valid_data = 0
        valid_angles = s2_a[idx_valid_data,:]
        valid_refl = s2_r[idx_valid_data,:]

        fig, ax = compare_reflectances(s2_r, s2_r_sim, site)
        fig.savefig(results_dir+f'/{site}_refl_distribution.svg')
        if site != "franc":
            fig, ax = compare_lai(lais, lai_sim, site)
            fig.savefig(results_dir+f'/{site}_LAI_distribution.svg')
            lut_lai = lut_pred(s2_r, s2_r_sim, lai_sim)
            lut_lai_advanced = lut_advanced_pred(s2_r, s2_r_sim, lai_sim, s2_a, s2_a_sim)  
            fig, ax = plot_lai_preds(lais, lut_lai_advanced, time_delta, site)
            fig.savefig(results_dir+f'/{site}_LUT_w_angles_LAI_pred.svg')
            fig, ax = plot_lai_preds(lais, lut_lai, time_delta, site)
            fig.savefig(results_dir+f'/{site}_LUT_LAI_pred.svg')
            fig, ax = plot_lai_preds(lais, sim_lai, time_delta, site)
            fig.savefig(results_dir+f'/{site}_prosail_vae_LAI_pred.svg')
        plt.close('all')
    return

def plot_s2r_vs_s2_r_pred(s2_r, s2_r_pred, prosail_vars=None, angles=None, site="italy", lai_ref=None, best_mae=None, delta_t=None):
    fig, ax = plt.subplots(dpi=200,tight_layout=True)
    ax.scatter(torch.arange(0,20,2), s2_r, c="g", marker="o")
    ax.scatter(torch.arange(0,20,2), s2_r_pred, c="r" , marker="+")

    ax.set_xticks(torch.arange(0,20,2))
    ax.set_xticklabels(BANDS)
    ax.set_ylabel("Reflectance")
    ax.legend(['Validation', 'Simulated'], loc='upper left')
    fig.suptitle(f'Closest matching simulated reflectance for {site} site')
    if prosail_vars is not None:
        title_str = ''
        for i in range(len(PROSAILVARS)):
            title_str += ' {}={:.3f} |'.format(PROSAILVARS[i], prosail_vars[i])
            if PROSAILVARS[i] == "lai":
                if lai_ref is not None:
                    title_str += ' {}={:.3f} |'.format("LAI ref", lai_ref)
            if ((i+1) % 4) == 0:
                title_str += '\n'
        if angles is not None:
            title_str += "\ntts={:.1f} | tto={:.1f} | psi={:.1f}".format(angles[0], angles[1], angles[2])
        if best_mae is not None :
            title_str += "\nmae={:.4f} |".format(best_mae)
        if delta_t is not None :
            title_str += " dt={:.0f}".format(delta_t)
        ax.set_title(title_str)
    return fig, ax

def get_quantiles_from_hist(hist, bin_range=[0,1]):
    hist = hist / hist.sum(0)
    hist_range = np.linspace(bin_range[0], bin_range[1], hist.shape[0])
    quantiles = np.zeros((6, hist.shape[1]))
    cdf = np.cumsum(hist, 0)
    q1 = np.apply_along_axis(lambda a: hist_range[np.where(a < 0.25)[0][-1]], 0, cdf)
    median = np.apply_along_axis(lambda a: hist_range[np.where(a < 0.50)[0][-1]], 0, cdf)
    q3 = np.apply_along_axis(lambda a: hist_range[np.where(a < 0.75)[0][-1]], 0, cdf)
    # iqr = q3 - q1
    # k = 1.5
    min_q = np.apply_along_axis(lambda a: hist_range[np.where(a > 0)[0][0]], 0, cdf)
    max_q = np.apply_along_axis(lambda a: hist_range[np.where(a < 1.0)[0][-1]], 0, cdf)
    means = (hist * hist_range.reshape(-1,1)).sum(0)
    quantiles[0, :] = np.maximum(min_q, bin_range[0])
    quantiles[1, :] = q1
    quantiles[2, :] = median
    quantiles[3, :] = q3
    quantiles[4, :] = np.minimum(max_q, bin_range[1])
    quantiles[5, :] = means
    return quantiles


def h_customized_box_plot(percentiles, axes, redraw = True, *args, **kwargs):
    """
    Generates a customized boxplot based on the given percentile values
    """
    if len(percentiles.shape)==1:
        n_box = 1
        percentiles = percentiles.reshape(-1,1)
    else:
        n_box = percentiles.shape[1]
    if percentiles.shape[0]==6:
        showmeans = True
        meanpointprops = dict(marker='D', markeredgecolor='black',
                      markerfacecolor='white')
        box_plot = axes.boxplot([[-9, -4, 2, 4, 9],]*n_box, vert=False, meanprops=meanpointprops, meanline=False, showmeans=showmeans, *args, **kwargs) 
    else:
        showmeans = False
        box_plot = axes.boxplot([[-9, -4, 2, 4, 9],]*n_box, vert=False, showmeans=showmeans, *args, **kwargs) 
    # Creates len(percentiles) no of box plots

    min_x, max_x = float('inf'), -float('inf')

    for box_no in range(n_box):
        pdata = percentiles[:,box_no]
        if len(pdata) == 6:
            (q1_start, q2_start, q3_start, q4_start, q4_end, means) = pdata
        elif len(pdata) == 5:
            (q1_start, q2_start, q3_start, q4_start, q4_end) = pdata
            mean = None
        else:
            raise ValueError("Percentile arrays for customized_box_plot must have either 5 or 6 values")

        # Lower cap
        box_plot['caps'][2*box_no].set_xdata([q1_start, q1_start])
        # xdata is determined by the width of the box plot

        # Lower whiskers
        box_plot['whiskers'][2*box_no].set_xdata([q1_start, q2_start])

        # Higher cap
        box_plot['caps'][2*box_no + 1].set_xdata([q4_end, q4_end])

        # Higher whiskers
        box_plot['whiskers'][2*box_no + 1].set_xdata([q4_start, q4_end])

        # Box
        path = box_plot['boxes'][box_no].get_path()
        path.vertices[0][0] = q2_start
        path.vertices[1][0] = q2_start
        path.vertices[2][0] = q4_start
        path.vertices[3][0] = q4_start
        path.vertices[4][0] = q2_start

        # Median
        box_plot['medians'][box_no].set_xdata([q3_start, q3_start])

        # Mean
        if means is not None:
            
            box_plot['means'][box_no].set_xdata([means])
        # else:
        min_x = min(q1_start, min_x)
        max_x = max(q4_end, max_x)

        # The y axis is rescaled to fit the new box plot completely with 10% 
        # of the maximum value at both ends
        axes.set_xlim([min_x*1.1, max_x*1.1])

    # If redraw is set to true, the canvas is updated.
    if redraw:
        axes.figure.canvas.draw()

    return box_plot

def plot_s2_sim_dist(s2_r_ref, aggregate_s2_hist, lai, tts, tto, psi, colors=None, lai_ref=None):
    quantiles = get_quantiles_from_hist(aggregate_s2_hist, bin_range=[0,1])
    fig, ax = plt.subplots(dpi=200)
    bplot = h_customized_box_plot(quantiles, ax, redraw = True)
    # if colors is None :
    #     cmap = plt.cm.get_cmap('rainbow')
    #     colors = [cmap(val/10) for val in range(10)]
    # for patch, color in zip(bplot['boxes'], colors):
    #     patch.set_facecolor(color)
    # for median in bplot['medians']:
    #         median.set(color='k', linewidth=2,)
    # axs[row, col].set_xticklabels(model_names)
    s = ax.scatter(s2_r_ref, np.arange(1,11))
    ax.xaxis.grid(True) 
    ax.set_yticklabels(BANDS)
    title = ""
    if lai_ref is not None:
        title = "lai ref={:.1f} | ".format(lai_ref)
    title += "lai={:.1f} | tts={:.1f} | tto={:.1f} | psi={:.1f}".format(lai, tts, tto, psi)
    ax.set_title(title)
    ax.set_xlabel("Reflectances")
    ax.legend([bplot["boxes"][0], s] , ["Simulation distribution", "S2 Bands"])
    return fig, ax

def find_close_simulation(relative_s2_time, site, rsr_dir, results_dir, samples_per_iter=1024, max_iter=100, n=3, lai_min=1.5, exclude_lai=False, max_delta=3):

    s2_r, s2_a, lais, time_delta = get_small_validation_data(relative_s2_time=relative_s2_time, site=site, filter_if_available_positions=True, lai_min=lai_min)
    abs_time_delta = time_delta.abs().numpy().reshape(-1,1)
    psimulator = ProsailSimulator()
    ssimulator = SensorSimulator(rsr_dir + "/sentinel2.rsr")
    (top_n_delta, top_n_s2_r, top_n_s2_a, 
     top_n_lais) = sort_by_smallest_deltas(abs_time_delta, s2_r.numpy(), s2_a.numpy(), lais.numpy(), n=n)
    for idx_in_situ_sample in range(len(top_n_delta)):
        delta_t = top_n_delta[idx_in_situ_sample]
        if delta_t < max_delta:
            lai_ref = top_n_lais[idx_in_situ_sample,0]
            s2_r_ref = top_n_s2_r[idx_in_situ_sample,:].reshape(1,-1)
            tts = top_n_s2_a[idx_in_situ_sample, 0]
            tto = top_n_s2_a[idx_in_situ_sample, 1]
            psi = top_n_s2_a[idx_in_situ_sample, 2]
            if exclude_lai:
                lai = None
            else:
                lai = lai_ref
            (best_prosail_vars, best_prosail_s2_sim, 
            n_drawn_samples, aggregate_s2_hist,
            best_mae) = simulate_prosail_samples_close_to_ref(s2_r_ref, noise=0, psimulator=psimulator, ssimulator=ssimulator, lai=lai, tts=tts, 
                                                tto=tto, psi=psi, eps_mae=1e-3, max_iter=max_iter, samples_per_iter=samples_per_iter)
            lai = best_prosail_vars[6]
            fig, ax = plot_s2r_vs_s2_r_pred(s2_r_ref, best_prosail_s2_sim, best_prosail_vars, 
                                            top_n_s2_a[idx_in_situ_sample, :],site=site, lai_ref=lai_ref, best_mae=best_mae, delta_t=delta_t)
            fig.savefig(results_dir+f'/{site}_{idx_in_situ_sample}_lai_ref_{not exclude_lai}_closest_reflectance_match.svg')
            fig, ax = plot_s2_sim_dist(s2_r_ref, aggregate_s2_hist, lai, tts, tto, psi, colors=None)
            fig.savefig(results_dir+f'/{site}_{idx_in_situ_sample}_lai_ref_{not exclude_lai}_simulation_distribution.svg')
    pass

def plot_lai_vs_mc_error_2d_hist(heatmap, extent, aggregate_lai_hist, min_lai=0, max_lai=10, lai_ref=None,
                                 lai_pred=None, min_err=None, site='france', dt = 0, log_err=True,
                                 all_cases_in_enveloppe_err=[],all_cases_in_enveloppe_LAI=[]):
    fig, axs = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)}, dpi=200)
    axs[1].set_xlim([extent[0], extent[1]])
    if log_err:
        axs[1].set_yscale('log') 
        x = np.linspace(extent[0], extent[1], heatmap.shape[0]+1)
        y = np.logspace(np.log10(extent[2]), np.log10(extent[3]), heatmap.shape[1]+1)
        pc = axs[1].pcolor(x,y,heatmap.transpose(1,0),cmap='BrBG', norm=LogNorm())
        if min_err is not None:
            axs[1].set_ylim(max(min_err - 2e-3, 5e-3), None)
    else:
        axs[1].imshow(heatmap.transpose(1,0), extent=extent, interpolation='nearest',cmap='BrBG', origin='lower')
        axs[1].set_ylim([extent[2],extent[3]]) 
    axs[0].plot(np.linspace(min_lai, max_lai, len(aggregate_lai_hist)), aggregate_lai_hist.reshape(-1))
    axs[0].axvline([lai_ref],c="black", ls='--')
    # axs[0].axvline([lai_pred],c="red", ls='--')
    # axs[1].imshow(heatmap.transpose(1,0), extent=extent, interpolation='nearest',cmap='BrBG', origin='lower')
    # axs[1].imshow(heatmap.transpose(1,0), interpolation='nearest',cmap='BrBG', origin='lower')
    axs[1].axvline([lai_ref],c="black", ls='--', label='Reference LAI')
    # axs[1].axvline([lai_pred], c="red", ls='--', label='LAI for closest simulated reflectance')
    # axs[1].axhline([min_err],c="red", ls='-.', label='Error for closest simulated reflectance')
    if len(all_cases_in_enveloppe_err)>0:
        median_lai = np.median(all_cases_in_enveloppe_LAI)
        axs[1].axvline([median_lai], c="green", ls='--', label='median LAI for simulated reflectances within uncertainty')
        axs[0].axvline([median_lai], c="green", ls='--')
        axs[1].scatter(all_cases_in_enveloppe_LAI, all_cases_in_enveloppe_err, c='g', marker='+', s=1)
    axs[1].legend(fontsize=6)
    axs[1].set_aspect('auto')
    axs[1].set_xlabel('LAI')
    axs[1].set_ylabel('Mean Absolute Relative Error')
    axs[0].set_title(f'Error of simulated reflectances vs LAI (Site : {site} | dt : {dt})')
    axs[0].set_ylim(0,None)
    axs[0].set_ylabel('Distribution')
    return fig, axs

def mc_simulation_error(relative_s2_time, site, rsr_dir, results_dir, samples_per_iter=1024, max_iter=100, n=3, lai_min=1.5, max_delta=3, 
                        uniform_mode=True, lai_corr=False, lai_conv_override=None):
    s2_r, s2_a, lais, time_delta = get_small_validation_data(relative_s2_time=relative_s2_time, site=site, filter_if_available_positions=True, lai_min=lai_min)
    if len(time_delta)==0:
        print("No data with specified requirement was available.")
        return 
    abs_time_delta = time_delta.abs().numpy().reshape(-1,1)
    psimulator = ProsailSimulator()
    ssimulator = SensorSimulator(rsr_dir + "/sentinel2.rsr")
    (top_n_delta, top_n_s2_r, top_n_s2_a, 
     top_n_lais) = sort_by_smallest_deltas(abs_time_delta, s2_r.numpy(), s2_a.numpy(), lais.numpy(), n=n)
    lai_preds = []
    lai_refs = []
    for idx_in_situ_sample in range(len(top_n_delta)):
        delta_t = top_n_delta[idx_in_situ_sample]
        if delta_t <= max_delta:
            lai_ref = top_n_lais[idx_in_situ_sample,0]
            s2_r_ref = top_n_s2_r[idx_in_situ_sample,:].reshape(1,-1)
            lai_refs.append(lai_ref)
            tts = top_n_s2_a[idx_in_situ_sample, 0]
            tto = top_n_s2_a[idx_in_situ_sample, 1]
            psi = top_n_s2_a[idx_in_situ_sample, 2]
            (best_prosail_vars, best_prosail_s2_sim, 
             heatmap, extent, aggregate_lai_hist, 
             best_mae,  
             all_cases_in_enveloppe_err, all_cases_in_enveloppe_LAI) = simulate_lai_with_rec_error_hist_with_enveloppe(s2_r_ref, noise=0, psimulator=psimulator, ssimulator=ssimulator, tts=tts, 
                                                tto=tto, psi=psi, max_iter=max_iter, samples_per_iter=samples_per_iter, lai_conv_override=lai_conv_override,
                                                weiss_mode=site=="weiss", uniform_mode=uniform_mode, lai_corr=lai_corr,
                                                )
            lai_pred = best_prosail_vars[6]
            lai_preds.append(lai_pred)
            fig, ax = plot_lai_vs_mc_error_2d_hist(heatmap, extent, aggregate_lai_hist, min_lai=0, max_lai=10, lai_ref=lai_ref,
                                                    lai_pred=lai_pred, min_err=best_mae, site=site, dt=delta_t,
                                                    all_cases_in_enveloppe_err=all_cases_in_enveloppe_err,
                                                    all_cases_in_enveloppe_LAI=all_cases_in_enveloppe_LAI)
            fig.savefig(results_dir+f'/{site}_{idx_in_situ_sample}_uni_{uniform_mode}_lai_vs_mc_sim_error_2d_hist.svg')
    

    pass

def sort_by_smallest_deltas(abs_time_delta, s2_r, s2_a, lais, n=5):
    smallest_time_delta = np.nanmin(abs_time_delta,1)
    ind_n_best = np.argpartition(smallest_time_delta, n)[:n]
    sorted_n_best = ind_n_best[np.argsort(smallest_time_delta[ind_n_best])]
    top_n_delta = smallest_time_delta[sorted_n_best]
    top_n_s2_r = s2_r[sorted_n_best, :]
    top_n_s2_a = s2_a[sorted_n_best, :]
    top_n_lais = lais[sorted_n_best, :]
    return top_n_delta, top_n_s2_r, top_n_s2_a, top_n_lais

def main():
    if socket.gethostname()=='CELL200973':
        relative_s2_time="both"
        site='france'
        rsr_dir = '/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/'
        results_dir = "/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/results/validation/"
        exclude_lai=False
        max_delta = 4
        # mc_simulation_error(relative_s2_time, site, rsr_dir, results_dir, samples_per_iter=1024, max_iter=2, n=3, 
        #                     lai_min=2.5, max_delta=3,uniform_mode=True, lai_corr=True, lai_conv_override=1000)
        # find_close_simulation(relative_s2_time, site, rsr_dir, results_dir, 
        #                       samples_per_iter=1024, max_iter=2, n=2, exclude_lai=exclude_lai, max_delta=max_delta)
        compare_datasets()                        
    else:
        rsr_dir = '/work/scratch/zerahy/prosailvae/data/'
        results_dir = "/work/scratch/zerahy/prosailvae/results/prosail_mc/"
        relative_s2_time="both"
        exclude_lai=False
        max_delta = 3
        site="weiss"
        mc_simulation_error(relative_s2_time, site, rsr_dir, results_dir, samples_per_iter=1024, max_iter=300, n=10, 
                            lai_min=3, max_delta=3, uniform_mode=True, lai_corr=True, lai_conv_override=1000)
        # for site in ["france", "italy1", "italy2"]:
        #     find_close_simulation(relative_s2_time, site, rsr_dir, results_dir, 
        #                           samples_per_iter=1024, max_iter=300, n=20, exclude_lai=exclude_lai, max_delta=max_delta)
    pass

if __name__ == "__main__":
    main()