#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 11:46:20 2022

@author: yoel
"""

import matplotlib.pyplot as plt

# plt.rcParams.update({
#   "text.usetex": True,
#   "font.family": "Helvetica"
# })

import numpy as np
import pandas as pd
import torch
from prosailvae.ProsailSimus import PROSAILVARS, ProsailVarsDist, BANDS



def plot_metrics(save_dir, alpha_pi, maer, mpiwr, picp, mare):
    fig = plt.figure(dpi=200)
    
    for i in range(len(maer.columns)):
        plt.plot(alpha_pi, picp[i,:].detach().cpu().numpy(), label=maer.columns[i])
    plt.legend()
    plt.xlabel('1-a')
    plt.ylabel('Prediction Interval Coverage Probability')
    fig.tight_layout()
    fig.savefig(save_dir+"/picp.svg")
    
    fig = plt.figure(dpi=200)
    for i in range(len(maer.columns)):
        plt.plot(alpha_pi, mpiwr.values[:,i], label=maer.columns[i])
    plt.legend()
    plt.xlabel('1-a')
    plt.ylabel('Mean Prediction Interval Width (Standardized)')
    fig.tight_layout()
    fig.savefig(save_dir+"/MPIWr.svg")
    
    fig = plt.figure()
    plt.grid(which='both', axis='y')
    ax = fig.add_axes([0,0,1,1])
    ax.bar(maer.columns, maer.values.reshape(-1),)
    plt.ylabel('Mean Absolute Error (Standardized)')
    ax.yaxis.grid(True)
    fig.tight_layout()
    fig.savefig(save_dir+"/MAEr.svg")
    
    fig = plt.figure(dpi=150)
    plt.grid(which='both', axis='y')
    ax = fig.add_axes([0,0,1,1])
    ax.bar(maer.columns, mare.detach().cpu().numpy())
    plt.ylabel('Mean Absolute Relative Error')
    plt.yscale('log')
    ax.yaxis.grid(True)
    fig.tight_layout()
    fig.savefig(save_dir+"/mare.svg")

def plot_rec_hist2D(prosail_VAE, loader, res_dir, nbin=50, bands_name=None):
    if bands_name is None:
        bands_name = BANDS
    original_prosail_s2_norm = prosail_VAE.decoder.ssimulator.apply_norm
    prosail_VAE.decoder.ssimulator.apply_norm = False
    recs_dist = torch.tensor([]).to(prosail_VAE.device)
    s2_r_dist = torch.tensor([]).to(prosail_VAE.device)
    with torch.no_grad():
        for i, batch in enumerate(loader):
            s2_r = batch[0].to(prosail_VAE.device)
            s2_r_dist = torch.concat([s2_r_dist, s2_r], axis=0)
            angles = batch[1].to(prosail_VAE.device)
            len_batch=s2_r.size(0)
            for j in range(len_batch):
                _, _, _, recs = prosail_VAE.forward(s2_r[j,:].unsqueeze(0), angles[j,:].unsqueeze(0), n_samples=100)
                recs_dist = torch.concat([recs_dist, recs], axis=0)
    n_bands = s2_r_dist.size(1)
    N = s2_r_dist.size(0)
    fig, axs = plt.subplots(2, n_bands//2 + n_bands % 2, dpi=120, tight_layout=True, figsize=(1 + 2*(n_bands//2 + n_bands%2), 1+2*2))
    for i in range(n_bands):
        axi = i%2
        axj = i//2

        xs = recs_dist[:,i,:].detach().cpu().numpy()
        xs_05 = np.quantile(xs, 0.05)
        xs_95 = np.quantile(xs, 0.95)
        ys = s2_r_dist[:,i].detach().cpu().numpy()
        ys_05 = np.quantile(ys, 0.05)
        ys_95 = np.quantile(ys, 0.95)
        min_b = min(xs_05, ys_05)
        max_b = max(xs_95, ys_95)
        xedges = np.linspace(min_b, max_b, nbin)
        yedges = np.linspace(min_b, max_b, nbin)
        heatmap = 0
        for j in range(N):
            xj = xs[j,:]
            yj = ys[j]
            hist, xedges, yedges = np.histogram2d(
                np.ones_like(xj) * yj, xj, bins=[xedges, yedges])
            heatmap += hist
        # heatmap = heatmap #np.flipud(np.rot90(heatmap))
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        
        axs[axi, axj].imshow(heatmap, extent=extent, interpolation='nearest',cmap='BrBG', origin='lower')
        axs[axi, axj].set_ylabel(bands_name[i])
        axs[axi, axj].set_xlabel("rec. " + bands_name[i])
        axs[axi, axj].plot([min_b, max_b], [min_b, max_b], c='w')
    plt.show()
    fig.savefig(res_dir + '/2d_rec_dist.svg')
    plt.close('all')
    prosail_VAE.decoder.ssimulator.apply_norm = original_prosail_s2_norm
    pass

def plot_lat_hist2D(tgt_dist, sim_pdfs, sim_supports, res_dir, nbin=50):
    n_lats = sim_pdfs.size(1)
    N = sim_pdfs.size(0)
    fig_all, axs_all = plt.subplots(2, n_lats//2 + n_lats%2, dpi=120, tight_layout=True, figsize=(1 + 2*(n_lats//2 + n_lats%2), 1+2*2))
    for i in range(n_lats):
        heatmap, extent = compute_dist_heatmap(tgt_dist[:,i], sim_pdfs[:,i,:], sim_supports[:,i,:], nbin=nbin)
        # heatmap = np.flipud(np.rot90(heatmap))
        fig, ax = plot_single_lat_hist_2D(heatmap, extent, res_dir=None, fig=None, ax=None, var_name=PROSAILVARS[i])
        fig.savefig(res_dir + f'/2d_pred_dist_{PROSAILVARS[i]}.svg')
        fig_all, axs_all[i%2, i//2] = plot_single_lat_hist_2D(heatmap, extent, res_dir=None, fig=fig_all, 
                                                              ax=axs_all[i%2, i//2], var_name=PROSAILVARS[i])
    if n_lats%2==1:
        fig_all.delaxes(axs_all[-1, -1])
    fig_all.savefig(res_dir + f'/2d_pred_dist_PROSAIL_VARS.svg')
    plt.close('all')
    pass

def compute_dist_heatmap(tgt_dist, sim_pdf, sim_support, nbin=50):
    N = sim_pdf.size(0)
    xs = sim_support.detach().cpu().numpy()
    ys = tgt_dist.detach().cpu().numpy()
    min_b = np.quantile(ys,0.05)
    max_b = np.quantile(ys,0.95)
    weights = sim_pdfs.detach().cpu().numpy()
    xedges = np.linspace(min_b, max_b, nbin)
    yedges = np.linspace(min_b, max_b, nbin)
    heatmap = 0
    for j in range(N):
        xj = xs[j,:]
        yj = ys[j]
        wj = weights[j,:]
        hist, xedges, yedges = np.histogram2d(
            np.ones_like(xj) * yj, xj, bins=[xedges, yedges], weights=wj)
        heatmap += hist
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]    
    return heatmap, extent

def plot_single_lat_hist_2D(heatmap=None, extent=None, tgt_dist=None, sim_pdf=None, sim_support=None,
                            res_dir=None, fig=None, ax=None, var_name=None, nbin=50):
    if heatmap is None or extent is None:
        if tgt_dist is not None and sim_pdf is not None and sim_support is not None:
            heatmap, extent = compute_dist_heatmap(tgt_dist, sim_pdf, sim_support, nbin=50)
        else:
            raise ValueError("Please input either heatmap and extent, or tgt_dist, sim_pdf and sim_support")
    if fig is None or ax is None:
        fig, ax = plt.subplots(dpi=120)
    ax.imshow(heatmap, extent=extent, interpolation='nearest',cmap='BrBG', origin='lower')
    ax.plot([extent[0], extent[1]], [extent[0], extent[1]], c='w')
    if var_name is not None:
        ax.set_ylabel(f"{var_name}")
        ax.set_xlabel(f"Predicted distribution of {var_name}")
    if res_dir is not None and var_name is not None:
        fig.savefig(res_dir + f'/2d_pred_dist_{var_name}.svg')
    return fig, ax
    
def plot_rec_and_latent(prosail_VAE, loader, res_dir, n_plots=10, bands_name=None):
    if bands_name is None:
        bands_name = BANDS
    original_prosail_s2_norm = prosail_VAE.decoder.ssimulator.apply_norm
    prosail_VAE.decoder.ssimulator.apply_norm = False
    for i in range(n_plots):
        sample_refl = loader.dataset[i:i+1][0].to(prosail_VAE.device)
        sample_refl.requires_grad=False
        angle = loader.dataset[i:i+1][1].to(prosail_VAE.device)
        ref =  loader.dataset[i:i+1][2].to(prosail_VAE.device)
        angle.requires_grad=False
        dist_params,_,sim,rec = prosail_VAE.forward(sample_refl, angle, 
                                                    n_samples=1000)

        lat_pdfs, lat_supports = prosail_VAE.lat_space.latent_pdf(dist_params)
        sim_pdfs, sim_supports = prosail_VAE.sim_space.sim_pdf(lat_pdfs, lat_supports, 
                                                               n_pdf_sample_points=3001)
        #gridspec_kw={'height_ratios':[len(PROSAILVARS)]+[1 for i in range(len(PROSAILVARS))]}
        #len(PROSAILVARS)+1,1, 
        fig = plt.figure(figsize=(12,8), dpi=150,)
    
        gs = fig.add_gridspec(len(PROSAILVARS),2)
        ax1 = fig.add_subplot(gs[:, 0])
        ax2=[]
        for j in range(len(PROSAILVARS)):
            ax2.append(fig.add_subplot(gs[j, 1]))
        rec_samples = rec.squeeze().detach().cpu().numpy()
        
        
        
        rec_samples = [rec_samples[j,:] for j in range(len(bands_name))]
        sim_samples = sim.squeeze().detach().cpu().numpy()
        sim_samples = [sim_samples[j,:] for j in range(len(PROSAILVARS))]
        
        ind1 = np.arange(len(bands_name))
        ax1.set_xlim(0,1)
        v1 = ax1.violinplot(rec_samples, points=100, positions=ind1,
               showmeans=True, showextrema=True, showmedians=False, vert=False)
        for b in v1['bodies']:
            # get the center
            m = np.mean(b.get_paths()[0].vertices[:, 1])
            b.get_paths()[0].vertices[:, 1] = np.clip(b.get_paths()[0].vertices[:, 1], m, np.inf)
            b.set_color('r')
            b.set_facecolor('red')
            b.set_edgecolor('red')
        for partname in ('cbars','cmins','cmaxes','cmeans'):
            v = v1[partname]
            v.set_edgecolor('red')
            v.set_linewidth(1)
        # ax1.barh(ind1, sample_refl.squeeze().cpu().numpy(), width, align='center', 
        #        alpha=0.5, color='royalblue', capsize=10)
        ax1.scatter(sample_refl.squeeze().cpu().numpy(),
                    ind1-0.1, color='black',s=15)
    
        ax1.set_yticks(ind1)
        ax1.set_yticklabels(bands_name)
        ax1.xaxis.grid(True)

        for j in range(len(PROSAILVARS)):
            # v2 = ax2[j].violinplot(sim_samples[j], points=100, positions=[ind2[j]+width],
            #        showmeans=True, showextrema=True, showmedians=False, vert=False)
            min_b = ProsailVarsDist.Dists[PROSAILVARS[j]]["min"]
            max_b = ProsailVarsDist.Dists[PROSAILVARS[j]]["max"]
            dist_max = sim_pdfs.squeeze()[j,:].detach().cpu().max().numpy()
            dist_argmax =  sim_pdfs.squeeze()[j,:].detach().cpu().argmax().numpy()
            ax2[j].set_xlim(min_b, max_b)
            # ax2[j].scatter([min_b,max_b],[ind2[j]+width,
            #                               ind2[j]+width],color='k')
            ax2[j].plot(sim_supports.squeeze()[j,:].detach().cpu().numpy(),
                        sim_pdfs.squeeze()[j,:].detach().cpu().numpy(),color='red')
            ax2[j].fill_between(sim_supports.squeeze()[j,:].detach().cpu().numpy(), 
                                sim_pdfs.squeeze()[j,:].detach().cpu().numpy(), 
                                np.zeros_like(sim_pdfs.squeeze()[j,:].detach().cpu().numpy()), 
                                alpha=0.3,
                                facecolor=(1,0,0,.4))
            ax2[j].plot([sim_supports.squeeze()[j,dist_argmax].detach().cpu().numpy(),
                         sim_supports.squeeze()[j,dist_argmax].detach().cpu().numpy()],
                        [0,dist_max], color='red')
            # for b in v2['bodies']:
            #     # get the center
            #     m = np.mean(b.get_paths()[0].vertices[:, 1])
            #     b.get_paths()[0].vertices[:, 1] = np.clip(b.get_paths()[0].vertices[:, 1], m, np.inf)
            #     b.set_color('r')
            #     b.set_facecolor('red')
            #     b.set_edgecolor('red')
            # for partname in ('cbars','cmins','cmaxes','cmeans'):
            #     v = v2[partname]
            #     v.set_edgecolor('red')
            #     v.set_linewidth(1)
            ax2[j].scatter(ref.squeeze()[j].detach().cpu().numpy(), 
                           dist_max/2, s=15, color='black')
    
            ax2[j].set_yticks([0])
            ax2[j].set_yticklabels([])
            ax2[j].set_ylabel(PROSAILVARS[j])
            ax2[j].xaxis.grid(True)
            
            
        # Save the figure and show
        plt.tight_layout()
        plt.show()
        fig.savefig(res_dir + f'/reflectance_rec_{i}.svg')
        plt.close('all')
    prosail_VAE.decoder.ssimulator.apply_norm = original_prosail_s2_norm
    
def loss_curve(loss_df, save_file, log_scale=False):
    loss_names = loss_df.columns.values.tolist()
    loss_names.remove("epoch")
    epochs = loss_df["epoch"]
    fig, ax = plt.subplots(dpi=150)
    ax.set_yscale('symlog', linthresh=1e-5)
    for i in range(len(loss_names)):
        loss = loss_df[loss_names[i]].values
        # if log_scale: # (loss<=0).any() or 
        #     if (loss<=0).any():
        #         loss += loss.min() + 1
        #     ax.set_yscale('log')
        ax.plot(epochs,loss, label=loss_names[i])
    ax.legend()
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    fig.savefig(save_file)
    
def plot_param_dist(res_dir, sim_dist, tgt_dist):
    fig = plt.figure(figsize=(18,12), dpi=150,)
    ax2=[]
    gs = fig.add_gridspec(len(PROSAILVARS),1)
    for j in range(len(PROSAILVARS)):
        ax2.append(fig.add_subplot(gs[j, 0]))
    
    for j in range(len(PROSAILVARS)):
        v2 = ax2[j].violinplot(sim_dist[:,j].squeeze().detach().cpu(), points=100, positions=[0],
                showmeans=True, showextrema=True, showmedians=False, vert=False)
        min_b = ProsailVarsDist.Dists[PROSAILVARS[j]]["min"]
        max_b = ProsailVarsDist.Dists[PROSAILVARS[j]]["max"]
        
        ax2[j].set_xlim(min_b, max_b)

        
        for b in v2['bodies']:
            # get the center
            m = np.mean(b.get_paths()[0].vertices[:, 1])
            b.get_paths()[0].vertices[:, 1] = np.clip(b.get_paths()[0].vertices[:, 1], m, np.inf)
            b.set_color('r')
            b.set_facecolor('blue')
            b.set_edgecolor('blue')
        for partname in ('cbars','cmins','cmaxes','cmeans'):
            v = v2[partname]
            v.set_edgecolor('blue')
            v.set_linewidth(1)
            
        v2 = ax2[j].violinplot(tgt_dist[:,j].detach().cpu(), points=100, positions=[0],
                showmeans=True, showextrema=True, showmedians=False, vert=False)
        for b in v2['bodies']:
            # get the center
            m = np.mean(b.get_paths()[0].vertices[:, 1])
            b.get_paths()[0].vertices[:, 1] = np.clip(b.get_paths()[0].vertices[:, 1], - np.inf, m)
            b.set_color('r')
            b.set_facecolor('red')
            b.set_edgecolor('red')
        for partname in ('cbars','cmins','cmaxes','cmeans'):
            v = v2[partname]
            v.set_edgecolor('red')
            v.set_linewidth(1)
            
        ax2[j].set_yticks([0])
        ax2[j].set_yticklabels([])
        ax2[j].set_ylabel(PROSAILVARS[j])
        ax2[j].xaxis.grid(True)
        
        
    # Save the figure and show
    plt.tight_layout()
    plt.show()
    fig.savefig(res_dir + '/prosail_dist.svg')

def plot_pred_vs_tgt(res_dir, sim_dist, tgt_dist):
    for i in range(len(PROSAILVARS)):
        fig, ax = plt.subplots(figsize=(7,7), dpi=150)
        ax.scatter(sim_dist[:,i].detach().cpu(),tgt_dist[:,i].detach().cpu(), marker='.',s=2)
        ax.set_xlabel(f'{PROSAILVARS[i]} predicted')
        ax.set_ylabel(f'{PROSAILVARS[i]} reference')
        ax.set_xlim(ProsailVarsDist.Dists[PROSAILVARS[i]]["min"],
                    ProsailVarsDist.Dists[PROSAILVARS[i]]["max"])
        ax.set_ylim(ProsailVarsDist.Dists[PROSAILVARS[i]]["min"],
                    ProsailVarsDist.Dists[PROSAILVARS[i]]["max"])
        ax.plot([ProsailVarsDist.Dists[PROSAILVARS[i]]["min"], 
                 ProsailVarsDist.Dists[PROSAILVARS[i]]["max"]],
                [ProsailVarsDist.Dists[PROSAILVARS[i]]["min"], 
                 ProsailVarsDist.Dists[PROSAILVARS[i]]["max"]],color='black')
        fig.savefig(res_dir + f'/pred_vs_ref_{PROSAILVARS[i]}.svg')

def plot_refl_dist(rec_dist, refl_dist, res_dir, normalized=False, ssimulator=None, bands_name=None):
    if bands_name is None:
        bands_name = BANDS

    filename='/sim_refl_dist.svg'
    xmax=1
    xmin=0
    if normalized:
        # bands_dist = (bands_dist - ssimulator.norm_mean) / ssimulator.norm_std
        filename='/sim_normalized_refl_dist.svg'
        xmax=6
        xmin=-6
    fig = plt.figure(figsize=(18,12), dpi=150,)
    ax2=[]
    gs = fig.add_gridspec(len(bands_name),1)
    for j in range(len(bands_name)):
        ax2.append(fig.add_subplot(gs[j, 0]))
    
    for j in range(len(bands_name)):
        v2 = ax2[j].violinplot(rec_dist[:,j].squeeze().detach().cpu(), points=100, positions=[0],
                showmeans=True, showextrema=True, showmedians=False, vert=False)

        ax2[j].set_xlim(xmin, xmax)

        
        for b in v2['bodies']:
            # get the center
            m = np.mean(b.get_paths()[0].vertices[:, 1])
            b.get_paths()[0].vertices[:, 1] = np.clip(b.get_paths()[0].vertices[:, 1], m, np.inf)
            b.set_color('r')
            b.set_facecolor('blue')
            b.set_edgecolor('blue')
        for partname in ('cbars','cmins','cmaxes','cmeans'):
            v = v2[partname]
            v.set_edgecolor('blue')
            v.set_linewidth(1)
        
        v2 = ax2[j].violinplot(refl_dist[:,j].squeeze().detach().cpu(), points=100, positions=[0],
                showmeans=True, showextrema=True, showmedians=False, vert=False)

        ax2[j].set_xlim(xmin, xmax)

        
        for b in v2['bodies']:
            # get the center
            m = np.mean(b.get_paths()[0].vertices[:, 1])
            b.get_paths()[0].vertices[:, 1] = np.clip(b.get_paths()[0].vertices[:, 1], -np.inf, m)
            b.set_color('r')
            b.set_facecolor('red')
            b.set_edgecolor('red')
        for partname in ('cbars','cmins','cmaxes','cmeans'):
            v = v2[partname]
            v.set_edgecolor('red')
            v.set_linewidth(1)
            
            
        ax2[j].set_yticks([0])
        ax2[j].set_yticklabels([])
        ax2[j].set_ylabel(bands_name[j])
        ax2[j].xaxis.grid(True)
        
        
    # Save the figure and show
    plt.tight_layout()
    plt.show()
    fig.savefig(res_dir + filename)
    return 

def pair_plot(tensor_1, tensor_2=None, features = ["",""], res_dir='', 
              filename='pair_plot.png'):
    def plot_single_pair(ax, feature_ind1, feature_ind2, _X, _y, _features, colormap):
        """Plots single pair of features.
    
        Parameters
        ----------
        ax : Axes
            matplotlib axis to be plotted
        feature_ind1 : int
            index of first feature to be plotted
        feature_ind2 : int
            index of second feature to be plotted
        _X : numpy.ndarray
            Feature dataset of of shape m x n
        _y : numpy.ndarray
            Target list of shape 1 x n
        _features : list of str
            List of n feature titles
        colormap : dict
            Color map of classes existing in target
    
        Returns
        -------
        None
        """
    
        # Plot distribution histogram if the features are the same (diagonal of the pair-plot).
        if feature_ind1 == feature_ind2:
            tdf = pd.DataFrame(_X[:, [feature_ind1]], columns = [_features[feature_ind1]])
            tdf['target'] = _y
            for c in colormap.keys():
                tdf_filtered = tdf.loc[tdf['target']==c]
                ax[feature_ind1, feature_ind2].hist(tdf_filtered[_features[feature_ind1]], color = colormap[c], bins = 30)
        else:
            # other wise plot the pair-wise scatter plot
            tdf = pd.DataFrame(_X[:, [feature_ind1, feature_ind2]], columns = [_features[feature_ind1], _features[feature_ind2]])
            tdf['target'] = _y
            for c in colormap.keys():
                tdf_filtered = tdf.loc[tdf['target']==c]
                ax[feature_ind1, feature_ind2].scatter(x = tdf_filtered[_features[feature_ind2]], y = tdf_filtered[_features[feature_ind1]], color=colormap[c], marker='.',s=2)
    
        # Print the feature labels only on the left side of the pair-plot figure
        # and bottom side of the pair-plot figure. 
        # Here avoiding printing the labels for inner axis plots.
        if feature_ind1 == len(_features) - 1:
            ax[feature_ind1, feature_ind2].set(xlabel=_features[feature_ind2], ylabel='')
        if feature_ind2 == 0:
            if feature_ind1 == len(_features) - 1:
                ax[feature_ind1, feature_ind2].set(xlabel=_features[feature_ind2], ylabel=_features[feature_ind1])
            else:
                ax[feature_ind1, feature_ind2].set(xlabel='', ylabel=_features[feature_ind1])
    
    def myplotGrid(X, y, features, colormap={0: "red", 1: "green", 2: "blue"}):
        """Plots a pair grid of the given features.
    
        Parameters
        ----------
        X : numpy.ndarray
            Dataset of shape m x n
        y : numpy.ndarray
            Target list of shape 1 x n
        features : list of str
            List of n feature titles
    
        Returns
        -------
        None
        """
    
        feature_count = len(features)
        # Create a matplot subplot area with the size of [feature count x feature count]
        fig, axis = plt.subplots(nrows=feature_count, ncols=feature_count)
        # Setting figure size helps to optimize the figure size according to the feature count.
        fig.set_size_inches(feature_count * 4, feature_count * 4)
    
        # Iterate through features to plot pairwise.
        for i in range(0, feature_count):
            for j in range(0, feature_count):
                plot_single_pair(axis, i, j, X, y, features, colormap)

        plt.show()
        return fig, axis
    X = tensor_1.detach().cpu().numpy()
    y = np.zeros(tensor_1.size(0))
    if tensor_2 is not None:
        X = np.concatenate((X,tensor_2.detach().cpu().numpy()))
        y = np.concatenate((y,np.ones(tensor_2.size(0))))
    fig, axis = myplotGrid(X, y, features, colormap={0:'blue', 1:'red'})
    fig.savefig(res_dir + filename)
    return

def plot_rec_error_vs_angles(tgt_dist, rec_dist, angles_dist,  res_dir='',):
    error_dist = (tgt_dist - rec_dist).abs().mean(1)
    fig, axs = plt.subplots(3,1,dpi=150)
    axs[0].scatter(angles_dist[:,0].detach().cpu().squeeze().numpy(), 
                    error_dist.detach().cpu().squeeze().numpy(), marker='.',s=2)
    axs[0].set_ylabel('Reconstruction \n MAE')
    axs[0].set_xlabel("Sun zenith")

    axs[1].scatter(angles_dist[:,1].detach().cpu().squeeze().numpy(), 
                    error_dist.detach().cpu().squeeze().numpy(), marker='.',s=2)
    axs[1].set_ylabel('Reconstruction \n MAE')
    axs[1].set_xlabel("S2 zenith")

    axs[2].scatter(angles_dist[:,2].detach().cpu().squeeze().numpy(), 
                    error_dist.detach().cpu().squeeze().numpy(), marker='.',s=2)
    axs[2].set_ylabel('Reconstruction \n MAE')
    axs[2].set_xlabel("Sun/S2 Relative azimuth")

    fig.savefig(res_dir+"/error_vs_angles.png")
    return

def plot_metric_boxplot(metric_percentiles, res_dir, metric_name='ae', model_names=None, 
                        features_names=PROSAILVARS, pltformat='slides', logscale=False, sharey=True):
    """Metric percentile sizes : if 3 : models 0, percentiles 1, features 2
                                 if 2 : percentiles 0, features 1"""
    if len(metric_percentiles.size())==2:
        n_suplots = metric_percentiles.size(1)
        if not sharey:
            fig, axs =  plt.subplots(1, n_suplots, dpi=150, sharey=sharey)
            fig.tight_layout()
            for i in range(n_suplots):
                bplot = customized_box_plot(metric_percentiles[:,i], axs[i], redraw=True, patch_artist=True)
                for box in bplot['boxes']:
                    box.set(color='green')
                for median in bplot['medians']:
                    median.set(color='k')
                axs[i].set_xticks([])
                axs[i].set_xticklabels([])
                if features_names is not None:
                    axs[i].title.set_text(features_names[i])  
                if logscale:
                    axs[i].set_yscale('symlog', linthresh=1e-5)
                axs[i].yaxis.grid(True)
            fig.tight_layout()
        else:
            fig, axs =  plt.subplots(1, 1, dpi=150, sharey=sharey)
            
            bplot = customized_box_plot(metric_percentiles, axs, redraw=True, patch_artist=True,)
            for box in bplot['boxes']:
                box.set(color='green')
            for median in bplot['medians']:
                    median.set(color='k')
            if features_names is not None:
                axs.set_xticklabels(features_names)
            if logscale:
                axs.set_yscale('symlog', linthresh=1e-5)
            axs.yaxis.grid(True)
            fig.tight_layout()
    elif len(metric_percentiles.size())==3:
        n_models = metric_percentiles.size(0)
        if model_names is None or len(model_names)!=n_models:
            model_names = [str(i+1) for i in range(n_models)]
        n_suplots = metric_percentiles.size(2)
        if pltformat=='article':
            n_rows = n_suplots // 2 + n_suplots % 2
            n_cols = 2
            figsize = (8.27, 11.69) #A4 paper size in inches
        else:
            n_rows = 2
            n_cols = n_suplots // 2 + n_suplots % 2
            figsize = (16, 9)
        fig, axs =  plt.subplots(n_rows, n_cols, dpi=150, figsize=figsize, sharey=sharey)
        if n_suplots%2==1:
            fig.delaxes(axs[-1, -1])
        for i in range(n_suplots):
            if pltformat=='article':
                row = i//2
                col = i%2
            else:
                row = i%2 
                col = i//2
            bplot = customized_box_plot(metric_percentiles[:,:,i].transpose(0,1), axs[row, col], redraw = True, 
                                patch_artist=True)
            cmap = plt.cm.get_cmap('rainbow')
            colors = [cmap(val/n_models) for val in range(n_models)]
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)
            for median in bplot['medians']:
                    median.set(color='k', linewidth=2,)
            axs[row, col].set_xticklabels(model_names)
            if features_names is not None:
                axs[row, col].set_title(features_names[i])  
            if logscale:
                axs[row, col].set_yscale('symlog', linthresh=1e-5)  
            axs[row, col].yaxis.grid(True) 
        fig.tight_layout()
    else:
        raise NotImplementedError()
    fig.savefig(res_dir + f"/{metric_name}_boxplot.svg")
    pass

def customized_box_plot(percentiles_tensor, axes, redraw = True, *args, **kwargs):
    """
    Generates a customized boxplot based on the given percentile values
    """
    if len(percentiles_tensor.size())==1:
        n_box = 1
        percentiles_tensor = percentiles_tensor.unsqueeze(1)
    else:
        n_box = percentiles_tensor.size(1)
    box_plot = axes.boxplot([[-9, -4, 2, 4, 9],]*n_box, *args, **kwargs) 
    # Creates len(percentiles) no of box plots

    min_y, max_y = float('inf'), -float('inf')

    for box_no in range(n_box):
        pdata = percentiles_tensor[:,box_no]
        if len(pdata) == 6:
            (q1_start, q2_start, q3_start, q4_start, q4_end, fliers_xy) = pdata
        elif len(pdata) == 5:
            (q1_start, q2_start, q3_start, q4_start, q4_end) = pdata
            fliers_xy = None
        else:
            raise ValueError("Percentile arrays for customized_box_plot must have either 5 or 6 values")

        # Lower cap
        box_plot['caps'][2*box_no].set_ydata([q1_start, q1_start])
        # xdata is determined by the width of the box plot

        # Lower whiskers
        box_plot['whiskers'][2*box_no].set_ydata([q1_start, q2_start])

        # Higher cap
        box_plot['caps'][2*box_no + 1].set_ydata([q4_end, q4_end])

        # Higher whiskers
        box_plot['whiskers'][2*box_no + 1].set_ydata([q4_start, q4_end])

        # Box
        path = box_plot['boxes'][box_no].get_path()
        path.vertices[0][1] = q2_start
        path.vertices[1][1] = q2_start
        path.vertices[2][1] = q4_start
        path.vertices[3][1] = q4_start
        path.vertices[4][1] = q2_start

        # Median
        box_plot['medians'][box_no].set_ydata([q3_start, q3_start])

        # Outliers
        if fliers_xy is not None and len(fliers_xy[0]) != 0:
            # If outliers exist
            box_plot['fliers'][box_no].set(xdata = fliers_xy[0],
                                           ydata = fliers_xy[1])

            min_y = min(q1_start, min_y, fliers_xy[1].min())
            max_y = max(q4_end, max_y, fliers_xy[1].max())

        else:
            min_y = min(q1_start, min_y)
            max_y = max(q4_end, max_y)

        # The y axis is rescaled to fit the new box plot completely with 10% 
        # of the maximum value at both ends
        axes.set_ylim([min_y*1.1, max_y*1.1])

    # If redraw is set to true, the canvas is updated.
    if redraw:
        axes.figure.canvas.draw()

    return box_plot

def gammacorr(s2_r,gamma=2):
    return s2_r.pow(1/gamma)

def normalize_patch_for_plot(s2_r_rgb, sr_min=None, sr_max=None):
    if len(s2_r_rgb.size())==3:
        assert s2_r_rgb.size(2)==3
        s2_r_rgb = s2_r_rgb.unsqueeze(0)
    else:
        assert s2_r_rgb.size(3)==3
    if sr_min is None or sr_max is None:
        sr = s2_r_rgb.reshape(-1, 3)
        sr_min = sr.min(0)[0]
        sr_max = sr.max(0)[0]
    else:
        assert sr_min.squeeze().size(0)==3
        assert sr_max.squeeze().size(0)==3
    sr_min = sr_min.squeeze().unsqueeze(0).unsqueeze(0).unsqueeze(0)
    sr_max = sr_max.squeeze().unsqueeze(0).unsqueeze(0).unsqueeze(0)
    return (s2_r_rgb - sr_min) / (sr_max - sr_min), sr_min, sr_max

def plot_patch_pairs(s2_r_pred, s2_r_ref, idx=0):
    # s2_r = swap_reflectances(s2_r)
    mean_l2_err = (s2_r_pred - s2_r_ref).pow(2).mean(1).unsqueeze(1).permute(0,2,3,1).detach().cpu()
    s2_r_ref_n, sr_min, sr_max = normalize_patch_for_plot(s2_r_ref[:,:3,:,:].permute(0,2,3,1).detach().cpu(), sr_min=None, sr_max=None)
    s2_r_ref_n_rgb = s2_r_ref_n[:,:,:,torch.tensor([2,1,0])] + 0.0
    s2_r_pred_n, sr_min, sr_max = normalize_patch_for_plot(s2_r_pred[:,:3,:,:].permute(0,2,3,1).detach().cpu(), sr_min=sr_min, sr_max=sr_max)
    s2_r_pred_n_rgb = s2_r_pred_n[:,:,:,torch.tensor([2,1,0])] + 0.0
    fig, ax = plt.subplots(1, 3, dpi=150, figsize=(9,3))
    ax[0].imshow(gammacorr(s2_r_ref_n_rgb[idx,:,:,:]))
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_title("Original patch")
    ax[1].imshow(gammacorr(s2_r_pred_n_rgb[idx,:,:,:]))
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].set_title("Reconstructed patch")
    ax[2].imshow(gammacorr(mean_l2_err[idx,:,:,:]))
    ax[2].set_xticks([])
    ax[2].set_yticks([])
    ax[2].set_title("L2 errors")
    return fig, ax

def plot_lai_preds(lais, lai_pred, time_delta=None, site=''):
    fig, ax = plt.subplots()
    lai_i = lais.squeeze()
    if time_delta is not None:
        sc = ax.scatter(lai_pred, lai_i, c=time_delta.abs())
        cbar = plt.colorbar(sc)
        cbar.ax.set_ylabel('Delta between reflectance and in situ measure (days)', rotation=270)
        cbar.ax.yaxis.set_label_coords(0.0,0.5)
    else:
        sc = ax.scatter(lai_pred, lai_i, s=1)
    ax.plot([min(lai_i.min(), lai_pred.min()),max(lai_i.max(), lai_pred.max())],
            [min(lai_i.min(), lai_pred.min()),max(lai_i.max(), lai_pred.max())],'k--')
    ax.set_xlabel('Predicted LAI')
    ax.set_ylabel(f"{site} LAI")# {LAI_columns(site)[i]}")
    ax.set_aspect('equal', 'box')
    # plt.gray()

    plt.show()
    return fig, ax