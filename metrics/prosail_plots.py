#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 11:46:20 2022

@author: yoel
"""
import matplotlib.pyplot as plt
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
    fig.savefig(save_dir+"/picp.svg")
    
    fig = plt.figure(dpi=200)
    for i in range(len(maer.columns)):
        plt.plot(alpha_pi, mpiwr.values[:,i], label=maer.columns[i])
    plt.legend()
    plt.xlabel('1-a')
    plt.ylabel('Mean Prediction Interval Width (Standardized)')
    fig.savefig(save_dir+"/MPIWr.svg")
    
    fig = plt.figure()
    plt.grid(which='both', axis='y')
    ax = fig.add_axes([0,0,1,1])
    ax.bar(maer.columns, maer.values.reshape(-1),)
    plt.ylabel('Mean Absolute Error (Standardized)')
    fig.savefig(save_dir+"/MAEr.svg")
    
    fig = plt.figure(dpi=150)
    plt.grid(which='both', axis='y')
    ax = fig.add_axes([0,0,1,1])
    ax.bar(maer.columns, mare.detach().cpu().numpy())
    plt.ylabel('Mean Absolute Relative Error')
    plt.yscale('log')
    fig.savefig(save_dir+"/mare.svg")

def plot_rec_hist2D(prosail_VAE, loader, res_dir, nbin=50):
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
        ys = s2_r_dist[:,i].detach().cpu().numpy()
        min_b = min(xs.min(),ys.min())
        max_b = max(xs.max(),ys.max())
        xedges = np.linspace(min_b, max_b, nbin)
        yedges = np.linspace(min_b, max_b, nbin)
        heatmap = 0
        for j in range(N):
            xj = xs[j,:]
            yj = ys[j]
            hist, xedges, yedges = np.histogram2d(
                xj, np.ones_like(xj) * yj, bins=[xedges, yedges])
            heatmap += hist
        heatmap = np.flipud(np.rot90(heatmap))
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        
        axs[axi, axj].imshow(heatmap, extent=extent, interpolation='nearest',cmap='plasma')
        axs[axi, axj].set_ylabel(BANDS[i])
        axs[axi, axj].set_xlabel("rec. " + BANDS[i])
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
        xs = sim_supports[:,i,:].detach().cpu().numpy()
        ys = tgt_dist[:,i].detach().cpu().numpy()
        weights = sim_pdfs[:,i,:].detach().cpu().numpy()
        min_b = ProsailVarsDist.Dists[PROSAILVARS[i]]["min"]
        max_b = ProsailVarsDist.Dists[PROSAILVARS[i]]["max"]
        xedges = np.linspace(min_b, max_b, nbin)
        yedges = np.linspace(min_b, max_b, nbin)
        heatmap = 0
        for j in range(N):
            xj = xs[j,:]
            yj = ys[j]
            wj = weights[j,:]
            hist, xedges, yedges = np.histogram2d(
                xj, np.ones_like(xj) * yj, bins=[xedges, yedges], weights=wj)
            heatmap += hist

        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        heatmap = np.flipud(np.rot90(heatmap))
        fig, ax = plt.subplots(dpi=120)
        ax.imshow(heatmap, extent=extent, interpolation='nearest',cmap='plasma')
        ax.set_ylabel(PROSAILVARS[i])
        ax.set_xlabel("Predicted distribution of " + PROSAILVARS[i])
        ax.plot([min_b, max_b], [min_b, max_b], c='w')
        axs_all[i%2, i//2].imshow(heatmap, extent=extent, interpolation='nearest',cmap='plasma')
        axs_all[i%2, i//2].set_ylabel(PROSAILVARS[i])
        axs_all[i%2, i//2].set_xlabel("Pred. " + PROSAILVARS[i])
        axs_all[i%2, i//2].plot([min_b, max_b], [min_b, max_b], c='w')
        fig.savefig(res_dir + f'/2d_pred_dist_{PROSAILVARS[i]}.svg')
        plt.close('all')
    if n_lats%2==1:
        fig_all.delaxes(axs_all[-1, -1])
    fig_all.savefig(res_dir + f'/2d_pred_dist_PROSAIL_VARS.svg')
    
    pass

def plot_rec_and_latent(prosail_VAE, loader, res_dir, n_plots=10):
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
        
        
        
        rec_samples = [rec_samples[j,:] for j in range(len(BANDS))]
        sim_samples = sim.squeeze().detach().cpu().numpy()
        sim_samples = [sim_samples[j,:] for j in range(len(PROSAILVARS))]
        
        ind1 = np.arange(len(BANDS))
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
        ax1.set_yticklabels(BANDS)
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
    for i in range(len(loss_names)):
        loss = loss_df[loss_names[i]].values
        if log_scale: # (loss<=0).any() or 
            if (loss<=0).any():
                loss += loss.min() + 1
            ax.set_yscale('log')
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

def plot_refl_dist(rec_dist, refl_dist, res_dir, normalized=False, ssimulator=None):

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
    gs = fig.add_gridspec(len(BANDS),1)
    for j in range(len(BANDS)):
        ax2.append(fig.add_subplot(gs[j, 0]))
    
    for j in range(len(BANDS)):
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
        ax2[j].set_ylabel(BANDS[j])
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