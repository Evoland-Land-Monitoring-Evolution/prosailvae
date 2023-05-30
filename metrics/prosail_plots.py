#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 11:46:20 2022

@author: yoel
"""

import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# plt.rcParams.update({
#   "text.usetex": True,
#   "font.family": "Helvetica"
# })

import numpy as np
import pandas as pd
import torch
from prosailvae.ProsailSimus import PROSAILVARS, ProsailVarsDist, BANDS
# from sensorsio.utils import rgb_render
from utils.image_utils import rgb_render

from mpl_toolkits.axes_grid1 import make_axes_locatable
from dataset.prepare_silvia_validation import load_validation_data
import seaborn as sns
import os

def plot_patches(patch_list, title_list=[], use_same_visu=True, colorbar=True, vmin=None, vmax=None):
    fig, axs = plt.subplots(1, len(patch_list), figsize=(3*len(patch_list), 3), dpi=200)
    minvisu = None 
    maxvisu = None
    for i, patch in enumerate(patch_list):
        # patch = patch.squeeze()
        if patch.size(0)==1:
            tensor_visu = patch_list[i].squeeze()
            im = axs[i].imshow(tensor_visu, vmin=vmin, vmax=vmax)#, cmap='YlGn')
            divider = make_axes_locatable(axs[i])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            if colorbar:
                fig.colorbar(im, cax=cax, orientation='vertical')
            else:
                plt.delaxes(ax = cax)
        else:
            if use_same_visu:
                tensor_visu, minvisu, maxvisu = rgb_render(patch, dmin=minvisu, dmax=maxvisu)
            else:
                tensor_visu, _, _ = rgb_render(patch, dmin=minvisu, dmax=maxvisu)
            axs[i].imshow(tensor_visu)
            divider = make_axes_locatable(axs[i])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            plt.delaxes(ax = cax)
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        if len(title_list) == len(patch_list):
            axs[i].set_title(title_list[i])
    return fig, axs

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
    weights = sim_pdf.detach().cpu().numpy()
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
    
def loss_curve(loss_df, save_file):
    loss_names = loss_df.columns.values.tolist()
    loss_names.remove("epoch")
    epochs = loss_df["epoch"]
    fig, ax = plt.subplots(dpi=150)
    min_loss=1000000
    if "loss_sum" in loss_names:
        loss_sum_min = loss_df['loss_sum'].values.min()
        loss_sum_min_epoch = loss_df['loss_sum'].values.argmin()
        ax.scatter([loss_sum_min_epoch], [loss_sum_min], label="loss_sum min")
    for i in range(len(loss_names)):
        loss = loss_df[loss_names[i]].values
        min_loss = min(loss.min(), min_loss)
        ax.plot(epochs,loss, label=loss_names[i])
    if min_loss>0:
        ax.set_yscale('log')
    else:
        ax.set_yscale('symlog', linthresh=1e-5)
        ax.set_ylim(bottom=min(0, min_loss))
    ax.legend()
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    fig.savefig(save_file)

def all_loss_curve(train_loss_df, valid_loss_df, info_df, save_file, log_scale=False):
    loss_names = train_loss_df.columns.values.tolist()
    loss_names.remove("epoch")
    epochs = train_loss_df["epoch"]
    fig, axs = plt.subplots(3,1, dpi=150, sharex=True)

    for i in range(len(loss_names)):
        train_loss = train_loss_df[loss_names[i]].values
        valid_loss = valid_loss_df[loss_names[i]].values
        axs[0].plot(epochs,train_loss, label=loss_names[i])
        axs[1].plot(epochs,valid_loss, label=loss_names[i])
    axs[2].plot(epochs, info_df['lr'], label="lr")
    train_loss_sum_min = train_loss_df['loss_sum'].values.min()
    train_loss_sum_min_epoch = train_loss_df['loss_sum'].values.argmin()
    axs[0].scatter([train_loss_sum_min_epoch], [train_loss_sum_min], label="loss_sum min")
    valid_loss_sum_min = valid_loss_df['loss_sum'].values.min()
    valid_loss_sum_min_epoch = valid_loss_df['loss_sum'].values.argmin()
    axs[1].scatter([valid_loss_sum_min_epoch], [valid_loss_sum_min], label="loss_sum min")
    if train_loss_sum_min>0:
        axs[0].set_yscale('log')
    else:
        axs[0].set_yscale('symlog', linthresh=1e-5)
        axs[0].set_ylim(bottom=min(0, train_loss_sum_min))
    if valid_loss_sum_min>0:
        axs[1].set_yscale('log')
    else:
        axs[1].set_yscale('symlog', linthresh=1e-5)
        axs[1].set_ylim(bottom=min(0, valid_loss_sum_min))
    axs[2].set_yscale('log')
    for i in range(3):
        
        axs[i].legend(fontsize=8)
    axs[2].set_xlabel('epoch')
    axs[0].set_ylabel('Train loss')
    axs[1].set_ylabel('Valid loss')
    axs[2].set_ylabel('LR')
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
    if res_dir is not None:
        fig.savefig(res_dir + filename)
    return fig, ax2

def plot_param_compare_dist(rec_dist, refl_dist, res_dir, normalized=False, params_name=None):
    if params_name is None:
        params_name = PROSAILVARS + ['phi_s', "phi_o", "psi"]

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
    gs = fig.add_gridspec(len(params_name),1)
    for j in range(len(params_name)):
        ax2.append(fig.add_subplot(gs[j, 0]))
    
    for j in range(len(params_name)):
        v2 = ax2[j].violinplot(rec_dist[:,j].squeeze().detach().cpu(), points=100, positions=[0],
                showmeans=True, showextrema=True, showmedians=False, vert=False)

        # ax2[j].set_xlim(xmin, xmax)

        
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

        # ax2[j].set_xlim(xmin, xmax)

        
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
        ax2[j].set_ylabel(params_name[j])
        ax2[j].xaxis.grid(True)
        
        
    # Save the figure and show
    plt.tight_layout()
    plt.show()
    if res_dir is not None:
        fig.savefig(res_dir + filename)
    return fig, ax2

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
    m, b = np.polyfit(lai_i.numpy(), lai_pred.numpy(), 1)
    r2 = r2_score(lai_i.numpy(), lai_pred.numpy())
    mse = (lais - lai_pred).pow(2).mean().numpy()
    if time_delta is not None:
        sc = ax.scatter(lai_i, lai_pred, c=time_delta.abs(), s=5)
        cbar = plt.colorbar(sc)
        cbar.ax.set_ylabel('Delta between reflectance and in situ measure (days)', rotation=270)
        cbar.ax.yaxis.set_label_coords(0.0,0.5)
    else:
        sc = ax.scatter(lai_i, lai_pred, s=1)

    minlim = min(lai_i.min(), lai_pred.min())
    maxlim = max(lai_i.max(), lai_pred.max())
    ax.plot([minlim, maxlim],
            [minlim, maxlim],'k--')
    ax.plot([minlim, maxlim],
            [m * minlim + b, m * maxlim + b],'r', label="{:.2f} x + {:.2f}\n r2 = {:.2f}\n MSE: {:.2f}".format(m,b,r2,mse))
    ax.legend()
    ax.set_ylabel('Predicted LAI')
    ax.set_xlabel(f"{site} LAI")# {LAI_columns(site)[i]}")
    ax.set_aspect('equal', 'box')
    # plt.gray()

    plt.show()
    return fig, ax

def plot_lai_vs_ndvi(lais, ndvi, time_delta=None, site=''):
    fig, ax = plt.subplots()
    lai_i = lais.squeeze()
    if time_delta is not None:
        sc = ax.scatter(lai_i, ndvi, c=time_delta.abs(), s=5)
        cbar = plt.colorbar(sc)
        cbar.ax.set_ylabel('Delta between reflectance and in situ measure (days)', rotation=270)
        cbar.ax.yaxis.set_label_coords(0.0, 0.5)
    else:
        sc = ax.scatter(lai_i, ndvi, s=1)
    ax.set_ylabel('NDVI')
    ax.set_ylim(0,1)
    ax.set_xlabel(f"{site} LAI")# {LAI_columns(site)[i]}")
    # plt.gray()

    plt.show()
    return fig, ax

def PROSAIL_2D_aggregated_results(plot_dir, all_s2_r, all_rec, all_lai, all_cab, all_cw,
                                  all_vars, all_weiss_lai, all_weiss_cab, all_weiss_cw, all_sigma, all_ccc,
                                  all_cw_rel, max_sigma=1.4):

    fig, ax = plt.subplots()
    ax.scatter((all_lai - all_weiss_lai).abs(), all_sigma[6,:], s=0.5)
    ax.set_xlabel('LAI absolute difference (SNAP LAI - predicted LAI)')
    ax.set_ylabel('LAI latent sigma')
    fig.savefig(f"{plot_dir}/lai_err_vs_sigma.png")
    fig, ax = plt.subplots()
    ax.scatter((all_cab - all_weiss_cab).abs(), all_sigma[1,:], s=0.5)
    ax.set_xlabel('Cab absolute difference (SNAP Cab - predicted Cab)')
    ax.set_ylabel('Cab latent sigma')
    fig.savefig(f"{plot_dir}/cab_err_vs_sigma.png")
    fig, ax = plt.subplots()
    ax.scatter((all_cw - all_weiss_cw).abs(), all_sigma[4,:], s=0.5)
    ax.set_xlabel('Cw absolute difference (SNAP Cw - predicted Cw)')
    ax.set_ylabel('Cw latent sigma')
    fig.savefig(f"{plot_dir}/cw_err_vs_sigma.png")

    n_cols = 4
    n_rows = 3
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(2*n_cols,n_rows*2), tight_layout=True, dpi=150)
    for idx, prosail_var in enumerate(PROSAILVARS):
        row = idx // n_cols
        col = idx % n_cols
        ax[row, col].hist(all_vars[idx,...].reshape(-1).cpu(), bins=50, density=True)
        ax[row, col].set_yticks([])
        ax[row, col].set_ylabel(prosail_var)
        ax[row, col].set_xlim(ProsailVarsDist.Dists[PROSAILVARS[idx]]['min'],
                              ProsailVarsDist.Dists[PROSAILVARS[idx]]['max'])
    fig.delaxes(ax[-1, -1])
    fig.suptitle(f"PROSAIL variables distributions")
    fig.savefig(f"{plot_dir}/all_prosail_var_pred_dist.png")
    n_cols = 4
    n_rows = 3
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(2*n_cols,n_rows*2), tight_layout=True, dpi=150, sharex=True)
    for idx, prosail_var in enumerate(PROSAILVARS):
        row = idx // n_cols
        col = idx % n_cols
        ax[row, col].hist(all_sigma[idx,...].reshape(-1).cpu(), bins=100, density=True, range=[0, max_sigma])
        ax[row, col].set_yticks([])
        ax[row, col].set_xlim(0, max_sigma)
        ax[row, col].set_ylabel(prosail_var)
    fig.delaxes(ax[-1, -1])
    fig.suptitle(f"PROSAIL variables sigma")
    fig.savefig(f"{plot_dir}/all_prosail_var_sigma.png")
    n_cols = 5
    n_rows = 2
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(2*n_cols,n_rows*2), tight_layout=True, dpi=150)
    for idx, band in enumerate(BANDS):
        row = idx // n_cols
        col = idx % n_cols
        ax[row, col].scatter(all_s2_r[idx,:].reshape(-1).cpu(),
                            all_rec[idx,:].reshape(-1).cpu(), s=0.5)
        xlim = ax[row, col].get_xlim()
        ylim = ax[row, col].get_ylim()
        ax[row, col].plot([min(xlim[0],ylim[0]), max(xlim[1],ylim[1])],
                        [min(xlim[0],ylim[0]), max(xlim[1],ylim[1]), ],'k--')
        ax[row, col].set_yticks([])
        ax[row, col].set_ylabel(f"Reconstructed {band}")
        ax[row, col].set_xlabel(f"True {band}")
        ax[row, col].set_aspect('equal')
    fig.savefig(f"{plot_dir}/all_bands_scatter_true_vs_pred.png")
    n_cols = 5
    n_rows = 2
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(2*n_cols, n_rows*2), tight_layout=True, dpi=150)
    for idx, band in enumerate(BANDS):
        row = idx // n_cols
        col = idx % n_cols
        xmin = min(all_s2_r[idx,:].cpu().min().item(), all_rec[idx,:].cpu().min().item())
        xmax = max(all_s2_r[idx,:].cpu().max().item(), all_rec[idx,:].cpu().max().item())
        ax[row, col].hist2d(all_s2_r[idx,:].reshape(-1).numpy(),
                            all_rec[idx,:].reshape(-1).cpu().numpy(),
                            range = [[xmin,xmax],[xmin,xmax]], bins=100, cmap='BrBG')
        xlim = ax[row, col].get_xlim()
        ylim = ax[row, col].get_ylim()
        ax[row, col].plot([min(xlim[0],ylim[0]), max(xlim[1],ylim[1])],
                        [min(xlim[0],ylim[0]), max(xlim[1],ylim[1]), ],'k--')
        ax[row, col].set_yticks([])
        ax[row, col].set_ylabel(f"Reconstructed {band}")
        ax[row, col].set_xlabel(f"True {band}")
        ax[row, col].set_aspect('equal')
    fig.savefig(f"{plot_dir}/all_bands_2dhist_true_vs_pred.png")
    fig, ax = plt.subplots(1, tight_layout=True, dpi=150)
    xmin = min(all_lai.cpu().min().item(), all_weiss_lai.cpu().min().item())
    xmax = max(all_lai.cpu().max().item(), all_weiss_lai.cpu().max().item())
    ax.hist2d(all_weiss_lai.cpu().numpy(), all_lai.cpu().numpy(),
              range = [[xmin,xmax], [xmin,xmax]], bins=100, cmap='BrBG')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.plot([min(xlim[0],ylim[0]), max(xlim[1],ylim[1])],
            [min(xlim[0],ylim[0]), max(xlim[1],ylim[1]), ],'k--')
    ax.set_ylabel("Predicted LAI")
    ax.set_xlabel("SNAP LAI")
    ax.set_aspect('equal')
    fig.savefig(f"{plot_dir}/all_lai_2dhist_true_vs_pred.png")

    fig, ax = plt.subplots(1, tight_layout=True, dpi=150)
    m, b = np.polyfit(all_weiss_lai.cpu().numpy(), all_lai.cpu().numpy(), 1)
    r2 = r2_score(all_weiss_lai.cpu().numpy(), all_lai.cpu().numpy())
    mse = (all_weiss_lai - all_lai).pow(2).mean().cpu().numpy()
    xmin = min(all_lai.cpu().min().item(), all_weiss_lai.cpu().min().item())
    xmax = max(all_lai.cpu().max().item(), all_weiss_lai.cpu().max().item())
    ax.scatter(all_weiss_lai.cpu().numpy(),
                        all_lai.cpu().numpy(),s=0.5)
    ax.plot([xmin, xmax],
            [m * xmin + b, m * xmax + b],'r', 
            label="{:.2f} x + {:.2f}\n r2 = {:.2f}\n MSE: {:.2f}".format(m,b,r2,mse))
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.plot([min(xlim[0],ylim[0]), max(xlim[1],ylim[1])],
                    [min(xlim[0],ylim[0]), max(xlim[1],ylim[1]), ],'k--')
    ax.legend()
    ax.set_ylabel("Predicted LAI")
    ax.set_xlabel("SNAP LAI")
    ax.set_aspect('equal')
    fig.savefig(f"{plot_dir}/all_lai_scatter_true_vs_pred.png")

    ccc = all_cab * all_lai
    fig, ax = plt.subplots(1, tight_layout=True, dpi=150)
    m, b = np.polyfit(all_weiss_cab.cpu().numpy(), ccc.cpu().numpy(), 1)
    r2 = r2_score(all_weiss_cab.cpu().numpy(), ccc.cpu().numpy())
    mse = (all_weiss_cab - ccc).pow(2).mean().cpu().numpy()
    xmin = min(ccc.cpu().min().item(), all_weiss_cab.cpu().min().item())
    xmax = max(ccc.cpu().max().item(), all_weiss_cab.cpu().max().item())
    ax.scatter(all_weiss_cab.cpu().numpy(),
                        ccc.cpu().numpy(),s=0.5)
    ax.plot([xmin, xmax],
            [m * xmin + b, m * xmax + b],'r', 
            label="{:.2f} x + {:.2f}\n r2 = {:.2f}\n MSE: {:.2f}".format(m,b,r2,mse))
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.plot([min(xlim[0],ylim[0]), max(xlim[1],ylim[1])],
                    [min(xlim[0],ylim[0]), max(xlim[1],ylim[1]), ],'k--')
    ax.legend()
    ax.set_ylabel(f"Predicted CCC")
    ax.set_xlabel(f"SNAP CCC")
    ax.set_aspect('equal')
    fig.savefig(f"{plot_dir}/all_ccc_scatter_true_vs_pred.png")

    # fig, ax = plt.subplots(1, tight_layout=True, dpi=150)
    # weiss_ccc = all_weiss_cab / all_weiss_lai * 10
    # m, b = np.polyfit(weiss_ccc.cpu().numpy(), all_cab.cpu().numpy(), 1)
    # r2 = r2_score(weiss_ccc.cpu().numpy(), all_cab.cpu().numpy())
    # mse = (weiss_ccc - all_cab).pow(2).mean().cpu().numpy()
    # xmin = min(all_cab.cpu().min().item(), weiss_ccc.cpu().min().item())
    # xmax = max(all_cab.cpu().max().item(), weiss_ccc.cpu().max().item())
    # ax.scatter(weiss_ccc.cpu().numpy(),
    #                     all_cab.cpu().numpy(),s=0.5)
    # ax.plot([xmin, xmax],
    #         [m * xmin + b, m * xmax + b],'r', 
    #         label="{:.2f} x + {:.2f}\n r2 = {:.2f}\n MSE: {:.2f}".format(m,b,r2,mse))
    # xlim = ax.get_xlim()
    # ylim = ax.get_ylim()
    # ax.plot([min(xlim[0],ylim[0]), max(xlim[1],ylim[1])],
    #                 [min(xlim[0],ylim[0]), max(xlim[1],ylim[1]), ],'k--')
    # ax.legend()
    # ax.set_ylabel(f"Predicted Cab")
    # ax.set_xlabel(f"SNAP 'CCC'")
    # ax.set_aspect('equal')
    # fig.savefig(f"{plot_dir}/all_other_cab_scatter_true_vs_pred.png")

    # fig, ax = plt.subplots(1, tight_layout=True, dpi=150)
    # m, b = np.polyfit(all_weiss_cab.cpu().numpy()*10, all_cab.cpu().numpy(), 1)
    # r2 = r2_score(all_weiss_cab.cpu().numpy()*10, all_cab.cpu().numpy())
    # mse = (all_weiss_cab*10 - all_cab).pow(2).mean().cpu().numpy()
    # xmin = min(all_cab.cpu().min().item(), all_weiss_cab.cpu().min().item()*10)
    # xmax = max(all_cab.cpu().max().item(), all_weiss_cab.cpu().max().item()*10)
    # ax.scatter(all_weiss_cab.cpu().numpy()*10,
    #                     all_cab.cpu().numpy(),s=0.5)
    # ax.plot([xmin, xmax],
    #         [m * xmin + b, m * xmax + b],'r', 
    #         label="{:.2f} x + {:.2f}\n r2 = {:.2f}\n MSE: {:.2f}".format(m,b,r2,mse))
    # xlim = ax.get_xlim()
    # ylim = ax.get_ylim()
    # ax.plot([min(xlim[0],ylim[0]), max(xlim[1],ylim[1])],
    #                 [min(xlim[0],ylim[0]), max(xlim[1],ylim[1]), ],'k--')
    # ax.legend()
    # ax.set_ylabel(f"Predicted Cab")
    # ax.set_xlabel(f"SNAP Cab")
    # ax.set_aspect('equal')
    # fig.savefig(f"{plot_dir}/all_cab_times10_scatter_true_vs_pred.png")

    # fig, ax = plt.subplots(1, tight_layout=True, dpi=150)
    # m, b = np.polyfit(all_weiss_cab.cpu().numpy(), all_ccc.cpu().numpy(), 1)
    # r2 = r2_score(all_weiss_cab.cpu().numpy(), all_ccc.cpu().numpy())
    # mse = (all_weiss_cab - all_ccc).pow(2).mean().cpu().numpy()
    # xmin = min(all_ccc.cpu().min().item(), all_weiss_cab.cpu().min().item())
    # xmax = max(all_ccc.cpu().max().item(), all_weiss_cab.cpu().max().item())
    # ax.scatter(all_weiss_cab.cpu().numpy(),
    #                     all_ccc.cpu().numpy(),s=0.5)
    # ax.plot([xmin, xmax],
    #         [m * xmin + b, m * xmax + b],'r', 
    #         label="{:.2f} x + {:.2f}\n r2 = {:.2f}\n MSE: {:.2f}".format(m,b,r2,mse))
    # xlim = ax.get_xlim()
    # ylim = ax.get_ylim()
    # ax.plot([min(xlim[0],ylim[0]), max(xlim[1],ylim[1])],
    #                 [min(xlim[0],ylim[0]), max(xlim[1],ylim[1]), ],'k--')
    # ax.legend()
    # ax.set_ylabel(f"Predicted CCC")
    # ax.set_xlabel(f"SNAP Cab")
    # ax.set_aspect('equal')
    # fig.savefig(f"{plot_dir}/all_ccc_scatter_true_vs_pred.png")

    cwc = all_lai * all_cw
    fig, ax = plt.subplots(1, tight_layout=True, dpi=150)
    m, b = np.polyfit(all_weiss_cw.cpu().numpy(), cwc.cpu().numpy(), 1)
    r2 = r2_score(all_weiss_cw.cpu().numpy(), cwc.cpu().numpy())
    mse = (all_weiss_cw - cwc).pow(2).mean().cpu().numpy()
    xmin = min(cwc.cpu().min().item(), all_weiss_cw.cpu().min().item())
    xmax = max(cwc.cpu().max().item(), all_weiss_cw.cpu().max().item())
    ax.scatter(all_weiss_cw.cpu().numpy(),
                        cwc.cpu().numpy(),s=0.5)
    ax.plot([xmin, xmax],
            [m * xmin + b, m * xmax + b],'r', 
            label="{:.2f} x + {:.2f}\n r2 = {:.2f}\n MSE: {:.2f}".format(m,b,r2,mse))
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.plot([min(xlim[0],ylim[0]), max(xlim[1],ylim[1])],
                    [min(xlim[0],ylim[0]), max(xlim[1],ylim[1]), ],'k--')
    ax.legend()
    ax.set_ylabel(f"Predicted CWC")
    ax.set_xlabel(f"SNAP CWC")
    ax.set_aspect('equal')
    fig.savefig(f"{plot_dir}/all_cwc_scatter_true_vs_pred.png")

    # fig, ax = plt.subplots(1, tight_layout=True, dpi=150)
    # m, b = np.polyfit(all_weiss_cw.cpu().numpy() * 10, all_cw.cpu().numpy(), 1)
    # r2 = r2_score(all_weiss_cw.cpu().numpy() * 10, all_cw.cpu().numpy())
    # mse = (all_weiss_cw * 10 - all_cw).pow(2).mean().cpu().numpy()
    # xmin = min(all_cw.cpu().min().item(), all_weiss_cw.cpu().min().item() * 10)
    # xmax = max(all_cw.cpu().max().item(), all_weiss_cw.cpu().max().item() * 10)
    # ax.scatter(all_weiss_cw.cpu().numpy() * 10,
    #                     all_cw.cpu().numpy(),s=0.5)
    # ax.plot([xmin, xmax],
    #         [m * xmin + b, m * xmax + b],'r', 
    #         label="{:.2f} x + {:.2f}\n r2 = {:.2f}\n MSE: {:.2f}".format(m,b,r2,mse))
    # xlim = ax.get_xlim()
    # ylim = ax.get_ylim()
    # ax.plot([min(xlim[0],ylim[0]), max(xlim[1],ylim[1])],
    #                 [min(xlim[0],ylim[0]), max(xlim[1],ylim[1]), ],'k--')
    # ax.legend()
    # ax.set_ylabel(f"Predicted Cw")
    # ax.set_xlabel(f"SNAP Cw")
    # ax.set_aspect('equal')
    # fig.savefig(f"{plot_dir}/all_cw_x10_scatter_true_vs_pred.png")

    # fig, ax = plt.subplots(1, tight_layout=True, dpi=150)
    # m, b = np.polyfit(all_weiss_cw.cpu().numpy(), all_cw_rel.cpu().numpy(), 1)
    # r2 = r2_score(all_weiss_cw.cpu().numpy(), all_cw_rel.cpu().numpy())
    # mse = (all_weiss_cw - all_cw_rel).pow(2).mean().cpu().numpy()
    # xmin = min(all_cw_rel.cpu().min().item(), all_weiss_cw.cpu().min().item())
    # xmax = max(all_cw_rel.cpu().max().item(), all_weiss_cw.cpu().max().item())
    # ax.scatter(all_weiss_cw.cpu().numpy(),
    #                     all_cw_rel.cpu().numpy(),s=0.5)
    # ax.plot([xmin, xmax],
    #         [m * xmin + b, m * xmax + b],'r', 
    #         label="{:.2f} x + {:.2f}\n r2 = {:.2f}\n MSE: {:.2f}".format(m,b,r2,mse))
    # xlim = ax.get_xlim()
    # ylim = ax.get_ylim()
    # ax.plot([min(xlim[0],ylim[0]), max(xlim[1],ylim[1])],
    #                 [min(xlim[0], ylim[0]), max(xlim[1],ylim[1]), ],'k--')
    # ax.legend()
    # ax.set_ylabel(f"Predicted CwRel")
    # ax.set_xlabel(f"SNAP Cw")
    # ax.set_aspect('equal')
    # fig.savefig(f"{plot_dir}/all_cwrel_scatter_true_vs_pred.png")
    return

def PROSAIL_2D_res_plots(plot_dir, sim_image, cropped_image, rec_image, weiss_lai, weiss_cab,
                         weiss_cw, sigma_image, i, info=None):
    if info is None:
        info = ["SENSOR","DATE","TILE"]
    n_cols = 4
    n_rows = 3
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(2*n_cols,n_rows*2), tight_layout=True, dpi=150)
    for idx in range(len(PROSAILVARS)):
        row = idx // n_cols
        col = idx % n_cols
        ax[row, col].hist(sim_image[idx,:,:].reshape(-1).cpu(), bins=50, density=True)
        ax[row, col].set_yticks([])
        ax[row, col].set_ylabel(PROSAILVARS[idx])
        ax[row, col].set_xlim(ProsailVarsDist.Dists[PROSAILVARS[idx]]['min'], 
                              ProsailVarsDist.Dists[PROSAILVARS[idx]]['max'])
    fig.delaxes(ax[-1, -1])
    fig.suptitle(f"PROSAIL variables distributions {info[1]} {info[2]}")
    fig.savefig(f"{plot_dir}/{i}_{info[1]}_{info[2]}_prosail_var_pred_dist.png")

    n_cols = 5
    n_rows = 2
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(2*n_cols,n_rows*2), tight_layout=True, dpi=150)
    for idx in range(len(BANDS)):
        row = idx // n_cols
        col = idx % n_cols
        ax[row, col].scatter(cropped_image[idx,:,:].reshape(-1).cpu(),
                            rec_image[idx,:,:].reshape(-1).cpu(), s=1)
        xlim = ax[row, col].get_xlim()
        ylim = ax[row, col].get_ylim()
        ax[row, col].plot([min(xlim[0],ylim[0]), max(xlim[1],ylim[1])],
                        [min(xlim[0],ylim[0]), max(xlim[1],ylim[1]), ],'k--')
        ax[row, col].set_yticks([])
        ax[row, col].set_ylabel(f"Reconstructed {BANDS[idx]}")
        ax[row, col].set_xlabel(f"True {BANDS[idx]}")
        ax[row, col].set_aspect('equal')
        fig.suptitle(f"Scatter plot S2 bands{info[1]} {info[2]}")
    fig.savefig(f'{plot_dir}/{i}_{info[1]}_{info[2]}_bands_scatter_true_vs_pred.png')

    n_cols = 5
    n_rows = 2
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(2*n_cols,n_rows*2), tight_layout=True)

    for idx in range(len(BANDS)):
        row = idx // n_cols
        col = idx % n_cols
        ax[row, col].hist(cropped_image[idx,:,:].reshape(-1).cpu(), bins=50, density=True)
        ax[row, col].hist(rec_image[idx,:,:].reshape(-1).cpu(), bins=50, alpha=0.5, density=True)
        ax[row, col].set_yticks([])
        ax[row, col].set_ylabel(BANDS[idx])
    fig.suptitle(f"Histogram S2 bands{info[1]} {info[2]}")
    fig.savefig(f'{plot_dir}/{i}_{info[1]}_{info[2]}_bands_hist_true_vs_pred.png')

    fig, _ = plot_patches((cropped_image.cpu(), rec_image.cpu(),
            (cropped_image[:10,...].cpu() - rec_image.cpu()).abs().mean(0).unsqueeze(0)),
            title_list=[f'original patch \n {info[1]} {info[2]}', 'reconstruction', 'mean absolute\n reconstruction error'])
    fig.savefig(f"{plot_dir}/{i}_{info[1]}_{info[2]}_patch_rec_rgb.png")

    fig, _ = plot_patches((cropped_image[torch.tensor([8,3,6]),...].cpu(),
                            rec_image[torch.tensor([8,3,6]),...].cpu()),
                            title_list=[f'original patch RGB:B8-B5-B11 \n {info[1]} {info[2]}', 'reconstruction'])
    fig.savefig(f"{plot_dir}/{i}_{info[1]}_{info[2]}_patch_rec_B8B5B11.png")

    vmin = min(sim_image[6,...].unsqueeze(0).cpu().min().item(), weiss_lai.unsqueeze(0).cpu().min().item())
    vmax = max(sim_image[6,...].unsqueeze(0).cpu().max().item(), weiss_lai.unsqueeze(0).cpu().max().item())
    fig, _ = plot_patches((cropped_image.cpu(), sim_image[6,...].unsqueeze(0).cpu(), weiss_lai.cpu()),
                            title_list=[f'original patch \n {info[1]} {info[2]}',
                                        'PROSAIL-VAE LAI', 'SNAP LAI'], 
                                        vmin=vmin, vmax=vmax)
    fig.savefig(f'{plot_dir}/{i}_{info[1]}_{info[2]}_LAI_prediction_vs_weiss.png')

    for i, varname in enumerate(PROSAILVARS):
        fig, _ = plot_patches((cropped_image.cpu(), sim_image[i,...].unsqueeze(0).cpu(),
                                                    sigma_image[i,...].unsqueeze(0).cpu()),
                                title_list=[f'original patch \n {info[1]} {info[2]}',
                                            f'PROSAIL-VAE {varname}',
                                            f"{varname} sigma"])
        fig.savefig(f'{plot_dir}/{i}_{info[1]}_{info[2]}_{varname}.png')

    ccc = sim_image[1,...] * sim_image[6,...]
    vmin = min(ccc.unsqueeze(0).cpu().min().item(), weiss_cab.unsqueeze(0).cpu().min().item())
    vmax = max(ccc.unsqueeze(0).cpu().max().item(), weiss_cab.unsqueeze(0).cpu().max().item())
    fig, _ = plot_patches((cropped_image.cpu(), ccc.unsqueeze(0).cpu(), weiss_cab.cpu()),
                            title_list=[f'original patch \n {info[1]} {info[2]}',
                                        'PROSAIL-VAE CCC', 'SNAP CCC'], 
                                        vmin=vmin, vmax=vmax)
    fig.savefig(f'{plot_dir}/{i}_{info[1]}_{info[2]}_CCC_prediction_vs_weiss.png')
    
    fig, _ = plot_patches((cropped_image.cpu(), ccc.unsqueeze(0).cpu() - weiss_cab.cpu()),
                            title_list=[f'original patch \n {info[1]} {info[2]}',
                                        'PROSAIL-VAE / SNAP CCC difference'], 
                                        vmin=None, vmax=None)
    fig.savefig(f'{plot_dir}/{i}_{info[1]}_{info[2]}_CCC_err_prediction_vs_weiss.png')

    cwc = sim_image[4,...] * sim_image[6,...]
    vmin = min(cwc.unsqueeze(0).cpu().min().item(), weiss_cw.unsqueeze(0).cpu().min().item())
    vmax = max(cwc.unsqueeze(0).cpu().max().item(), weiss_cw.unsqueeze(0).cpu().max().item())
    fig, _ = plot_patches((cropped_image.cpu(), cwc.unsqueeze(0).cpu(), weiss_cw.cpu()),
                            title_list=[f'original patch \n {info[1]} {info[2]}',
                                        'PROSAIL-VAE CWC', 'SNAP CWC'],
                                        vmin=vmin, vmax=vmax)
    fig.savefig(f'{plot_dir}/{i}_{info[1]}_{info[2]}_Cw_prediction_vs_weiss.png')

    cwc = sim_image[4,...] * sim_image[6,...]
    fig, _ = plot_patches((cropped_image.cpu(), cwc.unsqueeze(0).cpu() - weiss_cw.cpu()),
                            title_list=[f'original patch \n {info[1]} {info[2]}',
                                        'PROSAIL-VAE / SNAP \n CWC difference'],
                                        vmin=None, vmax=None)
    fig.savefig(f'{plot_dir}/{i}_{info[1]}_{info[2]}_CWC_err_prediction_vs_weiss.png')
    return

def plot_silvia_validation_patch(gdf, 
                                 pred_at_patch: np.ndarray, 
                                 pred_at_site: np.ndarray, 
                                 variable:str="lai"):
    df_sns_plot = pd.DataFrame({variable: gdf[variable].values.reshape(-1),
                                f"Predicted {variable}": pred_at_site,
                                "Land Cover": gdf["land cover"],
                                "x": gdf["x_idx"],
                                "y": gdf["y_idx"],
                                })
    fig, ax = plt.subplots()
    im = ax.imshow(pred_at_patch.squeeze())
    plt.colorbar(im)
    sns.scatterplot(data=df_sns_plot, x='x', y="y", hue="Land Cover", ax=ax)
    ax.set_xlabel('')
    ax.set_ylabel('')
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    return fig, ax

def patch_validation_reg_scatter_plot(gdf, patch_pred:np.ndarray,
                                      variable:str='lai',
                                      fig=None, ax=None, legend=True):
    x_idx = gdf["x_idx"].values.astype(int)
    y_idx = gdf["y_idx"].values.astype(int)
    ref = gdf[variable].values.reshape(-1)
    ref_uncert = gdf["uncertainty"].values
    pred_at_site = patch_pred[:, y_idx, x_idx].reshape(-1)
    df = pd.DataFrame({variable:ref,
                       f"Predicted {variable}": pred_at_site,
                       "Land Cover": gdf["land cover"]})
    if fig is None or ax is None:
        fig, ax = plt.subplots(dpi=150, figsize=(6,6))
    xmin = min(np.min(pred_at_site), np.min(ref))
    xmax = max(np.max(pred_at_site), np.max(ref))
    ax.plot([xmin, xmax], [xmin, xmax], '--k')
    m, b = np.polyfit(ref, pred_at_site, 1)
    r2 = r2_score(ref, pred_at_site)
    rmse = np.sqrt(np.mean((ref - pred_at_site)**2))
    perf_text = " y = {:.2f} x + {:.2f} \n r2: {:.2f} \n RMSE: {:.2f}".format(m,b,r2,rmse)
    ax.text(.05, .95, perf_text, ha='left', va='top', transform=ax.transAxes)
    line = ax.plot([xmin, xmax], [m * xmin + b, m * xmax + b],'r')
    
    g = sns.scatterplot(data=df, x=variable, y=f"Predicted {variable}",
                        hue="Land Cover", ax=ax)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(xmin, xmax)
    ax.set_aspect('equal', 'box')
    if not legend:
        ax.get_legend().remove()
    else:
        sns.move_legend(g, "upper center", bbox_to_anchor=(0.5, -0.1),ncol=len(pd.unique(gdf["land cover"]))//2,
                        frameon=True)
        fig.tight_layout()
    return fig, ax, g

def silvia_validation_plots(lai_pred, ccc_pred, data_dir, filename, res_dir=None):
    if isinstance(lai_pred, torch.Tensor):
        lai_pred = lai_pred.numpy()
    if isinstance(ccc_pred, torch.Tensor):
        ccc_pred = ccc_pred.numpy()
    gdf_lai, _, _ = load_validation_data(data_dir, filename, variable="lai")
    fig, ax, g = patch_validation_reg_scatter_plot(gdf_lai, patch_pred=lai_pred,
                                                variable='lai', fig=None, ax=None)
    if res_dir is not None:
        fig.savefig(os.path.join(res_dir, f"{filename}_scatter_lai.png"))

    lai_pred_at_site = lai_pred[:, gdf_lai["y_idx"].values.astype(int), 
                                gdf_lai["x_idx"].values.astype(int)].reshape(-1)
    fig, ax = plot_silvia_validation_patch(gdf_lai, lai_pred, lai_pred_at_site)
    if res_dir is not None:
        fig.savefig(os.path.join(res_dir, f"{filename}_field_lai.png"))

    gdf_lai_eff, _, _ = load_validation_data(data_dir, filename, 
                                                   variable="lai_eff")
    gdf_lai_eff = gdf_lai_eff.iloc[:51]
    fig, ax, g = patch_validation_reg_scatter_plot(gdf_lai_eff, patch_pred=lai_pred,
                                                variable='lai_eff', fig=None, ax=None)
    if res_dir is not None:
        fig.savefig(os.path.join(res_dir, f"{filename}_scatter_lai_eff.png"))

    gdf_ccc, _, _ = load_validation_data(data_dir, filename, variable="ccc")
    fig, ax, g = patch_validation_reg_scatter_plot(gdf_ccc, patch_pred=ccc_pred,
                                      variable='ccc', fig=None, ax=None)
    if res_dir is not None:
        fig.savefig(os.path.join(res_dir, f"{filename}_scatter_ccc.png"))

    gdf_ccc_eff, _, _ = load_validation_data(data_dir, filename, variable="ccc_eff")
    fig, ax, g = patch_validation_reg_scatter_plot(gdf_ccc_eff, patch_pred=ccc_pred,
                                      variable='ccc_eff', fig=None, ax=None)
    if res_dir is not None:
        fig.savefig(os.path.join(res_dir, f"{filename}_scatter_ccc_eff.png"))
    return
