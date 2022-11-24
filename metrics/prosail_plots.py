#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 11:46:20 2022

@author: yoel
"""
import matplotlib.pyplot as plt
import torch
import numpy as np
from prosailvae.ProsailSimus import PROSAILVARS, ProsailVarsDist

def plot_metrics(res_dir, alpha_pi, maer, mpiwr, picp, mare):
    fig = plt.figure(dpi=200)
    
    for i in range(len(maer.columns)):
        plt.plot(alpha_pi, picp[i,:].detach().cpu().numpy(), label=maer.columns[i])
    plt.legend()
    plt.xlabel('1-a')
    plt.ylabel('PICP')
    fig.savefig(res_dir+"/picp.svg")
    
    fig = plt.figure(dpi=200)
    for i in range(len(maer.columns)):
        plt.plot(alpha_pi, mpiwr.values[:,i], label=maer.columns[i])
    plt.legend()
    plt.xlabel('1-a')
    plt.ylabel('MPIWr')
    fig.savefig(res_dir+"/MPIWr.svg")
    
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.bar(maer.columns, maer.values.reshape(-1),)
    plt.ylabel('MAEr')
    fig.savefig(res_dir+"/MAEr.svg")
    
    fig = plt.figure(dpi=150)
    ax = fig.add_axes([0,0,1,1])
    ax.bar(maer.columns, mare.detach().cpu().numpy())
    plt.ylabel('MARE')
    plt.yscale('log')
    fig.savefig(res_dir+"/mare.svg")

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
        
        bands = ["b2", "b3", "b4", "b5", "b6", "b7", "b8", "b8a", "b11", "b12"]
        
        rec_samples = [rec_samples[j,:] for j in range(len(bands))]
        sim_samples = sim.squeeze().detach().cpu().numpy()
        sim_samples = [sim_samples[j,:] for j in range(len(PROSAILVARS))]
        
        ind1 = np.arange(len(bands))
        width = 0.35
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
        ax1.set_yticklabels(bands)
        ax1.xaxis.grid(True)
        
        # ind2 = np.arange(len(PROSAILVARS))
        width = 0.35
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
    prosail_VAE.decoder.ssimulator.apply_norm = original_prosail_s2_norm
    
def loss_curve(loss_df, save_file, log_scale=False):
    loss_names = loss_df.columns.values.tolist()
    loss_names.remove("epoch")
    epochs = loss_df["epoch"]
    fig, ax = plt.subplots(dpi=150)
    for i in range(len(loss_names)):
        loss = loss_df[loss_names[i]].values
        if (loss<=0).any() or log_scale:
            loss += loss.min() + 1
            ax.set_yscale('log')
        ax.plot(epochs,loss, label=loss_names[i])
    ax.legend()
    ax.set_xticks(epochs)
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    fig.savefig(save_file)