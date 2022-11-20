#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 12:55:03 2022

@author: yoel
"""
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import os 
from prosailvae.ProsailSimus import PROSAILVARS, get_ProsailVarsIntervalLen

def save_metrics(res_dir, mae, mpiw, picp, alpha_pi):
    metrics_dir = res_dir + "/metrics/"
    os.makedirs(metrics_dir)
    pd.DataFrame(data=mae.view(1,len(PROSAILVARS)).detach().cpu().numpy(), columns=PROSAILVARS, 
                 index=[0]).to_csv(metrics_dir + "/mae.csv")
    df_mpwi = pd.DataFrame(data=mpiw.view(-1, len(PROSAILVARS)).detach().cpu().numpy(), 
                           columns=PROSAILVARS)
    df_mpwi["alpha"] = alpha_pi
    df_mpwi.to_csv(metrics_dir + "/mpiw.csv")
    df_picp = pd.DataFrame(data=picp.view(-1, len(PROSAILVARS)).detach().cpu().numpy(), 
                           columns=PROSAILVARS)
    df_picp["alpha"] = alpha_pi
    df_picp.to_csv(metrics_dir + "/picp.csv")
    
    il = get_ProsailVarsIntervalLen().to(mpiw.device)
    mpiwr = (mpiw/il.view(-1,1)).transpose(0,1)
    maer = mae/il
    df_maer = pd.DataFrame(data=maer.view(-1, len(PROSAILVARS)).detach().cpu().numpy(), 
                           columns=PROSAILVARS)
    df_mpiwr = pd.DataFrame(data=mpiwr.view(-1, len(PROSAILVARS)).detach().cpu().numpy(), 
                           columns=PROSAILVARS)
    df_maer.to_csv(metrics_dir + "/maer.csv")
    df_mpiwr.to_csv(metrics_dir + "/mpiwr.csv")

def get_metrics(prosailVAE, loader,  
                n_pdf_sample_points=3001,
                alpha_conf=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]):
    
    device = prosailVAE.device
    error = torch.tensor([]).to(device)
    rel_error = torch.tensor([]).to(device)
    pic = torch.tensor([]).to(device)
    piw = torch.tensor([]).to(device)
    
    pi_lower = (np.array(alpha_conf)/2).tolist()
    pi_upper = (1-np.array(alpha_conf)/2).tolist()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc='Computing metrics', leave=True)):
            data = batch[0].to(device)
            angles = batch[1].to(device)
            tgt = batch[2].to(device)
            dist_params, z_mode, pheno_mode, _ = prosailVAE.point_estimate_rec(data, angles, mode='sim_mode')
            lat_pdfs, lat_supports = prosailVAE.lat_space.latent_pdf(dist_params)
            pheno_pdfs, pheno_supports = prosailVAE.sim_space.sim_pdf(lat_pdfs, lat_supports, n_pdf_sample_points=n_pdf_sample_points)
            pheno_pi_lower = prosailVAE.sim_space.sim_quantiles(lat_pdfs, lat_supports, alpha=pi_lower, n_pdf_sample_points=n_pdf_sample_points)
            pheno_pi_upper = prosailVAE.sim_space.sim_quantiles(lat_pdfs, lat_supports, alpha=pi_upper, n_pdf_sample_points=n_pdf_sample_points)
            error_i = pheno_mode.squeeze() - tgt
            error = torch.concat([error, error_i], axis=0)
            rel_error_i = (pheno_mode.squeeze() - tgt).abs() / (tgt.abs()+1e-10)
            rel_error = torch.concat([rel_error, rel_error_i], axis=0)
            piw_i = pheno_pi_upper - pheno_pi_lower
            piw = torch.concat([piw, piw_i], axis=0)
            pic_i = torch.logical_and(tgt.unsqueeze(2) > pheno_pi_lower, 
                                      tgt.unsqueeze(2) < pheno_pi_upper).float()
            pic = torch.concat([pic, pic_i], axis=0)
    mae = error.abs().mean(axis=0)     
    picp = pic.mean(axis=0)    
    mpiw = piw.mean(axis=0)
    mare = rel_error.mean(axis=0)
    return mae, mpiw, picp, mare