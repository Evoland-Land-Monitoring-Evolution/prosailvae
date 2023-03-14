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
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from dataset.juan_datapoints import get_interpolated_validation_data
from dataset.validate_prosail_weiss import load_weiss_dataset
import prosailvae

def save_metrics(res_dir, mae, mpiw, picp, alpha_pi, ae_percentiles, are_percentiles, piw_percentiles):
    metrics_dir = res_dir + "/metrics/"
    if not os.path.isdir(metrics_dir):
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
    
    interval_length = get_ProsailVarsIntervalLen().to(mpiw.device)
    mpiwr = (mpiw / interval_length.view(-1,1)).transpose(0,1)
    maer = mae / interval_length
    df_maer = pd.DataFrame(data=maer.view(-1, len(PROSAILVARS)).detach().cpu().numpy(), 
                           columns=PROSAILVARS)
    df_mpiwr = pd.DataFrame(data=mpiwr.view(-1, len(PROSAILVARS)).detach().cpu().numpy(), 
                           columns=PROSAILVARS)
    df_maer.to_csv(metrics_dir + "/maer.csv")
    df_mpiwr.to_csv(metrics_dir + "/mpiwr.csv")
    torch.save(ae_percentiles, metrics_dir + '/ae_percentiles.pt')
    aer_percentiles = ae_percentiles / interval_length.view(1,-1).detach().cpu().numpy()
    torch.save(aer_percentiles, metrics_dir + '/aer_percentiles.pt')
    torch.save(are_percentiles, metrics_dir + '/are_percentiles.pt')
    # torch.save(piw_percentiles, '/piw_percentiles.pt')

def get_percentiles_from_box_plots(bp):
    percentiles = torch.zeros((5,len(bp['boxes'])))
    for i in range(len(bp['boxes'])):
        percentiles[0,i] = torch.from_numpy(np.asarray(bp['caps'][2*i].get_ydata()[0]))
        percentiles[1,i] = torch.from_numpy(np.asarray(bp['boxes'][i].get_ydata()[0]))
        percentiles[2,i] = torch.from_numpy(np.asarray(bp['medians'][i].get_ydata()[0]))
        percentiles[3,i] = torch.from_numpy(np.asarray(bp['boxes'][i].get_ydata()[2]))
        percentiles[4,i] = torch.from_numpy(np.asarray(bp['caps'][2*i + 1].get_ydata()[0]))
                        #    (bp['fliers'][i].get_xdata(),
                        #     bp['fliers'][i].get_ydata()))
    return percentiles

def get_box_plot_percentiles(tensor):
    fig, ax = plt.subplots()
    all_tensor_percentiles = torch.zeros((5, tensor.size(1)))
    for i in range(tensor.size(1)):
        bp = ax.boxplot([tensor[:,i].numpy(),])
        percentiles = get_percentiles_from_box_plots(bp)
        all_tensor_percentiles[:,i] = percentiles.squeeze()
    return all_tensor_percentiles

def get_metrics(PROSAIL_VAE, loader,  
                n_pdf_sample_points=3001,
                alpha_conf=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]):
    
    
    device = PROSAIL_VAE.device
    error = torch.tensor([]).to(device)
    rel_error = torch.tensor([]).to(device)
    pic = torch.tensor([]).to(device)
    piw = torch.tensor([]).to(device)
    sim_dist = torch.tensor([]).to(device)
    pi_lower = (np.array(alpha_conf)/2).tolist()
    pi_upper = (1-np.array(alpha_conf)/2).tolist()
    tgt_dist = torch.tensor([]).to(device)
    rec_dist = torch.tensor([]).to(device)
    s2_r_dist = torch.tensor([]).to(device)
    angles_dist = torch.tensor([]).to(device)
    sim_pdfs = torch.tensor([]).to(device)
    sim_supports = torch.tensor([]).to(device)
    ssimulator = PROSAIL_VAE.decoder.ssimulator
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc='Computing metrics', leave=True)):
            s2_r = batch[0].to(device)
            s2_r_dist = torch.concat([s2_r_dist, s2_r], axis=0)
            angles = batch[1].to(device)
            tgt = batch[2].to(device)
            dist_params, z_mode, prosail_params_mode, rec = PROSAIL_VAE.point_estimate_rec(s2_r, angles, mode='sim_mode')
            lat_pdfs, lat_supports = PROSAIL_VAE.lat_space.latent_pdf(dist_params)
            sim_pdfs_i, sim_supports_i = PROSAIL_VAE.sim_space.sim_pdf(lat_pdfs, lat_supports, n_pdf_sample_points=n_pdf_sample_points)
            sim_pdfs = torch.concat([sim_pdfs, sim_pdfs_i], axis=0)
            sim_supports = torch.concat([sim_supports, sim_supports_i], axis=0)
            pheno_pi_lower = PROSAIL_VAE.sim_space.sim_quantiles(lat_pdfs, lat_supports, alpha=pi_lower, n_pdf_sample_points=n_pdf_sample_points)
            pheno_pi_upper = PROSAIL_VAE.sim_space.sim_quantiles(lat_pdfs, lat_supports, alpha=pi_upper, n_pdf_sample_points=n_pdf_sample_points)
            error_i = prosail_params_mode.squeeze() - tgt
            tgt_dist = torch.concat([tgt_dist, tgt], axis=0)
            error = torch.concat([error, error_i], axis=0)
            sim_dist = torch.concat([sim_dist, prosail_params_mode], axis=0)
            rec_dist = torch.concat([rec_dist, rec.squeeze() * ssimulator.norm_std + ssimulator.norm_mean], axis=0)
            rel_error_i = (prosail_params_mode.squeeze() - tgt).abs() / (tgt.abs()+1e-10)
            rel_error = torch.concat([rel_error, rel_error_i], axis=0)
            piw_i = pheno_pi_upper - pheno_pi_lower
            piw = torch.concat([piw, piw_i], axis=0)
            pic_i = torch.logical_and(tgt.unsqueeze(2) > pheno_pi_lower, 
                                      tgt.unsqueeze(2) < pheno_pi_upper).float()
            pic = torch.concat([pic, pic_i], axis=0)
            angles_dist = torch.concat([angles_dist, angles], axis=0)
    mae = error.abs().mean(axis=0)     
    ae_percentiles = get_box_plot_percentiles(error.abs().detach().cpu())
    picp = pic.mean(axis=0)    
    mpiw = piw.mean(axis=0)
    piw_percentiles = None # get_box_plot_percentiles(piw.detach().cpu())
    mare = rel_error.mean(axis=0)
    are_percentiles = get_box_plot_percentiles(rel_error.detach().cpu())

    return (mae, mpiw, picp, mare, sim_dist, tgt_dist, rec_dist, angles_dist, s2_r_dist, sim_pdfs, 
            sim_supports, ae_percentiles, are_percentiles, piw_percentiles)

def get_juan_validation_metrics(PROSAIL_VAE, juan_data_dir_path, lai_min=0, dt_max=10, 
                                sites = ["france", "spain1", "italy1", "italy2"], weiss_mode=False):
    list_lai_nlls = []
    list_lai_preds = []
    dt_list = []
    for site in sites:
        s2_r, s2_a, lais, dt = get_interpolated_validation_data(site, juan_data_dir_path, lai_min=lai_min, 
                                                                dt_max=dt_max, method="closest")
        if weiss_mode:
            s2_r = s2_r[:, torch.tensor([1,2,3,4,5,7,8,9])]
        prosail_ref_params = torch.zeros((s2_r.size(0), 11))
        prosail_ref_params[:,6] = lais.squeeze()
        juan_dataset = TensorDataset(s2_r.to(PROSAIL_VAE.device), 
                                     s2_a.to(PROSAIL_VAE.device), 
                                     prosail_ref_params.to(PROSAIL_VAE.device))
        juan_loader = DataLoader(juan_dataset,
                                batch_size=256,
                                num_workers=0)
        lai_nlls = PROSAIL_VAE.compute_lat_nlls(juan_loader).mean(0).squeeze()[6].cpu()
        list_lai_nlls.append(lai_nlls)
        _, _, prosail_params_mode, _ = PROSAIL_VAE.point_estimate_rec(s2_r.to(PROSAIL_VAE.device), s2_a.to(PROSAIL_VAE.device), mode='sim_mode')
        lai_pred = prosail_params_mode[:,6,:].cpu()
        list_lai_preds.append(torch.cat((lai_pred, lais), axis=1))
        dt_list.append(dt)

    return list_lai_nlls, list_lai_preds, dt_list

def load_weiss_data_from_txt(weiss_data_dir_path, device='cpu'):
    s2_r, prosail_vars = load_weiss_dataset(weiss_data_dir_path)
    s2_a = prosail_vars[:,-3:]
    prosail_ref_params = torch.as_tensor(prosail_vars[:,:11]).float().to(device)
    s2_r = torch.as_tensor(s2_r).float().to(device)
    s2_a = torch.as_tensor(s2_a).float().to(device) 
    return s2_r, s2_a, prosail_ref_params


def get_weiss_validation_metrics(PROSAIL_VAE, s2_r, s2_a, prosail_ref_params, n_pdf_sample_points=5001):
    with torch.no_grad():
        lais = torch.as_tensor(prosail_ref_params[:,6]).float().cpu().view(-1,1)
        weiss_dataset = TensorDataset(s2_r, s2_a, prosail_ref_params)
        weiss_loader = DataLoader(weiss_dataset, batch_size=512, num_workers=0)
        lai_nlls = PROSAIL_VAE.compute_lat_nlls(weiss_loader).mean(0).squeeze()[6].cpu()
        lai_pred = []
        sim_pdfs = torch.tensor([]).to(PROSAIL_VAE.device)
        sim_supports = torch.tensor([]).to(PROSAIL_VAE.device)
        for i, b in enumerate(weiss_loader):
            s2_r = b[0]
            s2_a = b[1]
            y = PROSAIL_VAE.encode(s2_r, s2_a)
            dist_params = PROSAIL_VAE.lat_space.get_params_from_encoder(y)
            lat_pdfs, lat_supports = PROSAIL_VAE.lat_space.latent_pdf(dist_params)
            prosail_params_mode = PROSAIL_VAE.sim_space.sim_mode(lat_pdfs, lat_supports, n_pdf_sample_points=n_pdf_sample_points)
            sim_pdfs_i, sim_supports_i = PROSAIL_VAE.sim_space.sim_pdf(lat_pdfs, lat_supports, n_pdf_sample_points=n_pdf_sample_points)
            sim_pdfs = torch.concat([sim_pdfs, sim_pdfs_i], axis=0)
            sim_supports = torch.concat([sim_supports, sim_supports_i], axis=0)
            lai = prosail_params_mode[:,6,:].cpu()
            lai_pred.append(lai)
        lai_pred = torch.cat(lai_pred, axis=0)
        lai_pred = torch.cat((lai_pred, lais), axis=1)

    return lai_nlls, lai_pred, sim_pdfs, sim_supports
