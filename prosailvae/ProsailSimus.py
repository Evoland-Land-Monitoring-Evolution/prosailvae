#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 13:39:15 2022

@author: yoel
"""
from typing import Tuple
from dataclasses import dataclass
import numpy as np
import torch
import prosail
from prosail import spectral_lib
from prosailvae.utils import gaussian_nll_loss, torch_select_unsqueeze
from prosailvae.spectral_indices import INDEX_DICT

PROSAILVARS = ["N", "cab", "car", "cbrown", "caw", "cm", 
               "lai", "lidfa", "hspot", "psoil", "rsoil"]
BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]

class SensorSimulator():
    """Simulates the reflectances of a sensor from a full spectrum and the
RSR of the sensor.

    The rsr_file is supposed to follow the 6S format: lambdas are given
    in micrometers and regularly sampled with a 0.001 step (1 nm). The
    input full spectrum can be a Prosail simulation and will be sampled
    with the same step. The wavelength ranges of the RSR are typically
    [0.350,2400] and Prosail gives [400, 2500]. The output will be given
    in [400, 2500] as the input.

    The RSR file columns are: lambda, solar spectrum and bands.
    """
    def __init__(self,
                 rsr_file: str,
                 prospect_range: Tuple[int, int] = (400, 2500),
                #  bands=[1, 2, 3, 4, 5, 6, 7, 8, 9, 11],
                 bands=[1, 2, 3, 4, 5, 6, 7, 8, 11, 12],
                 device='cpu',
                 norm_mean=None,
                 norm_std=None,
                 apply_norm=True):
          
        super().__init__()
        self.bands=bands
        self.device=device
        self.prospect_range = prospect_range
        self.rsr = torch.from_numpy(np.loadtxt(rsr_file, unpack=True)).to(device)
        self.nb_bands = self.rsr.shape[0] - 2
        self.rsr_range = (int(self.rsr[0, 0].item() * 1000),
                          int(self.rsr[0, -1].item() * 1000))
        self.nb_lambdas = prospect_range[1] - prospect_range[0] + 1
        self.rsr_prospect = torch.zeros([self.rsr.shape[0], self.nb_lambdas]).to(device)
        self.rsr_prospect[0, :] = torch.linspace(prospect_range[0],
                                                 prospect_range[1],
                                                 self.nb_lambdas).to(device)
        self.rsr_prospect[1:, :-(self.prospect_range[1] -
                                 self.rsr_range[1])] = self.rsr[1:, (
                                     self.prospect_range[0] -
                                     self.rsr_range[0]):]

        self.solar = self.rsr_prospect[1, :].unsqueeze(0)
        self.rsr = self.rsr_prospect[2:, :].unsqueeze(0)
        self.rsr = self.rsr[:,bands,:]
        
        if norm_mean is None:
            norm_mean = torch.zeros((1, len(bands)))
        else:
            if len(norm_mean.squeeze()) != len(bands):
                print(norm_mean.size())
                print(norm_mean)
                raise ValueError
        if norm_std is None:
            norm_std = torch.ones((1, len(bands)))
        else:
            if len(norm_std.squeeze()) != len(bands):
                print(norm_std.size())
                print(norm_std)
                raise ValueError
                
        self.norm_mean = norm_mean.float().to(device)
        self.norm_std = norm_std.float().to(device)
        self.apply_norm = apply_norm
        self.s2norm_factor_d = (self.rsr * self.solar).sum(axis=2)
        self.s2norm_factor_n = self.rsr * self.solar
        
    def change_device(self, device):
        self.device = device
        self.rsr = self.rsr.to(device)
        self.norm_mean = self.norm_mean.to(device)
        self.norm_std = self.norm_std.to(device)
        self.s2norm_factor_d = self.s2norm_factor_d.to(device)
        self.s2norm_factor_n = self.s2norm_factor_n.to(device)
        self.solar = self.solar.to(device)

        pass

    def __call__(self, prosail_output: torch.Tensor, apply_norm=None):
        return self.forward(prosail_output, apply_norm=apply_norm)
    
    def apply_s2_sensor(self, prosail_output: torch.Tensor) -> torch.Tensor:
        # The input should have shape = batch, wavelengths, otherwise,
        # we add the first dimension
        x = prosail_output
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        elif len(x.shape) == 2:
            x = x.unsqueeze(1)
        simu = (self.s2norm_factor_n * x).sum(
            axis=2) / self.s2norm_factor_d  
        return simu
    
    def normalize(self, s2_r, bands_dim=1):
        dim_s2_r = len(s2_r.size())
        return (s2_r - torch_select_unsqueeze(self.norm_mean, bands_dim, dim_s2_r)) / torch_select_unsqueeze(self.norm_std, bands_dim, dim_s2_r)
    
    def unnormalize(self, s2_r, bands_dim=1):
        dim_s2_r = len(s2_r.size())
        u_s2_r = s2_r * torch_select_unsqueeze(self.norm_std, bands_dim, dim_s2_r) 
        u_s2_r = u_s2_r + torch_select_unsqueeze(self.norm_mean, bands_dim, dim_s2_r)
        return u_s2_r

    def forward(self, prosail_output: torch.Tensor, apply_norm=None) -> torch.Tensor:
        simu = self.apply_s2_sensor(prosail_output)
        if apply_norm is None:
            apply_norm = self.apply_norm
        if apply_norm:
            simu = self.normalize(simu)
        return simu  # type: ignore
    
    def index_loss(self, s2_r, s2_rec, lossfn=gaussian_nll_loss,
                    index_terms=["NDVI", "mNDVI750", "CRI2", "NDII", "ND_lma", "LAI_savi"],
                    bands_dim=1):

        u_s2_r = self.unnormalize(s2_r)
        if self.apply_norm:
            u_s2_rec = self.unnormalize(s2_rec)
        else:
            u_s2_rec = s2_rec

        loss = torch.tensor(0.0).to(s2_r.device)
        for idx in index_terms:
            spectral_idx_tgt = INDEX_DICT[idx](u_s2_r, bands_dim=bands_dim)
            spectral_idx_rec = INDEX_DICT[idx](u_s2_rec, bands_dim=bands_dim)
            loss += lossfn(spectral_idx_tgt, spectral_idx_rec) 
        return loss

class ProsailSimulator():
    def __init__(self, factor: str = "SDR", typelidf: int = 2, device='cpu'):
        super().__init__()
        self.factor = factor
        self.typelidf = typelidf
        self.device=device
    def __call__(self, params):
        return self.forward(params)

    def change_device(self, device):
        self.device=device
        pass

    def forward(self, params):
        # params.shape == [x] => single sample
        # params.shape == [x,y] => batch
        assert params.shape[-1] == 14, f"{params.shape[-1]}"
        if len(params.shape) == 1:
            params = params.unsqueeze(0)
        prosail_refl = prosail.run_prosail(
            params,
            typelidf=torch.as_tensor(self.typelidf),
            factor=self.factor,
            device=self.device
        ).float()
        
        return prosail_refl
    
def get_ProsailVarsIntervalLen():
    d = ProsailVarsDist()
    return torch.tensor([d.N[1]-d.N[0],
                         d.cab[1]-d.cab[0],
                         d.car[1]-d.car[0],
                         d.cbrown[1]-d.cbrown[0],
                         d.caw[1]-d.caw[0],
                         d.cm[1]-d.cm[0],
                         d.lai[1]-d.N[0],
                         d.lidfa[1]-d.lidfa[0],
                         d.hspot[1]-d.hspot[0],
                         d.psoil[1]-d.psoil[0],
                         d.rsoil[1]-d.rsoil[0],
                         ])

@dataclass(frozen=True)
class ProsailVarsBounds:
    """Values from [AD7]VALSE2-TN-012-CCRS-LAI-v2.0.pdf and
    [AD10]S2PAD-VEGA-ATBD-0003-2_1_L2B_ATBD
        Each tuple is gives the range (min, max) for each variable.
        """
    #   N         | Leaf structure parameter
    N: Tuple[float, float] = (1.0, 3.0)
    #  cab        | Chlorophyll a+b concentration
    cab: Tuple[float, float] = (10.0, 100.0)
    #  car        | Carotenoid concentration
    car: Tuple[float, float] = (0.0, 2.0)
    #  cbrown     | Brown pigment => Cbp
    cbrown: Tuple[float, float] = (0.0, 2.0) 
    #  caw        | Equivalent water thickiness
    caw: Tuple[float, float] = (0.004, 0.04)
    #  cm         | Dry matter content (in g/cm2, while ATBD uses g/m2)
    cm: Tuple[float, float] = (20.0 / 10000, 165.0 / 10000)
    #  lai        | Leaf Area Index
    lai: Tuple[float, float] = (0.0, 10.0)
    #  lidfa      | Leaf angle distribution => ALA ?
    lidfa: Tuple[float, float] = (0.0, 90.0)
    #  hspot      | Hotspot parameter => HsD
    hspot: Tuple[float, float] = (0.0, 0.05) 
    #  psoil      | Dry/Wet soil factor
    psoil: Tuple[float, float] = (0.0, 1.0)
    #  rsoil      | Soil brigthness factor
    rsoil: Tuple[float, float] = (0.3, 3.5)
    
@dataclass(frozen=True)
class ProsailVarsDist:
    """Values from [AD7]VALSE2-TN-012-CCRS-LAI-v2.0.pdf and
    [AD10]S2PAD-VEGA-ATBD-0003-2_1_L2B_ATBD
        Each tuple is gives the range (min, max) for each variable.
        """
    sentinel2_max_tto = np.rad2deg(np.arctan(145/786))
    solar_max_zenith_angle = 60
    # Dists = {'N': {'min': 1.2, 'max': 1.8, 'mean': 1.5, 'std': 0.3, 'lai_conv': 10,
    #                'law': 'gaussian'},
    #          'cab': {'min': 20.0, 'max': 90.0, 'mean': 45.0, 'std': 30.0, 'lai_conv': 10,
    #                  'law': 'gaussian'},
    #          'car': {'min': 0, 'max': 2, 'mean': 1, 'std': 0.5, 'lai_conv': 10,
    #                   'law': 'gaussian'},
    #          'cbrown': {'min': 0.0, 'max': 2.0, 'mean': 0.0, 'std': 0.3, 'lai_conv': 10,
    #                     'law': 'gaussian'},
    #          'caw': {'min': 0.004, 'max': 0.04, 'mean': 0.01, 'std': 0.01, 'lai_conv': 10,
    #                  'law': 'gaussian'},
    #          'cm': {'min': 0.003, 'max': 0.011, 'mean': 0.005, 'std': 0.005, 'lai_conv': 10,
    #                 'law': 'gaussian'},
    #         #  'lai': {'min': 0.0, 'max': 10.0, 'mean': 1.7, 'std': 0.35, 'lai_conv': None,
    #         #          'law': 'lognormal'},
    #         # 'lai': {'min': 0.0, 'max': 10.0, 'mean': 1.2, 'std': 0.7, 'lai_conv': None,
    #         #          'law': 'lognormal'},
    #         'lai': {'min': 0.0, 'max': 10.0, 'mean': 2.0, 'std': 2.0, 'lai_conv': None,
    #                  'law': 'lognormal'},
    #          'lidfa': {'min': 5.0, 'max': 80.0, 'mean': 40.0, 'std': 20.0, 'lai_conv': 10,
    #                    'law': 'gaussian'},
    #          'hspot': {'min': 0.0, 'max': 0.05, 'mean': 0.01, 'std': 0.025, 'lai_conv': 10,
    #                    'law': 'gaussian'},
    #          'psoil': {'min': 0.0, 'max': 1.0, 'mean': None, 'std': None, 'lai_conv': None,
    #                    'law': 'uniform'},
    #          'rsoil': {'min': 0.3, 'max': 3.5, 'mean': None, 'std': None, 'lai_conv': None,
    #           'law': 'uniform'},
    #          'tts' : {'min': 0.0, 'max': solar_max_zenith_angle, 'mean': None, 'std': None, 'lai_conv': None,
    #           'law': 'uniform'},
    #          'tto' : {'min': 0.0, 'max': sentinel2_max_tto, 'mean': None, 'std': None, 'lai_conv': None,
    #           'law': 'uniform'},
    #          'psi' : {'min': 0.0, 'max': 360, 'mean': None, 'std': None, 'lai_conv': None,
    #           'law': 'uniform'}
    #          }
    Dists = {
            # 'N': {'min': 1.2, 'max': 2.2, 'mean': 1.5, 'std': 0.3, 'lai_conv': 10,
            #     'law': 'gaussian'},
            # 'N': {'min': 1.2, 'max': 1.8, 'mean': 1.5, 'std': 0.3, 'lai_conv': 10,
            #     'law': 'gaussian'},
            'N': {'min': 1.2, 'max': 1.8, 'mean': 1.3, 'std': 0.3, 'lai_conv': 10,
                'law': 'gaussian'},
            'cab': {'min': 20.0, 'max': 90.0, 'mean': 45.0, 'std': 30.0, 'lai_conv': 10,
                    'law': 'gaussian'},
            # 'car': {'min': 0, 'max': 2, 'mean': 1, 'std': 0.5, 'lai_conv': 10,
            #         'law': 'gaussian'},
            'car': {'min': 5, 'max': 23, 'mean': 11, 'std': 5, 'lai_conv': 10,
                    'law': 'gaussian'},
            'cbrown': {'min': 0.0, 'max': 2.0, 'mean': 0.0, 'std': 0.3, 'lai_conv': 10,
                    'law': 'gaussian'},
            # 'caw': {'min': 0.0075, 'max': 0.075, 'mean': 0.02, 'std': 0.02, 'lai_conv': 10,
            #         'law': 'gaussian'},
            'caw': {'min': 0.0075, 'max': 0.075, 'mean': 0.025, 'std': 0.02, 'lai_conv': 10,
            'law': 'gaussian'},
            'cm': {'min': 0.003, 'max': 0.011, 'mean': 0.005, 'std': 0.005, 'lai_conv': 10,
                'law': 'gaussian'},
            # 'lai': {'min': 0.0, 'max': 15.0, 'mean': 2.0, 'std': 2.0, 'lai_conv': None,
            #         'law': 'lognormal'},
            # 'lai': {'min': 0.0, 'max': 15.0, 'mean': 1.0, 'std': 0.6, 'lai_conv': None,
            #         'law': 'lognormal'},
            'lai': {'min': 0.0, 'max': 15.0, 'mean': 2.0, 'std': 3.0, 'lai_conv': None,
                    'law': 'gaussian'},
            'lidfa': {'min': 30.0, 'max': 80.0, 'mean': 60.0, 'std': 20.0, 'lai_conv': 10,
                    'law': 'gaussian'},
            'hspot': {'min': 0.1, 'max': 0.5, 'mean': 0.25, 'std': 0.5, 'lai_conv': None,
                    'law': 'gaussian'},
            # 'hspot': {'min': 0.001, 'max': 0.5, 'mean': 0.001, 'std': 0.1, 'lai_conv': 10,
            #         'law': 'gaussian'},
            'psoil': {'min': 0.0, 'max': 1.0, 'mean': None, 'std': None, 'lai_conv': None,
                    'law': 'uniform'},
            'rsoil': {'min': 0.3, 'max': 3.5, 'mean': None, 'std': None, 'lai_conv': None,
            'law': 'uniform'},
            'tts' : {'min': 0.0, 'max': solar_max_zenith_angle, 'mean': None, 'std': None, 'lai_conv': None,
            'law': 'uniform'},
            'tto' : {'min': 0.0, 'max': sentinel2_max_tto, 'mean': None, 'std': None, 'lai_conv': None,
            'law': 'uniform'},
            'psi' : {'min': 0.0, 'max': 360, 'mean': None, 'std': None, 'lai_conv': None,
            'law': 'uniform'}
            }
    #   N         | Leaf structure parameter
    N = (Dists['N']['min'], Dists['N']['max'], Dists['N']['mean'],
         Dists['N']['std'], Dists['N']['lai_conv'], Dists['N']['law'], "N")
    
    #  cab        | Chlorophyll a+b concentration
    cab = (Dists['cab']['min'], Dists['cab']['max'], Dists['cab']['mean'],
           Dists['cab']['std'], Dists['cab']['lai_conv'], Dists['cab']['law'], "cab")
    #  car        | Carotenoid concentration
    car = (Dists['car']['min'], Dists['car']['max'], Dists['car']['mean'],
           Dists['car']['std'], Dists['car']['lai_conv'], Dists['car']['law'], "car")
    #  cbrown     | Brown pigment => Cbp
    cbrown = (Dists['cbrown']['min'], Dists['cbrown']['max'], Dists['cbrown']['mean'],
              Dists['cbrown']['std'], Dists['cbrown']['lai_conv'], Dists['cbrown']['law'], "cbrown") 
    #  caw        | Equivalent water thickiness
    caw = (Dists['caw']['min'], Dists['caw']['max'], Dists['caw']['mean'],
           Dists['caw']['std'], Dists['caw']['lai_conv'], Dists['caw']['law'], "caw") 
    #  cm         | Dry matter content (in g/cm2, while ATBD uses g/m2)
    cm = (Dists['cm']['min'], Dists['cm']['max'], Dists['cm']['mean'],
          Dists['cm']['std'], Dists['cm']['lai_conv'], Dists['cm']['law'], "cm") 
    #  lai        | Leaf Area Index
    lai = (Dists['lai']['min'], Dists['lai']['max'], Dists['lai']['mean'],
           Dists['lai']['std'], Dists['lai']['lai_conv'], Dists['lai']['law'], "lai") 
    #  lidfa      | Leaf angle distribution => ALA ?
    lidfa = (Dists['lidfa']['min'], Dists['lidfa']['max'], Dists['lidfa']['mean'],
             Dists['lidfa']['std'], Dists['lidfa']['lai_conv'], Dists['lidfa']['law'], "lidfa") 
    #  hspot      | Hotspot parameter => HsD
    hspot = (Dists['hspot']['min'], Dists['hspot']['max'], Dists['hspot']['mean'],
             Dists['hspot']['std'], Dists['hspot']['lai_conv'], Dists['hspot']['law'], "hspot") 
    #  psoil      | Dry/Wet soil factor
    psoil = (Dists['psoil']['min'], Dists['psoil']['max'], Dists['psoil']['mean'],
             Dists['psoil']['std'], Dists['psoil']['lai_conv'], Dists['psoil']['law'], "psoil") 
    #  rsoil      | Soil brigthness factor
    rsoil = (Dists['rsoil']['min'], Dists['rsoil']['max'], Dists['rsoil']['mean'],
             Dists['rsoil']['std'], Dists['rsoil']['lai_conv'], Dists['rsoil']['law'], "rsoil") 
    #  tts      | Solar Zenith Angle
    tts = (Dists['tts']['min'], Dists['tts']['max'], Dists['tts']['mean'],
             Dists['tts']['std'], Dists['tts']['lai_conv'], Dists['tts']['law'], "tts") 
    #  tto      | Observer zenith angle
    tto = (Dists['tto']['min'], Dists['tto']['max'], Dists['tto']['mean'],
             Dists['tto']['std'], Dists['tto']['lai_conv'], Dists['tto']['law'], "tto") 
    #  psi      | Solar/Observer relative azimuth
    psi = (Dists['psi']['min'], Dists['psi']['max'], Dists['psi']['mean'],
             Dists['psi']['std'], Dists['psi']['lai_conv'], Dists['psi']['law'], "psi") 
    
def get_z2prosailparams_mat():
    bounds = ProsailVarsDist()
    z2prosailparams_mat = torch.zeros((11,11))
    z2prosailparams_mat[0,0] = bounds.N[1] - bounds.N[0]
    z2prosailparams_mat[1,1] = bounds.cab[1] - bounds.cab[0]
    z2prosailparams_mat[2,2] = bounds.car[1] - bounds.car[0]
    z2prosailparams_mat[3,3] = bounds.cbrown[1] - bounds.cbrown[0]
    z2prosailparams_mat[4,4] = bounds.caw[1] - bounds.caw[0]
    z2prosailparams_mat[5,5] = bounds.cm[1] - bounds.cm[0]
    z2prosailparams_mat[6,6] = bounds.lai[1] - bounds.lai[0]
    z2prosailparams_mat[7,7] = bounds.lidfa[1] - bounds.lidfa[0]
    z2prosailparams_mat[8,8] = bounds.hspot[1] - bounds.hspot[0]
    z2prosailparams_mat[9,9] = bounds.psoil[1] - bounds.psoil[0]
    z2prosailparams_mat[10,10] = bounds.rsoil[1] - bounds.rsoil[0]
    return z2prosailparams_mat

def get_z2prosailparams_offset():
    bounds = ProsailVarsDist()
    z2prosailparams_offset = torch.tensor([bounds.N[0],
                                           bounds.cab[0],
                                           bounds.car[0],
                                           bounds.cbrown[0],
                                           bounds.caw[0],
                                           bounds.cm[0],
                                           bounds.lai[0],
                                           bounds.lidfa[0],
                                           bounds.hspot[0],
                                           bounds.psoil[0],
                                           bounds.rsoil[0]]).view(-1,1)
    return z2prosailparams_offset

def get_prosailparams_pdf_span():
    bounds = ProsailVarsDist()
    return 1.1 * torch.tensor([bounds.N[1],
                               bounds.cab[1],
                               bounds.car[1],
                               bounds.cbrown[1],
                               bounds.caw[1],
                               bounds.cm[1],
                               bounds.lai[1],
                               bounds.lidfa[1],
                               bounds.hspot[1],
                               bounds.psoil[1],
                               bounds.rsoil[1]])

def plot_rsr(rsr, res_dir='.'):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import (AutoMinorLocator)
    fig, (ax1, ax2, ax3) = plt.subplots(1,3,sharey=True, facecolor='w', figsize=(10,4), dpi=150)
    bands = ["B01","B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8a","B09","B10", "B11", "B12"]
    lamb = np.arange(400,2501)
    for i in range(len(bands)):
        rsr_i = rsr.squeeze()[i,:].numpy()
        non_zero_rsr = rsr_i[rsr_i>0.001]
        non_zero_rsrl = lamb[rsr_i>0.001]
        ax1.plot(non_zero_rsrl, non_zero_rsr, label=bands[i])
        ax2.plot(non_zero_rsrl, non_zero_rsr, label=bands[i])
        ax3.plot(non_zero_rsrl, non_zero_rsr, label=bands[i])

    ax1.set_xlim(400,1000)
    ax2.set_xlim(1300,1700)
    ax3.set_xlim(2000,2400)

    ax1.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax3.spines['left'].set_visible(False)
    ax1.yaxis.tick_left()
    # ax1.tick_params(labelright='off')
    # ax2.tick_params(labelright='off')
    ax2.yaxis.set_visible(False)
    #ax2.set_yticklabels([])

    ax3.yaxis.tick_right()

    ax1.set_ylabel("Relative Spectral Response")
    ax2.set_xlabel("Wavelength (nm)")
    box = ax3.get_position()
    ax3.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    ax3.xaxis.set_minor_locator(AutoMinorLocator())
    ax3.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.tick_params(which='both', axis="both",direction="in")
    ax2.tick_params(which='both', axis="both",direction="in")
    ax3.tick_params(which='both', axis="both",direction="in")

    ax1.tick_params(which='both', width=1)
    ax1.tick_params(which='major', length=5)
    ax1.tick_params(which='minor', length=3)
    ax2.tick_params(which='major', length=5)
    ax2.tick_params(which='minor', length=3)
    ax3.tick_params(which='major', length=5)
    ax3.tick_params(which='minor', length=3)

    ax1t = ax1.twiny()
    ax1t.spines['right'].set_visible(False)
    ax1t.tick_params(which='both', direction = 'in')
    ax1t.tick_params(which='major', length=5)
    ax1t.tick_params(which='minor', length=3)
    ax1t.xaxis.set_minor_locator(AutoMinorLocator())
    ax1t.set_xticklabels([])


    ax2t = ax2.twiny()
    ax2t.spines['left'].set_visible(False)
    ax2t.spines['right'].set_visible(False)
    ax2t.tick_params(which='both', direction = 'in')
    ax2t.tick_params(which='major', length=5)
    ax2t.tick_params(which='minor', length=3)
    ax2t.xaxis.set_minor_locator(AutoMinorLocator())
    ax2t.set_xticklabels([])
    ax3.tick_params(axis="x", which='both', bottom=True, top=True, labelbottom=True, labeltop=False)

    fig.savefig(res_dir+'/rsr.svg')