#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 13:39:15 2022

@author: yoel
"""
from dataclasses import dataclass
import torch
import prosail
from prosail.sail_model import init_prosail_spectra

@dataclass
class ProsailVariables():
    N:torch.Tensor
    cab:torch.Tensor
    car:torch.Tensor
    cbrown:torch.Tensor
    cw:torch.Tensor
    cm:torch.Tensor
    lai:torch.Tensor
    lidfa:torch.Tensor
    hspot:torch.Tensor
    psoil:torch.Tensor
    rsoil:torch.Tensor
    ant:torch.Tensor|None=None
    prot:torch.Tensor|None=None
    cbc:torch.Tensor|None=None

@dataclass
class ProsailConfig():
    factor:str="SDR"
    typelidf:int=2
    spectral_resolution:int=7
    prospect_version:str="5"

@dataclass
class ProsailSpectra():
    soil_spectrum1:torch.Tensor|None=None
    soil_spectrum2:torch.Tensor|None=None
    nr:torch.Tensor|None=None
    kab:torch.Tensor|None=None
    kcar:torch.Tensor|None=None
    kbrown:torch.Tensor|None=None
    kw:torch.Tensor|None=None
    km:torch.Tensor|None=None
    kant:torch.Tensor|None=None
    kprot:torch.Tensor|None=None
    kcbc:torch.Tensor|None=None
    lambdas:torch.Tensor|None=None


PROSAILVARS = ["N", "cab", "car", "cbrown", "cw", "cm",
               "lai", "lidfa", "hspot", "psoil", "rsoil"]

BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]

class ProsailSimulator():
    def __init__(self, prosail_config:ProsailConfig, device="cpu"):
        super().__init__()
        self.prosail_config=prosail_config
        self.device=device
        self.prosail_spectra = ProsailSpectra(*init_prosail_spectra(R_down=self.prosail_config.spectral_resolution, 
                                                                    device=self.device, 
                                                                    prospect_version=self.prosail_config.prospect_version))
        
    def __call__(self, params):
        return self.forward(params)

    def change_device(self, device):
        self.device=device
        self.prosail_spectra = ProsailSpectra(*init_prosail_spectra(R_down=self.prosail_config.spectral_resolution, 
                                                                    device=self.device, 
                                                                    prospect_version=self.prosail_config.prospect_version))
        pass

    def forward(self, params:torch.Tensor):
        # params.shape == [x] => single sample
        # params.shape == [x,y] => batch
        if len(params.shape) == 1:
            params = params.unsqueeze(0)
        prosail_refl = prosail.run_prosail(
            params,
            typelidf=torch.as_tensor(self.prosail_config.typelidf),
            factor=self.prosail_config.factor,
            device=self.device,
            soil_spectrum1=self.prosail_spectra.soil_spectrum1,
            soil_spectrum2=self.prosail_spectra.soil_spectrum2,
            nr=self.prosail_spectra.nr,
            kab=self.prosail_spectra.kab,
            kcar=self.prosail_spectra.kcar,
            kbrown=self.prosail_spectra.kbrown,
            kw=self.prosail_spectra.kw,
            km=self.prosail_spectra.km,
            kant=self.prosail_spectra.kant,
            kprot=self.prosail_spectra.kprot,
            kcbc=self.prosail_spectra.kcbc,
            lambdas=self.prosail_spectra.lambdas,
            R_down=1, 
            init_spectra=False,
            prospect_version=self.prosail_config.prospect_version,
            zero_ant_prot_cbc_override=True
        ).float()
        
        return prosail_refl

if __name__=="__main__":
    prosail_config = ProsailConfig()
    psimulator = ProsailSimulator(prosail_config)

