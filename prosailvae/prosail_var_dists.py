#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 15:24:02 2022

@author: yoel
"""
from dataclasses import dataclass, asdict
import torch
import numpy as np

@dataclass
class VariableDistribution:
    low: float|None = None
    high: float|None = None
    loc: float|None = None
    scale: float|None = None    
    C_lai_min: float|None = None 
    C_lai_max: float|None = None     
    lai_conv: float|None = 10
    law: str = "gaussian"

@dataclass
class VariableBounds:
    low: float|None = None
    high: float|None = None

@dataclass#(frozen=True)
class ProsailVarsDistLegacy:
    """
    Variable distribution used previously
    """
    sentinel2_max_tto = np.rad2deg(np.arctan(145/786))
    solar_max_zenith_angle = 60
    N     :VariableDistribution = VariableDistribution(low=1.2,    high=1.8,   loc=1.3,   scale=0.3,   C_lai_min=None, C_lai_max=None, law="gaussian")
    cab   :VariableDistribution = VariableDistribution(low=20.0,   high=90.0,  loc=45.0,  scale=30.0,  C_lai_min=None, C_lai_max=None, law="gaussian")
    car   :VariableDistribution = VariableDistribution(low=5,      high=23,    loc=11,    scale=5,     C_lai_min=None, C_lai_max=None, law="gaussian")
    cbrown:VariableDistribution = VariableDistribution(low=0,      high=2,     loc=0.0,   scale=0.3,   C_lai_min=None, C_lai_max=None, law="gaussian")
    cw    :VariableDistribution = VariableDistribution(low=0.0075, high=0.075, loc=0.025, scale=0.02,  C_lai_min=None, C_lai_max=None, law="gaussian")
    cm    :VariableDistribution = VariableDistribution(low=0.003,  high=0.011, loc=0.005, scale=0.005, C_lai_min=None, C_lai_max=None, law="gaussian")
    lai   :VariableDistribution = VariableDistribution(low=0,      high=10,    loc=2,     scale=3,     C_lai_min=None, C_lai_max=None, law="gaussian")
    lidfa :VariableDistribution = VariableDistribution(low=30.0,   high=80.0,  loc=60,    scale=20,    C_lai_min=None, C_lai_max=None, law="gaussian")
    hspot :VariableDistribution = VariableDistribution(low=0.0,    high=0.5,   loc=0.25,  scale=0.5,   C_lai_min=None, C_lai_max=None, law="gaussian")
    psoil :VariableDistribution = VariableDistribution(low=0,      high=1,     loc=None,  scale=None,  C_lai_min=None, C_lai_max=None, law="uniform")
    rsoil :VariableDistribution = VariableDistribution(low=0.3,    high=3.5,   loc=None,  scale=None,  C_lai_min=None, C_lai_max=None, law="uniform")

    tts  = VariableDistribution(low=0, high=solar_max_zenith_angle,   
                                loc=None,  scale=None,  C_lai_min=None, C_lai_max=None, law="uniform")
    tto  = VariableDistribution(low=0, high=sentinel2_max_tto,     
                                loc=None,  scale=None,  C_lai_min=None, C_lai_max=None, law="uniform")
    psi  = VariableDistribution(low=0, high=360,   loc=None,  scale=None,  
                                C_lai_min=None, C_lai_max=None, law="uniform")
    def asdict(self):
        return asdict(self)
    
@dataclass#(frozen=True)
class SamplingProsailVarsDist:
    """Values from [AD7]VALSE2-TN-012-CCRS-LAI-v2.0.pdf and
    [AD10]S2PAD-VEGA-ATBD-0003-2_1_L2B_ATBD
        Each tuple is gives the range (min, high) for each variable.
        """
    N     :VariableDistribution = VariableDistribution(low=1.2,    high=2.2,   loc=1.5,   scale=0.3,   C_lai_min=1.3,    C_lai_max=1.8,   law="gaussian", lai_conv=10)
    cab   :VariableDistribution = VariableDistribution(low=20.0,   high=90.0,  loc=45.0,  scale=30.0,  C_lai_min=45,     C_lai_max=90,    law="gaussian", lai_conv=10)
    car   :VariableDistribution = VariableDistribution(low=5,      high=23,    loc=11,    scale=5,     C_lai_min=5,      C_lai_max=23,    law="gaussian", lai_conv=10)
    cbrown:VariableDistribution = VariableDistribution(low=0,      high=2,     loc=0.0,   scale=0.3,   C_lai_min=0,      C_lai_max=0.2,   law="gaussian", lai_conv=10)
    cw    :VariableDistribution = VariableDistribution(low=0.0075, high=0.075, loc=0.025, scale=0.02,  C_lai_min=0.0075, C_lai_max=0.075, law="gaussian", lai_conv=10)
    cm    :VariableDistribution = VariableDistribution(low=0.003,  high=0.011, loc=0.005, scale=0.005, C_lai_min=0.003,  C_lai_max=0.011, law="gaussian", lai_conv=10)
    lai   :VariableDistribution = VariableDistribution(low=0,      high=15,    loc=2,     scale=3,     C_lai_min=None,   C_lai_max=None,  law="gaussian", lai_conv=None)
    lidfa :VariableDistribution = VariableDistribution(low=30.0,   high=80.0,  loc=60,    scale=30,    C_lai_min=55,     C_lai_max=65,    law="gaussian", lai_conv=10)
    hspot :VariableDistribution = VariableDistribution(low=0.1,    high=0.5,   loc=0.2,   scale=0.5,   C_lai_min=0.1,    C_lai_max=0.5,   law="gaussian", lai_conv=1000)
    psoil :VariableDistribution = VariableDistribution(low=0,      high=1,     loc=0.5,   scale=0.5,   C_lai_min=0,      C_lai_max=1,     law="uniform",  lai_conv=10)
    rsoil :VariableDistribution = VariableDistribution(low=0.5,    high=3.5,   loc=1.2,   scale=2,     C_lai_min=0.5,    C_lai_max=1.2,   law="uniform",  lai_conv=10)
    def asdict(self):
        return asdict(self)

@dataclass#(frozen=True)
class ProsailVarsBounds:
    
    N:VariableBounds      = VariableBounds(low=1,      high=3       )
    cab:VariableBounds    = VariableBounds(low=0.0,    high=100     )
    car:VariableBounds    = VariableBounds(low=0,      high=40      )
    cbrown:VariableBounds = VariableBounds(low=0,      high=2       )
    cw:VariableBounds     = VariableBounds(low=0.0,    high=0.01    )
    cm:VariableBounds     = VariableBounds(low=0.0,    high=0.02    )
    lai:VariableBounds    = VariableBounds(low=0,      high=15      )
    lidfa:VariableBounds  = VariableBounds(low=30.0,   high=80.0    )
    hspot:VariableBounds  = VariableBounds(low=0.0,    high=0.5     )
    psoil:VariableBounds  = VariableBounds(low=0,      high=1       )
    rsoil:VariableBounds  = VariableBounds(low=0.0,    high=3.5     )
    def asdict(self):
        return asdict(self)

@dataclass#(frozen=True)
class ProsailVarsBoundsLegacy:
    """
    Variable bounds used previously
    """
    N:VariableBounds      = VariableBounds(low=1.2,    high=1.8,  )
    cab:VariableBounds    = VariableBounds(low=20.0,   high=90.0, )
    car:VariableBounds    = VariableBounds(low=5,      high=23,   )
    cbrown:VariableBounds = VariableBounds(low=0,      high=2,    )
    cw:VariableBounds     = VariableBounds(low=0.0075, high=0.075,)
    cm:VariableBounds     = VariableBounds(low=0.003,  high=0.011,)
    lai:VariableBounds    = VariableBounds(low=0,      high=10,   )
    lidfa:VariableBounds  = VariableBounds(low=30.0,   high=80.0, )
    hspot:VariableBounds  = VariableBounds(low=0.0,    high=0.5,  )
    psoil:VariableBounds  = VariableBounds(low=0,      high=1,    )
    rsoil:VariableBounds  = VariableBounds(low=0.3,    high=3.5,  )

    def asdict(self):
        return asdict(self)
    
def get_z2prosailparams_mat(bounds=None):
    z2prosailparams_mat = torch.diag(get_prosail_vars_interval_width(bounds=bounds))
    return z2prosailparams_mat

def get_z2prosailparams_bound(which='high', bounds=None):
    if bounds is None:
        bounds = ProsailVarsDistLegacy()
    if which =="high":
        return torch.tensor([   bounds.N.high,
                                bounds.cab.high,
                                bounds.car.high,
                                bounds.cbrown.high,
                                bounds.cw.high,
                                bounds.cm.high,
                                bounds.lai.high,
                                bounds.lidfa.high,
                                bounds.hspot.high,
                                bounds.psoil.high,
                                bounds.rsoil.high])
    elif which == "low":
        return torch.tensor([bounds.N.low,
                             bounds.cab.low,
                             bounds.car.low,
                             bounds.cbrown.low,
                             bounds.cw.low,
                             bounds.cm.low,
                             bounds.lai.low,
                             bounds.lidfa.low,
                             bounds.hspot.low,
                             bounds.psoil.low,
                             bounds.rsoil.low])
    else:
        raise ValueError

def get_z2prosailparams_offset(bounds):
    return get_z2prosailparams_bound(which='low', bounds=bounds).view(-1,1)

def get_prosailparams_pdf_span(bounds):
    return 1.1 * get_z2prosailparams_bound(which='high', bounds=bounds)

def get_prosail_vars_interval_width(bounds_type="legacy", bounds=None):
    if bounds is None:
        bounds = ProsailVarsDistLegacy()
    return get_z2prosailparams_bound(which='high', bounds=bounds) - get_z2prosailparams_bound(which='low', bounds=bounds)

def get_prosail_var_bounds(which="legacy"):
    if which == "legacy":
        return ProsailVarsBoundsLegacy()
    if which == "new":
        return ProsailVarsBounds()
    else:
        raise ValueError

def get_prosail_var_dist(which="legacy"):
    if which == "legacy":
        return ProsailVarsDistLegacy()
    if which == "new":
        return SamplingProsailVarsDist()
    else:
        raise ValueError

