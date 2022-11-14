#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 15:24:02 2022

@author: yoel
"""
import pandas as pd
bounds_V1 = pd.DataFrame({'min':{"N":1.0, "cab":10.0, "car":0.0, "cbrown":0.0, "caw":0.004,
                                 "cm":20.0/10000, "lai":0.0, "lidfa":0.0, "hspot":0.0, 
                                 "psoil":0.0, "rsoil":0.3},
                          'max':{"N":3.0, "cab":100.0, "car":1.0, "cbrown":2.0, "caw":0.04,
                                 "cm":165.0/10000, "lai":10.0, "lidfa":90.0, "hspot":0.05, 
                                 "psoil":1.0, "rsoil":3.5},
              
              })

bounds_V2 = {'min':{"N":1.2, "cab":20.0, "car":0, "cbrown":0.0, "caw":0.004,
                                 "cm":30.0/10000, "lai":0.0, "lidfa":5.0, "hspot":0.0, 
                                 "psoil":0.0, "rsoil":0.3},
             'max':{"N":1.8, "cab":90.0, "car":2, "cbrown":2.0, "caw":0.04,
                   "cm":110.0/10000, "lai":5.0, "lidfa":80.0, "hspot":0.05, 
                   "psoil":1.0, "rsoil":3.5},
             'mean':{"N":1.5, "cab":45.0, "car":None, "cbrown":0.0, "caw":None,
                   "cm":50.0/10000, "lai":0.5, "lidfa":40.0, "hspot":None, 
                   "psoil":None, "rsoil":None},
             'std':{"N":0.3, "cab":30.0, "car":None, "cbrown":0.30, "caw":None,
                   "cm":50.0/10000, "lai":1.0, "lidfa":20.0, "hspot":None, 
                   "psoil":None, "rsoil":None},
             'lai_conv':{"N":10, "cab":10, "car":10, "cbrown":10, "caw":10,
                   "cm":10, "lai":10, "lidfa":10, "hspot":10, 
                   "psoil":None, "rsoil":None},
             'law':{"N":'gaussian', "cab":'gaussian', 
                   "car":'gaussian', "cbrown":'gaussian', 
                   "caw":'gaussian', "cm":'gaussian', 
                   "lai":'lognormal', "lidfa":'gaussian', 
                   "hspot":'gaussian', 
                   "psoil":'uniform', "rsoil":'uniform'}
              }

fields = ["min", "max", "mean", "std", "lai_conv", "law"]
params = ["N", "cab", "car", "cbrown", "caw", "cm", "lai", "lidfa", "hspot", "psoil", "rsoil"]
inverted_bounds = {}
for i in range(len(params)):
    inverted_bounds[params[i]]={}
    for j in range(len(fields)):
        inverted_bounds[params[i]][fields[j]]=bounds_V2[fields[j]][params[i]]

class ProsailVarsDist:
    """Values from [AD7]VALSE2-TN-012-CCRS-LAI-v2.0.pdf and
    [AD10]S2PAD-VEGA-ATBD-0003-2_1_L2B_ATBD
        Each tuple is gives the range (min, max) for each variable.
        """
    Dists = {'N': {'min': 1.2, 'max': 1.8, 'mean': 1.5, 'std': 0.3, 'lai_conv': 10,
                   'law': 'gaussian'},
             'cab': {'min': 20.0, 'max': 90.0, 'mean': 45.0, 'std': 30.0, 'lai_conv': 10,
                     'law': 'gaussian'},
             'car': {'min': 0, 'max': 2, 'mean': None, 'std': None, 'lai_conv': 10,
                      'law': 'gaussian'},
             'cbrown': {'min': 0.0, 'max': 2.0, 'mean': 0.0, 'std': 0.3, 'lai_conv': 10,
                        'law': 'gaussian'},
             'caw': {'min': 0.004, 'max': 0.04, 'mean': None, 'std': None, 'lai_conv': 10,
                     'law': 'gaussian'},
             'cm': {'min': 0.003, 'max': 0.011, 'mean': 0.005, 'std': 0.005, 'lai_conv': 10,
                    'law': 'gaussian'},
             'lai': {'min': 0.0, 'max': 5.0, 'mean': 0.5, 'std': 1.0, 'lai_conv': 10,
                     'law': 'lognormal'},
             'lidfa': {'min': 5.0, 'max': 80.0, 'mean': 40.0, 'std': 20.0, 'lai_conv': 10,
                       'law': 'gaussian'},
             'hspot': {'min': 0.0, 'max': 0.05, 'mean': None, 'std': None, 'lai_conv': 10,
                       'law': 'gaussian'},
             'psoil': {'min': 0.0, 'max': 1.0, 'mean': None, 'std': None, 'lai_conv': None,
                       'law': 'uniform'},
             'rsoil': {'min': 0.3, 'max': 3.5, 'mean': None, 'std': None, 'lai_conv': None,
              'law': 'uniform'}}
    
    PROSAILVARS = ["N", "cab", "car", "cbrown", "caw", "cm", 
                   "lai", "lidfa", "hspot", "psoil", "rsoil"]
    #   N         | Leaf structure parameter
    N = (Dists['N']['min'], Dists['N']['max'], Dists['N']['mean'],
         Dists['N']['std'], Dists['N']['lai_conv'], Dists['N']['law'])
    
    #  cab        | Chlorophyll a+b concentration
    cab = (Dists['cab']['min'], Dists['cab']['max'], Dists['cab']['mean'],
           Dists['cab']['std'], Dists['cab']['lai_conv'], Dists['cab']['law'])
    #  car        | Carotenoid concentration
    car = (Dists['car']['min'], Dists['car']['max'], Dists['car']['mean'],
           Dists['car']['std'], Dists['car']['lai_conv'], Dists['car']['law'])
    #  cbrown     | Brown pigment => Cbp
    cbrown = (Dists['cbrown']['min'], Dists['cbrown']['max'], Dists['cbrown']['mean'],
              Dists['cbrown']['std'], Dists['cbrown']['lai_conv'], Dists['cbrown']['law']) 
    #  caw        | Equivalent water thickiness
    caw = (Dists['caw']['min'], Dists['caw']['max'], Dists['caw']['mean'],
           Dists['caw']['std'], Dists['caw']['lai_conv'], Dists['caw']['law']) 
    #  cm         | Dry matter content (in g/cm2, while ATBD uses g/m2)
    cm = (Dists['cm']['min'], Dists['cm']['max'], Dists['cm']['mean'],
          Dists['cm']['std'], Dists['cm']['lai_conv'], Dists['cm']['law']) 
    #  lai        | Leaf Area Index
    lai = (Dists['lai']['min'], Dists['lai']['max'], Dists['lai']['mean'],
           Dists['lai']['std'], Dists['lai']['lai_conv'], Dists['lai']['law']) 
    #  lidfa      | Leaf angle distribution => ALA ?
    lidfa = (Dists['lidfa']['min'], Dists['lidfa']['max'], Dists['lidfa']['mean'],
             Dists['lidfa']['std'], Dists['lidfa']['lai_conv'], Dists['lidfa']['law']) 
    #  hspot      | Hotspot parameter => HsD
    hspot = (Dists['hspot']['min'], Dists['hspot']['max'], Dists['hspot']['mean'],
             Dists['hspot']['std'], Dists['hspot']['lai_conv'], Dists['hspot']['law']) 
    #  psoil      | Dry/Wet soil factor
    psoil = (Dists['psoil']['min'], Dists['psoil']['max'], Dists['psoil']['mean'],
             Dists['psoil']['std'], Dists['psoil']['lai_conv'], Dists['psoil']['law']) 
    #  rsoil      | Soil brigthness factor
    rsoil = (Dists['rsoil']['min'], Dists['rsoil']['max'], Dists['rsoil']['mean'],
             Dists['rsoil']['std'], Dists['rsoil']['lai_conv'], Dists['rsoil']['law']) 