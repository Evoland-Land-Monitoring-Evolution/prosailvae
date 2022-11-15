#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 14:19:21 2022

@author: yoel
"""
import torch.nn as nn
import torch

class Encoder(nn.Module):
    """ 
    A class used to represent an encoder of an auto encoder. This class is to be inherited by all encoders

    ...

    Methods
    -------
    encode(x)
        Encode time series x.
    """
    def encode():
        raise NotImplementedError


class NNEncoder(Encoder):
    """ 
    A class used to represent a simple MLP encoder of an auto encoder. 
    ...

    Attributes
    ----------
    net : nn.Sequential
        NN layers of the encoder
    
    Methods
    -------
    encode(x)
        Encode time series x using net.
    """
    def __init__(self, input_size=73, output_size=12, 
                 hidden_layers_size=[400, 500, 300, 100], 
                 last_activation=None, device='cpu'): 
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_features=input_size, 
                                out_features=hidden_layers_size[0]))
        
        for i in range(len(hidden_layers_size)-1):
            
            in_features = hidden_layers_size[i]
            out_features = hidden_layers_size[i+1]
            layers.append(nn.Linear(in_features=in_features, out_features=out_features))
            layers.append(nn.ReLU())
            # layers.append(nn.BatchNorm1d(num_features=out_features))
        layers.append(nn.Linear(in_features=hidden_layers_size[-1], 
                                out_features=output_size))
        
        if last_activation is not None :
            layers.append(last_activation)
        
        self.net = nn.Sequential(*layers)
        self.device=device
        
    def encode(self, x):
        y=self.net(x)
        return y


class ProsailNNEncoder(Encoder):
    """ 
    A class used to represent a simple MLP encoder of an auto encoder. 
    ...

    Attributes
    ----------
    net : nn.Sequential
        NN layers of the encoder
    
    Methods
    -------
    encode(x)
        Encode time series x using net.
    """
    def __init__(self, s2refl_size=10, output_size=12, 
                 hidden_layers_size=[400, 500, 300, 100], 
                 last_activation=None, device='cpu'): 
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_features=s2refl_size + 2 * 3, 
                                out_features=hidden_layers_size[0]))
        
        for i in range(len(hidden_layers_size)-1):
            
            in_features = hidden_layers_size[i]
            out_features = hidden_layers_size[i+1]
            layers.append(nn.Linear(in_features=in_features, out_features=out_features))
            layers.append(nn.ReLU())
            # layers.append(nn.BatchNorm1d(num_features=out_features))
        layers.append(nn.Linear(in_features=hidden_layers_size[-1], 
                                out_features=output_size))
        
        if last_activation is not None :
            layers.append(last_activation)
        
        self.net = nn.Sequential(*layers)
        self.device=device
        
    def encode(self, s2_refl, angles):
        print(s2_refl.device, angles.device)
        y=self.net(torch.concat((s2_refl, 
                                 torch.cos(torch.deg2rad(angles)),
                                 torch.sin(torch.deg2rad(angles))
                                 ), axis=1))
        raise NotImplementedError
        return y