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
                 last_activation=None, device='cpu', norm_mean=None, norm_std=None): 
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
        self.device=device
        self.net = nn.Sequential(*layers).to(device)
        if norm_mean is None:
            norm_mean = torch.zeros((1,s2refl_size))
        if norm_std is None:
            norm_std = torch.ones((1,s2refl_size))
        self.norm_mean = norm_mean.float().to(device)
        self.norm_std = norm_std.float().to(device)
    
    def change_device(self, device):
        self.device=device
        self.norm_mean = self.norm_mean.to(device)
        self.norm_std = self.norm_std.to(device)
        self.net = self.net.to(device)
        self = self.to(device)

    def encode(self, s2_refl, angles):
        
        normed_refl = (s2_refl - self.norm_mean) / self.norm_std
        y=self.net(torch.concat((normed_refl, 
                                 torch.cos(torch.deg2rad(angles)),
                                 torch.sin(torch.deg2rad(angles))
                                 ), axis=1))
        return y


class EncoderResBlock(Encoder):
    """ 
    A class used to represent a residual MLP encoder of an auto encoder. 
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
    def __init__(self,
                 hidden_layers_size=128, 
                 depth=2,
                 last_activation=None, device='cpu'): 
        super().__init__()

        layers = []        
        for i in range(depth):
            layers.append(nn.Linear(in_features=hidden_layers_size, out_features=hidden_layers_size))
            layers.append(nn.ReLU())
            # layers.append(nn.BatchNorm1d(num_features=out_features))
        
        if last_activation is not None :
            layers.append(last_activation)
        self.device=device
        self.net = nn.Sequential(*layers).to(device)
        
    def change_device(self, device):
        self.device=device
        self.net = self.net.to(device)
        self = self.to(device)

    def forward(self, x):
        y=self.net(x)
        return y + x

class ProsailRNNEncoder(Encoder):
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
                 n_res_block = 3,
                 res_block_layer_sizes=512,
                 res_block_layer_depth=2,
                 last_activation=None, device='cpu', norm_mean=None, norm_std=None): 
        super().__init__()

        resnet = []
        resnet.append(nn.Linear(in_features=s2refl_size + 2 * 3, 
                                out_features=res_block_layer_sizes))

        for i in range(n_res_block):
            resblock = EncoderResBlock(hidden_layers_size=res_block_layer_sizes,depth=res_block_layer_depth,
                                        last_activation=None, device=device)
            resnet.append(resblock)
            # layers.append(nn.BatchNorm1d(num_features=out_features))
        resnet.append(nn.Linear(in_features=res_block_layer_sizes, 
                                out_features=output_size))
        
        if last_activation is not None :
            resnet.append(last_activation)
        self.device=device
        self.net = nn.Sequential(*resnet).to(device)
        if norm_mean is None:
            norm_mean = torch.zeros((1,s2refl_size))
        if norm_std is None:
            norm_std = torch.ones((1,s2refl_size))
        self.norm_mean = norm_mean.float().to(device)
        self.norm_std = norm_std.float().to(device)
    
    def change_device(self, device):
        self.device=device
        self.norm_mean = self.norm_mean.to(device)
        self.norm_std = self.norm_std.to(device)
        self.net = self.net.to(device)
        self = self.to(device)

    def encode(self, s2_refl, angles):
        normed_refl = (s2_refl - self.norm_mean) / self.norm_std
        y=self.net(torch.concat((normed_refl, 
                                 torch.cos(torch.deg2rad(angles)),
                                 torch.sin(torch.deg2rad(angles))
                                 ), axis=1))
        return y