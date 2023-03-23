#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 14:19:21 2022

@author: yoel
"""
import torch.nn as nn
import torch
from prosailvae.utils import torch_select_unsqueeze
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
        return y, angles


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

class EncoderNNBlock(Encoder):
    """ 
    A class used to represent a MPL block encoder of an auto encoder. 
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
        return y

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
        resnet.append(nn.ReLU())

        for i in range(n_res_block):
            # resblock = nn.RNN(input_size=res_block_layer_sizes, hidden_size=res_block_layer_depth, 
            #                     num_layers=res_block_layer_depth,nonlinearity='relu',batch_first=True)
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
            norm_mean = torch.zeros((1, s2refl_size))
        if norm_std is None:
            norm_std = torch.ones((1, s2refl_size))
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
        return y, angles

def batchify_batch_latent(y):
    # Input dim (B x 2L x H x W)
    y = y.permute(0,2,3,1)
    return y.reshape(-1, y.size(3))

def get_invalid_symetrical_padding(enc_kernel_sizes):
    hw = 0
    for i in range(len(enc_kernel_sizes)):
        hw += enc_kernel_sizes[i]//2
    return hw

class ProsailCNNEncoder(nn.Module):
    """
    Implements an encoder with alternate
    convolutional and Relu layers
    """

    def __init__(self, encoder_sizes: list[int]=[20], enc_kernel_sizes: list[int]=[3],
                 device='cpu', norm_mean=None, norm_std=None, lat_space_size=10, padding="valid"):
        """
        Constructor

        :param encoder_sizes: Number of features for each layer
        :param enc_kernel_sizes: List of kernel sizes for each layer
        """

        super().__init__()
        self.device=device
        enc_blocks: list[nn.Module] = sum(
            [
                [
                    nn.Conv2d(
                        in_layer,
                        out_layer,
                        kernel_size=kernel_size,
                        padding=padding,
                    ),
                    nn.ReLU(),
                ]
                for in_layer, out_layer, kernel_size in zip(
                    encoder_sizes, encoder_sizes[1:], enc_kernel_sizes
                )
            ],
            [],
        )
        self.cnet = nn.Sequential(*enc_blocks).to(device)
        self.nb_enc_cropped_hw = sum(
            [(kernel_size - 1) // 2 for kernel_size in enc_kernel_sizes]
        )
        self.mu_conv = nn.Conv2d(encoder_sizes[-1], encoder_sizes[-1]//2, kernel_size=1).to(device)
        self.logvar_conv = nn.Conv2d(
            encoder_sizes[-1], encoder_sizes[-1]//2, kernel_size=1
        ).to(device)
        if norm_mean is None:
            norm_mean = torch.zeros((lat_space_size,1,1,))
        if norm_std is None:
            norm_std = torch.ones((lat_space_size,1,1))
        self.norm_mean = norm_mean.float().to(device)
        self.norm_std = norm_std.float().to(device)

    def encode(self, s2_refl, angles):
        """
        Forward pass of the convolutionnal encoder

        :param x: Input tensor of shape [N,C_in,H,W]

        :return: Output Dataclass that holds mu and var
                 tensors of shape [N,C_out,H,W]
        """
        normed_refl = (s2_refl - self.norm_mean) / self.norm_std
        if len(normed_refl.size())==3:
            normed_refl = normed_refl.unsqueeze(0)
        if len(angles.size())==3:
            angles = angles.unsqueeze(0)
        y=self.cnet(torch.concat((normed_refl, 
                                 torch.cos(torch.deg2rad(angles)),
                                 torch.sin(torch.deg2rad(angles))
                                 ), axis=1))
        y_mu = self.mu_conv(y)
        y_logvar = self.logvar_conv(y)
        y_mu_logvar = torch.concat([y_mu, y_logvar], axis=1)
        if self.nb_enc_cropped_hw > 0:
            angles = angles[:,:,self.nb_enc_cropped_hw:-self.nb_enc_cropped_hw,self.nb_enc_cropped_hw:-self.nb_enc_cropped_hw]
        return batchify_batch_latent(y_mu_logvar), batchify_batch_latent(angles)

    def change_device(self, device):
        self.device=device
        self.norm_mean = self.norm_mean.to(device)
        self.norm_std = self.norm_std.to(device)
        self.cnet = self.dnet.to(device)
        self.mu_conv = self.mu_conv.to(device)
        self.logvar_conv = self.logvar_conv.to(device)
        self = self.to(device)

class EncoderCResNetBlock(Encoder):
    """ 
    A class used to represent a residual CNN block of for a CNN auto encoder. 
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
                 output_size=128, 
                 depth=2,
                 kernel_size=3,
                 last_activation=None, 
                 device='cpu', 
                 input_size=10,
                 stride=1,
                 padding="valid"): 
        super().__init__()

        layers = []        
        for i in range(depth):
            layers.append(nn.Conv2d(
                        input_size,
                        output_size,
                        kernel_size=kernel_size,
                        padding=padding,
                        stride=stride
                    ))
            if i < depth-1:
                layers.append(nn.ReLU())
        
        if last_activation is not None :
            layers.append(last_activation)
        self.device=device
        self.net = nn.Sequential(*layers).to(device)
        self.hw=0
        for j in range(depth):
            self.hw += kernel_size//2
        
    def change_device(self, device):
        self.device=device
        self.net = self.net.to(device)
        self = self.to(device)

    def forward(self, x):
        y=self.net(x)
        x_cropped = x
        patch_size = x.size(-1)
        if self.hw > 0:
            x_cropped = x[...,self.hw:patch_size-self.hw, self.hw:patch_size-self.hw]
        return y + x_cropped

class ProsailRCNNEncoder(nn.Module):
    """
    Implements an encoder with alternate
    convolutional and Relu layers
    """

    def __init__(self, 
                 s2refl_size=8,
                 first_layer_kernel=7,
                 first_layer_size=64,
                 crnn_group_sizes: list[int]=[64,64], 
                 crnn_group_depth: list[int]=[2,2], 
                 crnn_group_kernel_sizes: list[int]=[3,3],
                 crnn_group_n = [1,1],
                 device='cpu', norm_mean=None, norm_std=None, output_size=11,
                 padding='valid'):
        """
        Constructor

        :param encoder_sizes: Number of features for each layer
        :param enc_kernel_sizes: List of kernel sizes for each layer
        """

        super().__init__()
        self.device=device
        network = []
        network.append(nn.Conv2d(s2refl_size + 2*3, first_layer_size, first_layer_kernel, padding=padding))
        input_sizes = [first_layer_size] + crnn_group_sizes
        assert len(crnn_group_sizes) == len(crnn_group_depth) and len(crnn_group_depth) == len(crnn_group_kernel_sizes) and len(crnn_group_kernel_sizes) == len(crnn_group_n)
        for i in range(len(crnn_group_n)):
            for _ in range(crnn_group_n[i]):
                network.append(EncoderCResNetBlock(output_size=crnn_group_sizes[i],
                                                   depth=crnn_group_depth[i],
                                                   kernel_size=crnn_group_kernel_sizes[i],
                                                   input_size=input_sizes[i],
                                                   padding=padding))
                network.append(nn.ReLU())
        self.cnet = nn.Sequential(*network).to(device)
        self.mu_conv = nn.Conv2d(input_sizes[-1], output_size, kernel_size=1, padding=padding).to(device)
        self.logvar_conv = nn.Conv2d(input_sizes[-1], output_size, kernel_size=1, padding=padding).to(device)
        if norm_mean is None:
            norm_mean = torch.zeros((s2refl_size,1,1,))
        if norm_std is None:
            norm_std = torch.ones((s2refl_size,1,1))
        self.norm_mean = norm_mean.float().to(device)
        self.norm_std = norm_std.float().to(device)
        self.hw = first_layer_kernel//2
        for i in range(len(crnn_group_n)):
            for j in range(crnn_group_n[i]):
                for k in range(crnn_group_depth[i]):
                    self.hw += crnn_group_kernel_sizes[i]//2

    def encode(self, s2_refl, angles):
        """
        Forward pass of the convolutionnal encoder

        :param x: Input tensor of shape [N,C_in,H,W]

        :return: Output Dataclass that holds mu and var
                 tensors of shape [N,C_out,H,W]
        """
        normed_refl = (s2_refl - torch_select_unsqueeze(self.norm_mean,1,4)) / torch_select_unsqueeze(self.norm_std,1,4)
        if len(normed_refl.size())==3:
            normed_refl = normed_refl.unsqueeze(0)
        if len(angles.size())==3:
            angles = angles.unsqueeze(0)
        y=self.cnet(torch.concat((normed_refl, 
                                 torch.cos(torch.deg2rad(angles)),
                                 torch.sin(torch.deg2rad(angles))
                                 ), axis=1))
        y_mu = self.mu_conv(y)
        y_logvar = self.logvar_conv(y)
        y_mu_logvar = torch.concat([y_mu, y_logvar], axis=1)
        angles = angles[:,:,self.hw:-self.hw,self.hw:-self.hw]
        return batchify_batch_latent(y_mu_logvar), batchify_batch_latent(angles)

    def change_device(self, device):
        self.device=device
        self.norm_mean = self.norm_mean.to(device)
        self.norm_std = self.norm_std.to(device)
        self.cnet = self.dnet.to(device)
        self.mu_conv = self.mu_conv.to(device)
        self.logvar_conv = self.logvar_conv.to(device)
        self = self.to(device)


