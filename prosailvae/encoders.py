#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 14:19:21 2022

@author: yoel
"""
import torch.nn as nn
from utils.image_utils import batchify_batch_latent, crop_s2_input
import torch
from utils.utils import torch_select_unsqueeze


class Encoder(nn.Module):
    """ 
    A class used to represent an encoder of an auto encoder. This class is to be inherited by all encoders

    ...

    Methods
    -------
    encode(x)
        Encode time series x.
    """
    def encode(self):
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
    def __init__(self, input_size:int=73, output_size:int=12,
                 hidden_layers_size:list[int]|None=None,
                 last_activation=None, device:int='cpu',
                 bands:torch.Tensor|None=None):
        if hidden_layers_size is None:
            hidden_layers_size=[512, 512]
        if bands is None:
            bands = torch.arange(10)
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
        
        self.net = nn.Sequential(*layers).to(device)
        self.bands = bands.to(device)
        self.device = device
        self._spatial_encoding = False

    def get_spatial_encoding(self):
        """
        Return private attribute about wether the encoder takes patches as input
        """
        return self._spatial_encoding
        
    def encode(self, x):
        """
        Encode input data. Asserts input dimension is Batch x Features.
        """
        y = self.net.forward(x[...,self.bands])
        return y
    
    def forward(self, x):
        """
        Encode input data
        """
        return self.encode(x)


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
    def __init__(self, s2refl_size:int=10, output_size:int=12,
                 hidden_layers_size:list[int]|None=None,
                 last_activation=None, device:str='cpu', 
                 norm_mean:torch.Tensor|None=None, norm_std:torch.Tensor|None=None,
                 bands:torch.Tensor | None=None): 
        super().__init__()
        if hidden_layers_size is None:
            hidden_layers_size=[512, 512]
        if bands is None:
            bands = torch.arange(10)
        self.bands = bands.to(device)
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
        self._spatial_encoding = True

    def get_spatial_encoding(self):
        """
        Return private attribute about wether the encoder takes patches as input
        """
        return self._spatial_encoding

    def change_device(self, device):
        """
        Move the class attributes to desired device
        """
        self.device=device
        self.norm_mean = self.norm_mean.to(device)
        self.norm_std = self.norm_std.to(device)
        self.net = self.net.to(device)
        self.bands = self.bands.to(device)

    def encode(self, s2_refl, angles):
        """
        Encode S2 reflectances and angles. Asserts s2_refl dimension is batch x features.
        """
        normed_refl = (s2_refl - self.norm_mean) / self.norm_std
        encoder_output = self.net(torch.concat((normed_refl[:, self.bands],
                                   torch.cos(torch.deg2rad(angles)),
                                   torch.sin(torch.deg2rad(angles))
                                  ), axis=1))
        return encoder_output, angles
    
    def forward(self, s2_refl, angles):
        """
        Encode S2 reflectances and angles
        """
        return self.encode(s2_refl, angles)


class EncoderResBlock(nn.Module):
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
                 hidden_layers_size:int=128,
                 depth:int=2,
                 last_activation=None, device:str='cpu'):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.append(nn.Linear(in_features=hidden_layers_size, out_features=hidden_layers_size))
            layers.append(nn.ReLU())

        if last_activation is not None :
            layers.append(last_activation)
        self.device=device
        self.net = nn.Sequential(*layers).to(device)

    def change_device(self, device:str):
        """
        Move the class attributes to desired device
        """
        self.device=device
        self.net = self.net.to(device)

    def forward(self, x:torch.Tensor):
        y = self.net(x)
        return y + x

class EncoderNNBlock(nn.Module):
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
                 hidden_layers_size:int=128, 
                 depth:int=2,
                 last_activation=None, device:str='cpu'): 
        super().__init__()

        layers = []        
        for _ in range(depth):
            layers.append(nn.Linear(in_features=hidden_layers_size, out_features=hidden_layers_size))
            layers.append(nn.ReLU())
            # layers.append(nn.BatchNorm1d(num_features=out_features))
        
        if last_activation is not None :
            layers.append(last_activation)
        self.device=device
        self.net = nn.Sequential(*layers).to(device)
        
    def change_device(self, device:str):
        """
        Move the class attributes to desired device
        """
        self.device=device
        self.net = self.net.to(device)

    def forward(self, x):
        y = self.net(x)
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
    def __init__(self, s2refl_size:int=10, output_size:int=12, n_res_block:int=3,
                 res_block_layer_sizes:int=512, res_block_layer_depth:int=2,
                 last_activation=None, device:str='cpu', bands:torch.Tensor|None=None,
                 norm_mean:torch.Tensor|None=None, norm_std:torch.Tensor|None=None):
        super().__init__()
        if bands is None:
            assert s2refl_size==10
            bands = torch.arange(10)
        self.bands = bands.to(device)
        resnet = []
        # First Layer
        resnet.append(nn.Linear(in_features=s2refl_size + 2 * 3,
                                out_features=res_block_layer_sizes))
        resnet.append(nn.ReLU())
        # Residual connexion blocks
        for _ in range(n_res_block):
            resblock = EncoderResBlock(hidden_layers_size=res_block_layer_sizes,
                                       depth=res_block_layer_depth,
                                       last_activation=None, device=device)
            resnet.append(resblock)
        # Last layer
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
        self._spatial_encoding = False

    def get_spatial_encoding(self):
        """
        Return private attribute about wether the encoder takes patches as input
        """
        return self._spatial_encoding

    def change_device(self, device):
        """
        Move the class attributes to desired device
        """
        self.device=device
        self.norm_mean = self.norm_mean.to(device)
        self.norm_std = self.norm_std.to(device)
        self.net = self.net.to(device)

    def encode(self, s2_refl, angles):
        """
        Encode S2 reflectances and angles
        """
        normed_refl = (s2_refl - self.norm_mean) / self.norm_std
        encoder_output = self.net(torch.concat((normed_refl[:, self.bands],
                                                torch.cos(torch.deg2rad(angles)),
                                                torch.sin(torch.deg2rad(angles))
                                 ), axis=1))
        return encoder_output, angles

    def forward(self, s2_refl, angles):
        """
        Encode S2 reflectances and angles
        """
        return self.encode(s2_refl, angles)


class ProsailCNNEncoder(nn.Module):
    """
    Implements an encoder with alternate
    convolutional and Relu layers
    """

    def __init__(self, encoder_sizes: list[int]=[20], enc_kernel_sizes: list[int]=[3],
                 device:str='cpu', norm_mean:torch.Tensor|None=None,
                 norm_std:torch.Tensor|None=None, lat_space_size:int=10, padding:str="valid",
                 bands:torch.Tensor|None=None):
        """
        Constructor

        :param encoder_sizes: Number of features for each layer
        :param enc_kernel_sizes: List of kernel sizes for each layer
        """

        super().__init__()
        if bands is None:
            bands = torch.arange(10)
        self.bands = bands.to(device)
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
        self.logvar_conv = nn.Conv2d(encoder_sizes[-1], encoder_sizes[-1]//2, kernel_size=1).to(device)
        self.mu_logvar_conv = nn.Conv2d(encoder_sizes[-1], encoder_sizes[-1], kernel_size=1).to(device)
        if norm_mean is None:
            norm_mean = torch.zeros((lat_space_size,1,1,))
        if norm_std is None:
            norm_std = torch.ones((lat_space_size,1,1))
        self.norm_mean = norm_mean.float().to(device)
        self.norm_std = norm_std.float().to(device)
        self._spatial_encoding = True

    def get_spatial_encoding(self):
        """
        Return private attribute about wether the encoder takes patches as input
        """
        return self._spatial_encoding
    
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
        y = self.cnet(torch.concat((normed_refl[:, self.bands, ...],
                                 torch.cos(torch.deg2rad(angles)),
                                 torch.sin(torch.deg2rad(angles))
                                 ), axis=1))
        # y_mu = self.mu_conv(y)
        # y_logvar = self.logvar_conv(y)
        # y_mu_logvar = torch.concat([y_mu, y_logvar], axis=1)
        y_mu_logvar = self.mu_logvar_conv.forward(y)
        if self.nb_enc_cropped_hw > 0:
            angles = crop_s2_input(angles, self.nb_enc_cropped_hw)
        return batchify_batch_latent(y_mu_logvar), batchify_batch_latent(angles)
    
    def forward(self, s2_refl, angles):
        """
        Encodes S2 reflectances and angles
        """
        return self.encode(s2_refl, angles)

    def change_device(self, device):
        """
        Move the class attributes to desired device
        """
        self.device = device
        self.norm_mean = self.norm_mean.to(device)
        self.norm_std = self.norm_std.to(device)
        self.cnet = self.cnet.to(device)
        self.mu_conv = self.mu_conv.to(device)
        self.logvar_conv = self.logvar_conv.to(device)
        self.bands = self.bands.to(device)

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
                 output_size:int=128,
                 depth:int=2,
                 kernel_size:int=3,
                 last_activation=None,
                 device:str='cpu',
                 input_size:int=10,
                 stride:int=1,
                 padding:str="valid"):
        super().__init__()
        layers = []
        input_sizes = [input_size] + [output_size for i in range(depth-1)]
        for i in range(depth):
            layers.append(nn.Conv2d(
                        input_sizes[i],
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
        self.nb_enc_cropped_hw=0
        for _ in range(depth):
            self.nb_enc_cropped_hw += kernel_size//2
        
    def change_device(self, device):
        """
        Move the class attributes to desired device
        """
        self.device = device
        self.net = self.net.to(device)

    def forward(self, x):
        y=self.net(x)
        x_cropped = x
        patch_size = x.size(-1)
        if self.nb_enc_cropped_hw > 0:
            x_cropped = x[...,self.nb_enc_cropped_hw:patch_size-self.nb_enc_cropped_hw, 
                                self.nb_enc_cropped_hw:patch_size-self.nb_enc_cropped_hw]
        return y + x_cropped

class ProsailRCNNEncoder(nn.Module):
    """
    Implements an encoder with alternate
    convolutional and Relu layers
    """

    def __init__(self, 
                 s2refl_size:int=8,
                 first_layer_kernel:int=7,
                 first_layer_size:int=64,
                 crnn_group_sizes: list[int]=[64,64],
                 crnn_group_depth: list[int]=[2,2],
                 crnn_group_kernel_sizes: list[int]=[3,3],
                 crnn_group_n: list[int] = [1,1],
                 device:str='cpu', norm_mean:torch.Tensor|None=None, 
                 norm_std:torch.Tensor|None=None, output_size:int=11,
                 padding:str='valid', bands:torch.Tensor|None=None
                 ):
        """
        Constructor

        :param encoder_sizes: Number of features for each layer
        :param enc_kernel_sizes: List of kernel sizes for each layer
        """

        super().__init__()
        if bands is None:
            bands = torch.arange(10)
        self.bands = bands.to(device)
        self.device=device
        network = []
        network.append(nn.Conv2d(s2refl_size + 2*3, first_layer_size, first_layer_kernel, 
                                 padding=padding))
        input_sizes = [first_layer_size] + crnn_group_sizes
        assert len(crnn_group_sizes) == len(crnn_group_depth) 
        assert len(crnn_group_depth) == len(crnn_group_kernel_sizes) 
        assert len(crnn_group_kernel_sizes) == len(crnn_group_n)
        n_groups = len(crnn_group_n)
        for i in range(n_groups):
            for _ in range(crnn_group_n[i]):
                network.append(EncoderCResNetBlock(output_size=crnn_group_sizes[i],
                                                   depth=crnn_group_depth[i],
                                                   kernel_size=crnn_group_kernel_sizes[i],
                                                   input_size=input_sizes[i],
                                                   padding=padding))
                network.append(nn.ReLU())
        self.cnet = nn.Sequential(*network).to(device)
        self.mu_conv = nn.Conv2d(input_sizes[-1], output_size, kernel_size=1,
                                 padding=padding).to(device)
        self.logvar_conv = nn.Conv2d(input_sizes[-1], output_size, kernel_size=1,
                                     padding=padding).to(device)
        self.mu_logvar_conv = nn.Conv2d(input_sizes[-1], 2*output_size, kernel_size=1,
                                        padding=padding).to(device)
        if norm_mean is None:
            norm_mean = torch.zeros((s2refl_size,1,1))
        if norm_std is None:
            norm_std = torch.ones((s2refl_size,1,1))
        self.norm_mean = norm_mean.float().to(device)
        self.norm_std = norm_std.float().to(device)
        self.nb_enc_cropped_hw = first_layer_kernel//2
        for i in range(n_groups):
            for _ in range(crnn_group_n[i]):
                for _ in range(crnn_group_depth[i]):
                    self.nb_enc_cropped_hw += crnn_group_kernel_sizes[i]//2
        
        self._spatial_encoding = True

    def get_spatial_encoding(self):
        """
        Return private attribute about wether the encoder takes patches as input
        """
        return self._spatial_encoding
    
    def encode(self, s2_refl, angles):
        """
        Forward pass of the convolutionnal encoder

        :param x: Input tensor of shape [N,C_in,H,W]

        :return: Output Dataclass that holds mu and var
                 tensors of shape [N,C_out,H,W]
        """
        normed_refl = (s2_refl - torch_select_unsqueeze(self.norm_mean,1,4)) / torch_select_unsqueeze(self.norm_std,1,4)
        if len(normed_refl.size())==3:
            normed_refl = normed_refl.unsqueeze(0) # Ensures batch dimension appears
        if len(angles.size())==3:
            angles = angles.unsqueeze(0)
        y = self.cnet(torch.concat((normed_refl[:,self.bands,...],
                                    torch.cos(torch.deg2rad(angles)),
                                    torch.sin(torch.deg2rad(angles))
                                   ), axis=1))
        # y_mu = self.mu_conv(y)
        # y_logvar = self.logvar_conv(y)
        # y_mu_logvar = torch.concat([y_mu, y_logvar], axis=1)
        y_mu_logvar = self.mu_logvar_conv(y)
        angles = angles[:,:,self.nb_enc_cropped_hw:-self.nb_enc_cropped_hw,
                            self.nb_enc_cropped_hw:-self.nb_enc_cropped_hw]
        return batchify_batch_latent(y_mu_logvar), batchify_batch_latent(angles)

    def forward(self, s2_refl, angles):
        """
        Encodes S2 reflectances and angles
        """
        return self.encode(s2_refl, angles)

    def change_device(self, device):
        """
        Move the class attributes to desired device
        """
        self.device=device
        self.norm_mean = self.norm_mean.to(device)
        self.norm_std = self.norm_std.to(device)
        self.cnet = self.cnet.to(device)
        self.mu_conv = self.mu_conv.to(device)
        self.logvar_conv = self.logvar_conv.to(device)
        self.mu_logvar_conv = self.mu_logvar_conv.to(device)
        self.bands = self.bands.to(device)


