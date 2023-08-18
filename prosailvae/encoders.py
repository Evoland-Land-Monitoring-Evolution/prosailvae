#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 14:19:21 2022

@author: yoel
"""
from dataclasses import dataclass, field
import torch.nn as nn
import torch
from utils.utils import torch_select_unsqueeze, standardize, IOStandardizeCoeffs
from utils.image_utils import batchify_batch_latent, crop_s2_input, check_is_patch
from .spectral_indices import get_spectral_idx

@dataclass
class EncoderConfig:
    """
    Configuration to initialize any encoder
    """
    io_coeffs:IOStandardizeCoeffs
    encoder_type:str='rnn'
    input_size:int = 16
    output_size:int = 12
    device:str='cpu'
    bands:torch.Tensor|None = torch.arange(10)
    last_activation:nn.Module|None = None
    n_latent_params:int=2
    layer_sizes:list[int]|None = field(default_factory=lambda: [128])
    kernel_sizes: list[int] = field(default_factory=lambda: [3])
    padding:str = "valid"

    first_layer_kernel:int = 3
    first_layer_size:int = 128
    block_layer_sizes: list[int] = field(default_factory=lambda: [128, 128])
    block_layer_depths: list[int] = field(default_factory=lambda: [2, 2])
    block_kernel_sizes: list[int] = field(default_factory=lambda: [3, 1])
    block_n: list[int] = field(default_factory=lambda: [1, 2])

class Encoder(nn.Module):
    """ 
    A class used to represent an encoder of an auto encoder. 
    This class is to be inherited by all encoders

    ...

    Methods
    -------
    encode(x)
        Encode time series x.
    """
    def encode(self):
        raise NotImplementedError


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
    def __init__(self, config:EncoderConfig, device:str='cpu',):
        super().__init__()
        if hidden_layers_size is None:
            hidden_layers_size=[512, 512]
            bands = config
        if bands is None:
            bands = torch.arange(10)
        self.bands = bands.to(device)
        layers = []
        layers.append(nn.Linear(in_features=config.input_size,
                                out_features=hidden_layers_size[0]))

        for i in range(len(hidden_layers_size)-1):

            in_features = hidden_layers_size[i]
            out_features = hidden_layers_size[i+1]
            layers.append(nn.Linear(in_features=in_features, out_features=out_features))
            layers.append(nn.ReLU())
            # layers.append(nn.BatchNorm1d(num_features=out_features))
        layers.append(nn.Linear(in_features=hidden_layers_size[-1],
                                out_features=config.n_latent_params * config.output_size))

        if config.last_activation is not None :
            layers.append(config.last_activation)
        self.device=device
        self.net = nn.Sequential(*layers).to(device)

        bands_loc = config.io_coeffs.bands.loc 
        idx_loc = config.io_coeffs.idx.loc 
        angles_loc = config.io_coeffs.angles.loc 
        bands_scale = config.io_coeffs.bands.scale 
        idx_scale = config.io_coeffs.idx.scale 
        angles_scale = config.io_coeffs.angles.scale 
        bands_loc = bands_loc if bands_loc is not None else torch.zeros((config.input_size))
        bands_scale = bands_scale if bands_scale is not None else torch.ones((config.input_size))
        idx_loc = idx_loc if idx_loc is not None else torch.zeros((5))
        idx_scale = idx_scale if idx_scale is not None else torch.ones((5))
        angles_loc = angles_loc if angles_loc is not None else torch.zeros((3))
        angles_scale = angles_scale if angles_scale is not None else torch.ones((3))
        self.bands_loc = bands_loc.float().to(device)
        self.bands_scale = bands_scale.float().to(device)
        self.idx_loc = idx_loc.float().to(device)
        self.idx_scale = idx_scale.float().to(device)
        self.angles_loc = angles_loc.float().to(device)
        self.angles_scale = angles_scale.float().to(device)
        self._spatial_encoding = False
        self.nb_enc_cropped_hw = 0

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
        self.bands_loc = self.bands_loc.to(device)
        self.bands_scale = self.bands_scale.to(device)
        self.idx_loc = self.idx_loc.to(device)
        self.idx_scale = self.idx_scale.to(device)
        self.angles_loc = self.angles_loc.to(device)
        self.angles_scale = self.angles_scale.to(device)
        self.net = self.net.to(device)
        self.bands = self.bands.to(device)

    def encode(self, s2_refl, angles):
        """
        Encode S2 reflectances and angles. Asserts s2_refl dimension is batch x features.
        """
        spectral_idx = get_spectral_idx(s2_refl, bands_dim=1)
        normed_refl = standardize(s2_refl[:, self.bands], loc=self.bands_loc, scale=self.bands_scale, dim=1)
        # normed_angles = standardize(torch.concat(torch.cos(torch.deg2rad(angles)),
                                                #  torch.sin(torch.deg2rad(angles)), 
                                                #  axis=1), 
                                                #  loc=self.angles_loc, scale=self.angles_scale, dim=1)   
        normed_angles = standardize(torch.cos(torch.deg2rad(angles)), self.angles_loc, 
                                    self.angles_scale, dim=1)                      
        encoder_output = self.net(torch.concat((normed_refl, normed_angles, spectral_idx), axis=1))
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
        for i in range(depth):
            layers.append(nn.Linear(in_features=hidden_layers_size, out_features=hidden_layers_size))
            if i < depth - 1:
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
    def __init__(self, config:EncoderConfig, device:str='cpu'):
        super().__init__()
        bands = config.bands
        if bands is None:
            bands = torch.arange(10)
        self.bands = bands.to(device)
        resnet = []
        # First Layer
        resnet.append(nn.Linear(in_features=config.input_size,
                                out_features=config.block_layer_sizes[0]))
        resnet.append(nn.ReLU())
        # Residual connexion blocks
        n_groups = len(config.block_n)
        for i in range(n_groups):
            for _ in range(config.block_n[i]):
                resblock = EncoderResBlock(hidden_layers_size=config.block_layer_sizes[i],
                                           depth=config.block_layer_depths[i],
                                           last_activation=None, device=device)
                resnet.append(resblock)
                resnet.append(nn.ReLU())
        # Last layer
        resnet.append(nn.Linear(in_features=config.block_layer_sizes[-1],
                                out_features=config.n_latent_params * config.output_size))
        if config.last_activation is not None :
            resnet.append(config.last_activation)
        self.device=device
        self.net = nn.Sequential(*resnet).to(device)
        bands_loc = config.io_coeffs.bands.loc 
        idx_loc = config.io_coeffs.idx.loc 
        angles_loc = config.io_coeffs.angles.loc 
        bands_scale = config.io_coeffs.bands.scale 
        idx_scale = config.io_coeffs.idx.scale 
        angles_scale = config.io_coeffs.angles.scale 
        bands_loc = bands_loc if bands_loc is not None else torch.zeros((config.input_size))
        bands_scale = bands_scale if bands_scale is not None else torch.ones((config.input_size))
        idx_loc = idx_loc if idx_loc is not None else torch.zeros((5))
        idx_scale = idx_scale if idx_scale is not None else torch.ones((5))
        angles_loc = angles_loc if angles_loc is not None else torch.zeros((3))
        angles_scale = angles_scale if angles_scale is not None else torch.ones((3))
        self.bands_loc = bands_loc.float().to(device)
        self.bands_scale = bands_scale.float().to(device)
        self.idx_loc = idx_loc.float().to(device)
        self.idx_scale = idx_scale.float().to(device)
        self.angles_loc = angles_loc.float().to(device)
        self.angles_scale = angles_scale.float().to(device)
        self._spatial_encoding = False
        self.nb_enc_cropped_hw = 0

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
        self.bands_loc = self.bands_loc.to(device)
        self.bands_scale = self.bands_scale.to(device)
        self.idx_loc = self.idx_loc.to(device)
        self.idx_scale = self.idx_scale.to(device)
        self.angles_loc = self.angles_loc.to(device)
        self.angles_scale = self.angles_scale.to(device)
        self.net = self.net.to(device)

    def encode(self, s2_refl, angles):
        """
        Encode S2 reflectances and angles
        """
        if len(s2_refl.size())==4:
            s2_refl = batchify_batch_latent(s2_refl)
            angles = batchify_batch_latent(angles)
        spectral_idx = get_spectral_idx(s2_refl, bands_dim=1)
        if s2_refl.size(1) == self.bands_loc.size(0): # Same number of bands in input than in normalization
            normed_refl = standardize(s2_refl, loc=self.bands_loc, scale=self.bands_scale, dim=1)[:, self.bands]
        elif len(self.bands) == self.bands_loc.size(0): # Same number of bands in bands than in normalization
            normed_refl = standardize(s2_refl[:, self.bands], loc=self.bands_loc, scale=self.bands_scale, dim=1)
        else:
            raise NotImplementedError
        # normed_angles = standardize(torch.concat((torch.cos(torch.deg2rad(angles)),
        #                                           torch.sin(torch.deg2rad(angles))), 
        #                                           axis=1), 
        #                             loc=self.angles_loc, scale=self.angles_scale, dim=1)
        normed_angles = standardize(torch.cos(torch.deg2rad(angles)), self.angles_loc, 
                                    self.angles_scale, dim=1)
        encoder_output = self.net(torch.concat((normed_refl, normed_angles, spectral_idx), axis=1))
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

    def __init__(self, config:EncoderConfig,
                 device:str='cpu'):
        """
        Constructor

        :param encoder_sizes: Number of features for each layer
        :param kernel_sizes: List of kernel sizes for each layer
        """

        super().__init__()
        bands = config.bands
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
                        padding=config.padding,
                    ),
                    nn.ReLU(),
                ]
                for in_layer, out_layer, kernel_size in zip(
                    config.layer_sizes, config.layer_sizes[1:], config.kernel_sizes
                )
            ],
            [],
        )
        self.cnet = nn.Sequential(*enc_blocks).to(device)
        self.nb_enc_cropped_hw = sum(
            [(kernel_size - 1) // 2 for kernel_size in config.kernel_sizes]
        )
        self.mu_conv = nn.Conv2d(config.layer_sizes[-1], config.layer_sizes[-1]//2,
                                 kernel_size=1).to(device)
        self.logvar_conv = nn.Conv2d(config.layer_sizes[-1], config.layer_sizes[-1]//2,
                                     kernel_size=1).to(device)
        self.mu_logvar_conv = nn.Conv2d(config.layer_sizes[-1], config.layer_sizes[-1],
                                        kernel_size=1).to(device)
        bands_loc = config.io_coeffs.bands.loc 
        idx_loc = config.io_coeffs.idx.loc 
        angles_loc = config.io_coeffs.angles.loc 
        bands_scale = config.io_coeffs.bands.scale 
        idx_scale = config.io_coeffs.idx.scale 
        angles_scale = config.io_coeffs.angles.scale 
        bands_loc = bands_loc if bands_loc is not None else torch.zeros((config.input_size))
        bands_scale = bands_scale if bands_scale is not None else torch.ones((config.input_size))
        idx_loc = idx_loc if idx_loc is not None else torch.zeros((5))
        idx_scale = idx_scale if idx_scale is not None else torch.ones((5))
        angles_loc = angles_loc if angles_loc is not None else torch.zeros((3))
        angles_scale = angles_scale if angles_scale is not None else torch.ones((3))
        self.bands_loc = bands_loc.float().to(device)
        self.bands_scale = bands_scale.float().to(device)
        self.idx_loc = idx_loc.float().to(device)
        self.idx_scale = idx_scale.float().to(device)
        self.angles_loc = angles_loc.float().to(device)
        self.angles_scale = angles_scale.float().to(device)
        self._spatial_encoding = True

    def get_spatial_encoding(self):
        """
        Return private attribute about wether the encoder takes patches as input
        """
        return self._spatial_encoding
    
    def encode(self, s2_r, angles):
        """
        Forward pass of the convolutionnal encoder

        :param x: Input tensor of shape [N,C_in,H,W]

        :return: Output Dataclass that holds mu and var
                 tensors of shape [N,C_out,H,W]
        """
        is_patch = check_is_patch(s2_r)
        if not is_patch:
            raise AttributeError("Input data is a not a patch: spatial encoder can only take patches as input")
        spectral_idx = get_spectral_idx(s2_refl, bands_dim=1)
        normed_refl = standardize(s2_r, self.bands_loc, self.bands_scale, dim=1)[:, self.bands, ...]
        if len(normed_refl.size())==3:
            normed_refl = normed_refl.unsqueeze(0)
        if len(angles.size())==3:
            angles = angles.unsqueeze(0)
        # normed_angles = standardize(torch.concat((torch.cos(torch.deg2rad(angles)), 
        #                                           torch.sin(torch.deg2rad(angles))), axis=1), 
        #                             self.angles_loc, self.angles_scale, dim=1)
        normed_angles = standardize(torch.cos(torch.deg2rad(angles)), self.angles_loc, 
                            self.angles_scale, dim=1)
        y = self.cnet(torch.concat((normed_refl, normed_angles, spectral_idx), axis=1))
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
        self.bands_loc = self.bands_loc.to(device)
        self.bands_scale = self.bands_scale.to(device)
        self.idx_loc = self.idx_loc.to(device)
        self.idx_scale = self.idx_scale.to(device)
        self.angles_loc = self.angles_loc.to(device)
        self.angles_scale = self.angles_scale.to(device)
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


class ProsailResCNNEncoder(nn.Module):
    """
    Implements an encoder with alternate
    convolutional and Relu layers
    """

    def __init__(self, config:EncoderConfig, device='cpu'):
        """
        Constructor
        """
        super().__init__()
        bands = config.bands
        if bands is None:
            bands = torch.arange(10)
        self.bands = bands.to(device)
        self.device = device
        network = []
        network.append(nn.Conv2d(config.input_size, config.first_layer_size,
                                 config.first_layer_kernel, padding=config.padding))
        network.append(nn.ReLU)
        input_sizes = [config.first_layer_size] + config.block_layer_sizes
        assert len(config.block_layer_sizes) == len(config.block_layer_depths)
        assert len(config.block_layer_depths) == len(config.block_kernel_sizes)
        assert len(config.block_kernel_sizes) == len(config.block_n)
        n_groups = len(config.block_n)
        for i in range(n_groups):
            for _ in range(config.block_n[i]):
                network.append(EncoderCResNetBlock(output_size=config.block_layer_sizes[i],
                                                   depth=config.block_layer_depths[i],
                                                   kernel_size=config.block_kernel_sizes[i],
                                                   input_size=input_sizes[i],
                                                   padding=config.padding))
                network.append(nn.ReLU())
        self.cnet = nn.Sequential(*network).to(device)
        self.mu_conv = nn.Conv2d(input_sizes[-1], config.output_size, kernel_size=1,
                                 padding=config.padding).to(device)
        self.logvar_conv = nn.Conv2d(input_sizes[-1], config.output_size, kernel_size=1,
                                     padding=config.padding).to(device)
        self.mu_logvar_conv = nn.Conv2d(input_sizes[-1],
                                        config.n_latent_params * config.output_size, 
                                        kernel_size=1, padding=config.padding).to(device)
        bands_loc = config.io_coeffs.bands.loc 
        idx_loc = config.io_coeffs.idx.loc 
        angles_loc = config.io_coeffs.angles.loc 
        bands_scale = config.io_coeffs.bands.scale 
        idx_scale = config.io_coeffs.idx.scale 
        angles_scale = config.io_coeffs.angles.scale 
        bands_loc = bands_loc if bands_loc is not None else torch.zeros((config.input_size))
        bands_scale = bands_scale if bands_scale is not None else torch.ones((config.input_size))
        idx_loc = idx_loc if idx_loc is not None else torch.zeros((5))
        idx_scale = idx_scale if idx_scale is not None else torch.ones((5))
        angles_loc = angles_loc if angles_loc is not None else torch.zeros((3))
        angles_scale = angles_scale if angles_scale is not None else torch.ones((3))
        self.bands_loc = bands_loc.float().to(device)
        self.bands_scale = bands_scale.float().to(device)
        self.idx_loc = idx_loc.float().to(device)
        self.idx_scale = idx_scale.float().to(device)
        self.angles_loc = angles_loc.float().to(device)
        self.angles_scale = angles_scale.float().to(device)

        self.nb_enc_cropped_hw = config.first_layer_kernel//2
        for i in range(n_groups):
            for _ in range(config.block_n[i]):
                for _ in range(config.block_layer_depths[i]):
                    self.nb_enc_cropped_hw += config.block_kernel_sizes[i]//2
        self._spatial_encoding = True


    def get_spatial_encoding(self):
        """
        Return private attribute about whether the encoder takes patches as input
        """
        return self._spatial_encoding
    
    def encode(self, s2_refl, angles):
        """
        Forward pass of the convolutionnal encoder

        :param x: Input tensor of shape [N,C_in,H,W]

        :return: Output Dataclass that holds mu and var
                 tensors of shape [N,C_out,H,W]
        """
        is_patch = check_is_patch(s2_refl)
        if not is_patch:
            raise AttributeError("Input data is a not a patch: spatial encoder can only take patches as input")
        spectral_idx = get_spectral_idx(s2_refl, bands_dim=1)
        normed_refl = standardize(s2_refl, self.bands_loc, self.bands_scale, dim=1)[:,self.bands,...]
        if len(normed_refl.size())==3:
            normed_refl = normed_refl.unsqueeze(0) # Ensures batch dimension appears
        if len(angles.size())==3:
            angles = angles.unsqueeze(0)
        # normed_angles = standardize(torch.concat((torch.cos(torch.deg2rad(angles)),
        #                                           torch.sin(torch.deg2rad(angles))), axis=1), 
        normed_angles = standardize(torch.cos(torch.deg2rad(angles)), self.angles_loc, 
                                    self.angles_scale, dim=1)
        y = self.cnet(torch.concat((normed_refl, normed_angles, spectral_idx), axis=1))
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
        self.bands_loc = self.bands_loc.to(device)
        self.bands_scale = self.bands_scale.to(device)
        self.idx_loc = self.idx_loc.to(device)
        self.idx_scale = self.idx_scale.to(device)
        self.angles_loc = self.angles_loc.to(device)
        self.angles_scale = self.angles_scale.to(device)
        self.cnet = self.cnet.to(device)
        self.mu_conv = self.mu_conv.to(device)
        self.logvar_conv = self.logvar_conv.to(device)
        self.mu_logvar_conv = self.mu_logvar_conv.to(device)
        self.bands = self.bands.to(device)


def get_encoder(config:EncoderConfig, device:str="cpu"):
    if config.encoder_type=='nn':
        encoder = ProsailNNEncoder(config, device)
    elif config.encoder_type=='rnn':
        encoder = ProsailRNNEncoder(config, device)
    elif config.encoder_type=='rcnn':
        # first_layer_kernel = 3
        # first_layer_size = 128
        # crnn_group_sizes = [128,128]
        # crnn_group_depth = [2,2]
        # crnn_group_kernel_sizes = [3,1]
        # crnn_group_n = [1,3]
        # # crnn_group_n = [1,2]
        encoder = ProsailResCNNEncoder(config, device)
    elif config.encoder_type=='cnn':
        # encoder_sizes = [vae_params["input_size"] + 2 * 3] + vae_params["hidden_layers_size"] + [output_size]
        # enc_kernel_sizes = [3] * (len(encoder_sizes) - 1)
        encoder = ProsailCNNEncoder(config, device)
    else:
        raise NotImplementedError
    return encoder
