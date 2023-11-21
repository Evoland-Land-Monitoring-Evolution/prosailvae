from dataclasses import dataclass

import torch


@dataclass
class S2R:
    """
    Data class to hold Sentinel-2 Bands
    """

    b02: torch.Tensor | None = None
    b03: torch.Tensor | None = None
    b04: torch.Tensor | None = None
    b05: torch.Tensor | None = None
    b06: torch.Tensor | None = None
    b07: torch.Tensor | None = None
    b08: torch.Tensor | None = None
    b8a: torch.Tensor | None = None
    b11: torch.Tensor | None = None
    b12: torch.Tensor | None = None


@dataclass
class SimVAEForwardResults:
    """
    Data class to hold the output of a forward pass of simvae
    """

    lat_dist_params: torch.Tensor
    lat_sample: torch.Tensor
    sim_variables: torch.Tensor
    reconstruction: torch.Tensor
    lat_hyperprior_params: torch.Tensor | None = None


@dataclass
class PROSAILVAEInputs:
    """
    Dataclass to hold sentinel reflectances and angles to be encoded
    """

    s2_r: torch.Tensor
    s2_a: torch.Tensor


@dataclass
class SimVAEInputs:
    """
    Data class to hold the output of a forward pass of simvae
    """

    encoder_input: torch.Tensor
    s2_a: torch.Tensor
    ref_sim_variables: torch.Tensor | None = None
