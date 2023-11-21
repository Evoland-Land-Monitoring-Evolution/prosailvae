from dataclasses import dataclass

import torch


@dataclass
class ViewAngles:
    sunzen: torch.Tensor
    obszen: torch.Tensor
    relazi: torch.Tensor


@dataclass
class S2Bands:
    B2: torch.Tensor
    B3: torch.Tensor
    B4: torch.Tensor
    B5: torch.Tensor
    B6: torch.Tensor
    B7: torch.Tensor
    B8: torch.Tensor
    B8A: torch.Tensor
    B11: torch.Tensor
    B12: torch.Tensor
