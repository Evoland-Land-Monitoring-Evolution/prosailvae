from dataclasses import dataclass, fields
from enum import Enum

import torch


def parse_tensor_to_dict(tensor, keys, dim):
    data_dict = {}
    assert len(keys) == tensor.size(dim)
    for index, key in enumerate(keys):
        data_dict[key] = tensor.select(dim, index)
    return data_dict


class S2Reflectance:
    def __init__(
        self, s2_r_tensor, input_bands, bands_dim, bands_loc, bands_scale
    ) -> None:
        self.bands_loc = bands_loc
        self.bands_scale = bands_scale
        self.bands_dim = bands_dim
        self.reflectance_data = parse_tensor_to_dict(s2_r_tensor, input_bands)
        pass

    def get_tensor(self, bands):
        pass

    class Band(Enum):
        """
        Enum class representing Sentinel2 spectral bands
        """

        B2 = "B2"
        B3 = "B3"
        B4 = "B4"
        B5 = "B5"
        B6 = "B6"
        B7 = "B7"
        B8 = "B8"
        B8A = "B8A"
        B9 = "B9"
        B10 = "B10"
        B11 = "B11"
        B12 = "B12"

    # Aliases
    B2 = Band.B2
    B3 = Band.B3
    B4 = Band.B4
    B5 = Band.B5
    B6 = Band.B6
    B7 = Band.B7
    B8 = Band.B8
    B8A = Band.B8A
    B9 = Band.B9
    B10 = Band.B10
    B11 = Band.B11
    B12 = Band.B12

    ALL = [B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12]
    SNAP = [B3, B4, B5, B6, B7, B8A, B11, B12]


@dataclass
class S2AData:
    SunZen: torch.Tensor
    ViewZen: torch.Tensor
    RelAzi: torch.Tensor


def main():
    pass


if __name__ == "__main__":
    main()
