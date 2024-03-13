#!/usr/bin/env python3
"""
Created on Wed Nov  9 13:39:15 2022

@author: yoel
"""

from pathlib import Path

import numpy as np
import prosail
import torch
from prosail.sail_model import init_prosail_spectra
from scipy.interpolate import interp1d
from scipy.signal import decimate

from .spectral_indices import get_spectral_idx

# from prosail import spectral_lib
from .utils.utils import gaussian_nll_loss, standardize, unstandardize


def subsample_spectra(tensor, R_down=1, axis=0, method="interp"):
    if R_down > 1:
        assert 2100 % R_down == 0
        if method == "block_mean":
            if tensor.size(0) == 2101:
                tensor = tensor[:-1].reshape(-1, R_down).mean(1)
        elif method == "decimate":
            device = tensor.device
            decimated_array = decimate(tensor.detach().cpu().numpy(), R_down).copy()
            tensor = torch.from_numpy(decimated_array).to(device)
        elif method == "interp":
            device = tensor.device
            f = interp1d(np.arange(400, 2501), tensor.detach().cpu().numpy())
            sampling = np.arange(400, 2501, R_down)
            array = np.apply_along_axis(f, 0, sampling)
            tensor = torch.from_numpy(array).float().to(device)
        else:
            raise NotImplementedError
    return tensor


PROSAILVARS = [
    "N",
    "cab",
    "car",
    "cbrown",
    "cw",
    "cm",
    "lai",
    "lidfa",
    "hspot",
    "psoil",
    "rsoil",
]
BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]


def get_bands_idx(bvnet_bands=False):
    """
    Outputs the index of bands to be taken into account in the reflectance tensor
    """
    bands = torch.arange(
        10
    )  # Bands are supposed to come in number order, not by resolution group
    prosail_bands = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12]

    if bvnet_bands:  # Removing B2 and B8
        print("Weiss bands enabled")
        bands = torch.tensor([1, 2, 3, 4, 5, 7, 8, 9])  # removing b2 and b8
        prosail_bands = [2, 3, 4, 5, 6, 8, 11, 12]
    return bands, prosail_bands


def apply_along_axis(function, x, fn_arg, axis: int = 0):
    return torch.stack(
        [function(x_i, fn_arg) for x_i in torch.unbind(x, dim=axis)], dim=axis
    )


def decimate_1Dtensor(tensor, R_down=1):
    device = tensor.device
    decimated_array = decimate(tensor.detach().cpu().numpy(), R_down).copy()
    return torch.from_numpy(decimated_array).to(device)


def sanitize_rsr_file_path(rsr_file: str) -> str:
    """Look for rsr file into the data folder if the provided file is not found"""
    if Path(rsr_file).is_file():
        return rsr_file
    local_file = Path(f"{__file__}").parent.parent.parent / "data" / rsr_file
    if local_file.is_file():
        return local_file
    raise FileNotFoundError(f"{rsr_file} not found")


class SensorSimulator:
    """Simulates the reflectances of a sensor from a full spectrum and the
    RSR of the sensor.

        The rsr_file is supposed to follow the 6S format: lambdas are given
        in micrometers and regularly sampled with a 0.001 step (1 nm). The
        input full spectrum can be a Prosail simulation and will be sampled
        with the same step. The wavelength ranges of the RSR are typically
        [0.350,2400] and Prosail gives [400, 2500]. The output will be given
        in [400, 2500] as the input.

        The RSR file columns are: lambda, solar spectrum and bands.
    """

    def __init__(
        self,
        rsr_file: str,
        prospect_range: tuple[int, int] = (400, 2500),
        #  bands=[1, 2, 3, 4, 5, 6, 7, 8, 9, 11],
        bands=None,
        device="cpu",
        bands_loc=None,
        bands_scale=None,
        idx_loc=None,
        idx_scale=None,
        apply_norm=True,
        R_down=1,
    ):
        super().__init__()
        if bands is None:
            bands = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12]
        self.R_down = R_down
        self.bands = bands
        self.device = device
        self.prospect_range = prospect_range
        self.rsr = torch.from_numpy(
            np.loadtxt(sanitize_rsr_file_path(rsr_file), unpack=True)
        ).to(device)
        self.nb_bands = self.rsr.shape[0] - 2
        self.rsr_range = (
            int(self.rsr[0, 0].item() * 1000),
            int(self.rsr[0, -1].item() * 1000),
        )
        self.nb_lambdas = prospect_range[1] - prospect_range[0] + 1
        # if self.R_down > 1:
        #     self.nb_lambdas = (prospect_range[1] - prospect_range[0]) // self.R_down
        self.rsr_prospect = torch.zeros([self.rsr.shape[0], self.nb_lambdas]).to(device)
        self.rsr_prospect[0, :] = torch.linspace(
            prospect_range[0], prospect_range[1], self.nb_lambdas
        ).to(device)
        self.rsr_prospect[
            1:, : -(self.prospect_range[1] - self.rsr_range[1])
        ] = self.rsr[1:, (self.prospect_range[0] - self.rsr_range[0]) :]

        self.solar = self.rsr_prospect[1, :].unsqueeze(0)
        self.rsr = self.rsr_prospect[2:, :].unsqueeze(0)
        self.rsr = self.rsr[:, bands, :]

        bands_loc = bands_loc if bands_loc is not None else torch.zeros(len(bands))
        bands_scale = bands_scale if bands_scale is not None else torch.ones(len(bands))
        idx_loc = idx_loc if idx_loc is not None else torch.zeros(5)
        idx_scale = idx_scale if idx_scale is not None else torch.ones(5)

        self.bands_loc = bands_loc.float().to(device)
        self.bands_scale = bands_scale.float().to(device)
        self.idx_loc = idx_loc.float().to(device)
        self.idx_scale = idx_scale.float().to(device)
        self.apply_norm = apply_norm

        self.s2norm_factor_d = (self.rsr * self.solar).sum(axis=2)
        self.s2norm_factor_n = self.rsr * self.solar
        if self.R_down > 1:
            self.s2norm_factor_n = (
                self.s2norm_factor_n[:, :, :-1]
                .reshape(1, len(bands), -1, R_down)
                .mean(3)
                * self.R_down
            )

    def change_device(self, device):
        self.device = device
        self.rsr = self.rsr.to(device)
        self.bands_loc = self.bands_loc.to(device)
        self.bands_scale = self.bands_scale.to(device)
        self.idx_loc = self.idx_loc.to(device)
        self.idx_scale = self.idx_scale.to(device)
        self.s2norm_factor_d = self.s2norm_factor_d.to(device)
        self.s2norm_factor_n = self.s2norm_factor_n.to(device)
        self.solar = self.solar.to(device)

        pass

    def __call__(self, prosail_output: torch.Tensor, apply_norm=None):
        return self.forward(prosail_output, apply_norm=apply_norm)

    def apply_s2_sensor(self, prosail_output: torch.Tensor) -> torch.Tensor:
        # The input should have shape = batch, wavelengths, otherwise,
        # we add the first dimension
        x = prosail_output
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        elif len(x.shape) == 2:
            x = x.unsqueeze(1)
        simu = (self.s2norm_factor_n * x).sum(axis=2) / self.s2norm_factor_d
        return simu

    def normalize(self, s2_r, bands_dim=1):
        return standardize(s2_r, self.bands_loc, self.bands_scale, dim=bands_dim)

    def unnormalize(self, s2_r, bands_dim=1):
        return unstandardize(s2_r, self.bands_loc, self.bands_scale, dim=bands_dim)

    def forward(self, prosail_output: torch.Tensor, apply_norm=None) -> torch.Tensor:
        simu = self.apply_s2_sensor(prosail_output)
        if apply_norm is None:  # using SensorSimulator apply_norm as default
            apply_norm = self.apply_norm
        if apply_norm:
            simu = self.normalize(simu)
        return simu  # type: ignore

    def index_loss(
        self,
        s2_r,
        s2_rec,
        lossfn=gaussian_nll_loss,
        normalize_idx=True,
        s2_r_bands_dim=1,
        rec_bands_dim=2,
    ):
        # u_s2_r = self.unnormalize(s2_r)
        if self.apply_norm:
            u_s2_rec = self.unnormalize(s2_rec, rec_bands_dim)
        else:
            u_s2_rec = s2_rec

        spectral_idx_tgt = get_spectral_idx(s2_r=s2_r, bands_dim=s2_r_bands_dim)
        spectral_idx_rec = get_spectral_idx(s2_r=u_s2_rec, bands_dim=rec_bands_dim)
        # spectral_idx_rec = spectral_idx_rec.mean(s2_r_bands_dim, keepdim=True)

        if normalize_idx:
            spectral_idx_tgt = standardize(
                spectral_idx_tgt, self.idx_loc, self.idx_scale, s2_r_bands_dim
            )
            spectral_idx_rec = standardize(
                spectral_idx_rec, self.idx_loc, self.idx_scale, rec_bands_dim
            )
        loss = lossfn(spectral_idx_tgt, spectral_idx_rec)

        return loss


class ProsailSimulator:
    def __init__(
        self,
        factor: str = "SDR",
        typelidf: int = 2,
        device="cpu",
        R_down: int = 1,
        prospect_version="5",
    ):
        super().__init__()
        self.factor = factor
        self.typelidf = typelidf
        self.device = device
        self.R_down = R_down
        self.prospect_version = prospect_version
        [
            self.soil_spectrum1,
            self.soil_spectrum2,
            self.nr,
            self.kab,
            self.kcar,
            self.kbrown,
            self.kw,
            self.km,
            self.kant,
            self.kprot,
            self.kcbc,
            self.lambdas,
        ] = init_prosail_spectra(
            R_down=self.R_down,
            device=self.device,
            prospect_version=self.prospect_version,
        )

    def __call__(self, params):
        return self.forward(params)

    def change_device(self, device):
        self.device = device
        self.soil_spectrum1 = self.soil_spectrum1.to(device)
        self.soil_spectrum2 = self.soil_spectrum2.to(device)
        self.nr = self.nr.to(device)
        self.kab = self.kab.to(device)
        self.kcar = self.kcar.to(device)
        self.kbrown = self.kbrown.to(device)
        self.kw = self.kw.to(device)
        self.km = self.km.to(device)
        self.kant = self.kant.to(device)
        self.kprot = self.kprot.to(device)
        self.kcbc = self.kcbc.to(device)
        self.lambdas = self.lambdas.to(device)
        pass

    def forward(self, params: torch.Tensor):
        # params.shape == [x] => single sample
        # params.shape == [x,y] => batch
        assert params.shape[-1] == 14, f"{params.shape[-1]}"
        if len(params.shape) == 1:
            params = params.unsqueeze(0)
        prosail_refl = prosail.run_prosail(
            N=params[..., 0].unsqueeze(
                -1
            ),  # change input (use dictionnary for instance)
            cab=params[..., 1].unsqueeze(-1),
            car=params[..., 2].unsqueeze(-1),
            cbrown=params[..., 3].unsqueeze(-1),
            cw=params[..., 4].unsqueeze(-1),
            cm=params[..., 5].unsqueeze(-1),
            lai=params[..., 6].unsqueeze(-1),
            lidfa=params[..., 7].unsqueeze(-1),
            hspot=params[..., 8].unsqueeze(-1),
            rsoil=params[..., 9].unsqueeze(-1),
            psoil=params[..., 10].unsqueeze(-1),
            tts=params[..., 11].unsqueeze(-1),
            tto=params[..., 12].unsqueeze(-1),
            psi=params[..., 13].unsqueeze(-1),
            typelidf=torch.as_tensor(self.typelidf),
            factor=self.factor,
            device=self.device,
            soil_spectrum1=self.soil_spectrum1,
            soil_spectrum2=self.soil_spectrum2,
            nr=self.nr,
            kab=self.kab,
            kcar=self.kcar,
            kbrown=self.kbrown,
            kw=self.kw,
            km=self.km,
            kant=self.kant,
            kprot=self.kprot,
            kcbc=self.kcbc,
            lambdas=self.lambdas,
            R_down=1,
            init_spectra=False,
            prospect_version=self.prospect_version,
        ).float()

        return prosail_refl
