from pathlib import Path
from typing import Literal, TypeAlias, cast

import hydra
import torch

from prosailvae.encoders import ProsailResCNNEncoder, ProsailRNNEncoder
from prosailvae.latentspace import TruncatedNormalLatent
from prosailvae.simspaces import LinearVarSpace
from prosailvae.simvae import SimVAE
from prosailvae.utils.utils import load_standardize_coeffs

Encoder: TypeAlias = ProsailRNNEncoder | ProsailResCNNEncoder
OutputType = Literal["z", "sim"]
DeviceType: TypeAlias = str | torch.types._device

S2_BANDS = 10
S2_ANGLES = 3


class InverseProsail(torch.nn.Module):
    """Pre-trained inverse PROSAIL model"""

    def __init__(
        self,
        encoder: Encoder,
        latent_space: TruncatedNormalLatent,
        sim_space: LinearVarSpace,
        output_type: OutputType = "sim",
    ):
        super().__init__()
        self.encoder = encoder
        self.latent_space = latent_space
        self.sim_space = sim_space
        self.output_type = output_type

    def forward(self, refls: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        y, _ = self.encoder.encode(refls, angles)
        dist_params: torch.Tensor = self.latent_space.get_params_from_encoder(y)
        if self.output_type == "z":
            return dist_params
        res = torch.cat(
            [
                self.sim_space.z2sim(dist_params[:, :, :1]),
                self.sim_space.z2sim(dist_params[:, :, 1:]),
            ],
            dim=-1,
        )
        return res


def retrieve_modules(
    ckpt_file: Path,
    config_dir: Path,
    coeff_dir: Path,
    device: DeviceType = "cpu",
) -> tuple[Encoder, TruncatedNormalLatent, LinearVarSpace]:
    """Get the Encoder, LatentSpace (statistical) and VarSpace (physical) from a PVAE
    lightnig checkpoint"""
    checkpoint = torch.load(ckpt_file, map_location=torch.device(device))
    io_coeffs = load_standardize_coeffs(str(coeff_dir))

    encoder_state_dict = {
        k.split(".", maxsplit=2)[-1]: v
        for k, v in checkpoint["state_dict"].items()
        if k.startswith("model.encoder")
    }

    with hydra.initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        config = hydra.compose(config_name="config.yaml")
        config.model.model.config.device = device
        config.model.model.config.decoder.ssimulator.device = device
        config.model.model.config.decoder.prosailsimulator.device = device
        config.model.model.config.lat_space.device = device
        config.model.model.config.sim_space.device = device

        simvae: SimVAE = hydra.utils.instantiate(config.model.model.config)

        encoder = cast(Encoder, simvae.encoder)
        encoder.load_state_dict(encoder_state_dict)
        encoder.bands_loc = io_coeffs.bands.loc
        encoder.bands_scale = io_coeffs.bands.scale
        encoder.idx_loc = io_coeffs.idx.loc
        encoder.idx_scale = io_coeffs.idx.scale
        encoder.angles_loc = io_coeffs.angles.loc
        encoder.angles_scale = io_coeffs.angles.scale

        latent_space = cast(TruncatedNormalLatent, simvae.lat_space)
        sim_space = cast(LinearVarSpace, simvae.sim_space)

    return encoder, latent_space, sim_space


def convert_to_inverse_prosail(
    ckpt_file: Path,
    config_dir: Path,
    coeff_dir: Path,
    output_type: Literal["z", "sim"],
    device: DeviceType = "cpu",
) -> tuple[InverseProsail, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    encoder, latent_space, sim_space = retrieve_modules(
        ckpt_file, config_dir, coeff_dir, device
    )
    encoder.eval()
    latent_space.eval()
    sim_space.eval()

    with torch.no_grad():
        batch_size = 7
        angles = torch.randn((batch_size, S2_ANGLES))
        refls = torch.randint(0, 100, (batch_size, S2_BANDS)) / 100
        y, angles_out = encoder.encode(refls, angles)
        dist_params = latent_space.get_params_from_encoder(y)

    print(f"{dist_params=}")
    print(f"{dist_params.shape=}")

    mus = dist_params[:, :, :1]
    sigmas = dist_params[:, :, 1:]

    if output_type == "sim":
        mus = sim_space.z2sim(mus)
        sigmas = sim_space.z2sim(sigmas)

    net = InverseProsail(encoder, latent_space, sim_space, output_type=output_type)
    res = net(refls, angles)

    torch.allclose(res[:, :, :1], mus, rtol=1e-03, atol=1e-05)
    torch.allclose(res[:, :, 1:], sigmas, rtol=1e-03, atol=1e-05)

    return net, res, (refls, angles)
