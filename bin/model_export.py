"""Export the encoder of a PROSAIL-VAE lightning checkpoint to Pytorch, TorchScript
and ONNX formats"""

import argparse
from pathlib import Path
from typing import Literal, TypeAlias, cast

import hydra
import numpy as np
import onnx
import onnxruntime  # type: ignore
import torch
import torch.onnx

from prosailvae.encoders import ProsailResCNNEncoder, ProsailRNNEncoder
from prosailvae.latentspace import TruncatedNormalLatent
from prosailvae.simspaces import LinearVarSpace
from prosailvae.simvae import SimVAE
from prosailvae.utils.utils import load_standardize_coeffs

Encoder: TypeAlias = ProsailRNNEncoder | ProsailResCNNEncoder

DATA_DIR = Path(f"{__file__}").parent.parent / "data"
DEVICE = "cpu"
S2_BANDS = 10
S2_ANGLES = 3


def retrieve_modules(
    ckpt_file: Path, config_dir: Path
) -> tuple[Encoder, TruncatedNormalLatent, LinearVarSpace]:
    """Get the Encoder, LatentSpace (statistical) and VarSpace (physical) from a PVAE
    lightnig checkpoint"""
    checkpoint = torch.load(ckpt_file, map_location=torch.device(DEVICE))
    io_coeffs = load_standardize_coeffs(str(DATA_DIR))

    encoder_state_dict = {
        k.split(".", maxsplit=2)[-1]: v
        for k, v in checkpoint["state_dict"].items()
        if k.startswith("model.encoder")
    }

    with hydra.initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        config = hydra.compose(config_name="config.yaml")
        config.model.model.config.device = DEVICE
        config.model.model.config.decoder.ssimulator.device = DEVICE
        config.model.model.config.decoder.prosailsimulator.device = DEVICE
        config.model.model.config.lat_space.device = DEVICE
        config.model.model.config.sim_space.device = DEVICE

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


class InverseProsail(torch.nn.Module):
    """Pre-trained inverse PROSAIL model"""

    def __init__(
        self,
        encoder: Encoder,
        latent_space: TruncatedNormalLatent,
        sim_space: LinearVarSpace,
        output_type: Literal["z", "sim"] = "sim",
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


def torch_export(
    net: InverseProsail,
    output_dir: Path,
    data: tuple[torch.Tensor, torch.Tensor],
    res: torch.Tensor,
) -> None:
    """Export to native Pytorch"""
    output_torch_file = output_dir / f"{ckpt_file.stem}.pth"
    net.eval()
    torch.save(net, output_torch_file)
    # Check the ouput of the exported model
    torch_net = torch.load(output_torch_file)
    torch_net.eval()
    refls, angles = data
    torch_res = torch_net(refls, angles)
    mus = res[:, :, :1]
    sigmas = res[:, :, 1:]
    torch.allclose(torch_res[:, :, :1], mus, rtol=1e-03, atol=1e-05)
    torch.allclose(torch_res[:, :, 1:], sigmas, rtol=1e-03, atol=1e-05)


def torchscript_export(
    net: InverseProsail,
    output_dir: Path,
    data: tuple[torch.Tensor, torch.Tensor],
    res: torch.Tensor,
) -> None:
    """Export to TorchScript"""
    output_torchscript_file = output_dir / f"{ckpt_file.stem}.torchscript"
    net.eval()
    script = torch.jit.trace(net, data)
    script.save(output_torchscript_file)
    # Check the ouput of the exported model
    ts_net = torch.jit.load(output_torchscript_file)
    ts_net.eval()
    refls, angles = data
    torch_res = ts_net(refls, angles)
    mus = res[:, :, :1]
    sigmas = res[:, :, 1:]
    torch.allclose(torch_res[:, :, :1], mus, rtol=1e-03, atol=1e-05)
    torch.allclose(torch_res[:, :, 1:], sigmas, rtol=1e-03, atol=1e-05)


# TODO: test when onnx will support aten::deg2rad
def onnx_export(
    net: InverseProsail,
    output_dir: Path,
    data: tuple[torch.Tensor, torch.Tensor],
    res: torch.Tensor,
) -> None:
    """Export to ONNX"""
    output_onnx_file = output_dir / f"{ckpt_file.stem}.onnx"
    net.eval()
    refls, angles = data
    torch.onnx.export(
        net,
        data,
        output_onnx_file,
        export_params=True,
        opset_version=10,
        do_constant_folding=False,
        input_names=["refls", "angles"],
        output_names=["output"],
        dynamic_axes={
            "refls": [0],  # variable length axes
            "angles": [0],
            "output": [0],
        },
    )
    # Check the ouput of the exported model
    onnx_net = onnx.load(output_onnx_file)
    onnx.checker.check_model(onnx_net)
    ort_session = onnxruntime.InferenceSession(
        output_onnx_file, providers=["CPUExecutionProvider"]
    )
    ort_inputs = {
        ort_session.get_inputs()[0].name: refls.numpy(),
        ort_session.get_inputs()[1].name: angles.numpy(),
    }
    ort_ouputs = ort_session.run(["output"], ort_inputs)

    mus = res[:, :, :1].numpy()
    sigmas = res[:, :, 1:].numpy()
    np.testing.assert_allclose(mus, ort_ouputs[:, :, :1], rtol=1e-03, atol=1e-05)
    np.testing.assert_allclose(sigmas, ort_ouputs[:, :, 1:], rtol=1e-03, atol=1e-05)


def convert_to_inverse_prosail(
    ckpt_file: Path, config_dir: Path, output_type: Literal["z", "sim"]
) -> tuple[InverseProsail, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    encoder, latent_space, sim_space = retrieve_modules(ckpt_file, config_dir)
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


def get_parser() -> argparse.ArgumentParser:
    """
    Generate argument parser for cli
    """
    parser = argparse.ArgumentParser(
        Path(__file__).name,
        description="Export PVAE encoder from lightning checkpoint to Pytorch and onnx",
    )

    parser.add_argument(
        "--checkpoint",
        "-cp",
        type=str,
        help="Path to lightning checkpoint",
        required=True,
    )
    parser.add_argument(
        "--config",
        "-cfg",
        type=str,
        help="Path to hydra config directory",
        required=True,
    )

    parser.add_argument(
        "--output", type=str, help="Ouptut directory path", required=True
    )
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    output_type: Literal["z", "sim"] = "sim"
    ckpt_file = Path(args.checkpoint)
    config_dir = Path(args.config)
    output_dir = Path(args.output)

    net, results, data = convert_to_inverse_prosail(ckpt_file, config_dir, output_type)

    torch_export(net, output_dir, data, results)
    torchscript_export(net, output_dir, data, results)
