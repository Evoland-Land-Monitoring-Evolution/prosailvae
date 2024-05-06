"""Export the encoder of a PROSAIL-VAE lightning checkpoint to Pytorch
and TorchScript formats"""

import argparse
from pathlib import Path

import torch

from prosailvae.models.inverse_prosail import (
    InverseProsail,
    OutputType,
    convert_to_inverse_prosail,
)

DATA_DIR = Path(f"{__file__}").parent.parent / "data"


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


def get_parser() -> argparse.ArgumentParser:
    """
    Generate argument parser for cli
    """
    parser = argparse.ArgumentParser(
        Path(__file__).name,
        description=(
            "Export PVAE encoder from lightning checkpoint"
            " to Pytorch and Torch Script"
        ),
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
        "--coeff_dir",
        type=str,
        help=(
            "Path to directory containing normalizing coefficients. "
            "Will use those provided with the module by default."
        ),
        required=False,
    )

    parser.add_argument(
        "--device",
        type=str,
        help="Device on which the model will be built (cpu | gpu)",
        default="cpu",
        required=False,
    )

    parser.add_argument(
        "--output", type=str, help="Ouptut directory path", required=True
    )
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    output_type: OutputType = "sim"
    ckpt_file = Path(args.checkpoint)
    config_dir = Path(args.config)
    coeff_dir = Path(DATA_DIR) if args.coeff_dir is None else Path(args.coeff_dir)
    output_dir = Path(args.output)

    net, results, data = convert_to_inverse_prosail(
        ckpt_file, config_dir, coeff_dir, output_type, args.device
    )

    torch_export(net, output_dir, data, results)
    torchscript_export(net, output_dir, data, results)
