"""Export the encoder of a PROSAIL-VAE lightning checkpoint to onnx format"""

import argparse
import os
from pathlib import Path

import onnxruntime
import torch

from prosailvae.models.inverse_prosail import (
    InverseProsail,
    OutputType,
    convert_to_inverse_prosail,
)

DATA_DIR = Path(f"{__file__}").parent.parent / "data"


def onnx_export(
    net: InverseProsail,
    output_dir: Path,
    data: tuple[torch.Tensor, torch.Tensor],
    res: torch.Tensor,
) -> None:
    """Export to TorchScript"""
    output_onnx_file = os.path.join(output_dir, "prosailvae.onnx")
    net.eval()

    # Export the model
    torch.onnx.export(
        net,  # model being run
        data,
        output_onnx_file,  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=11,  # the ONNX version to export the model to
        input_names=["refls", "angles"],  # the model's input names
        dynamic_axes={
            "refls": [0],  # variable length axes
            "angles": [0],
            "output": [0],
        },
    )

    sess_opt = onnxruntime.SessionOptions()
    sess_opt.intra_op_num_threads = 16
    ort_session = onnxruntime.InferenceSession(
        output_onnx_file,
        sess_opt,
        providers=["CPUExecutionProvider"],
    )
    input = {"refls": data[0].numpy(), "angles": data[1].numpy()}  # (B, N)
    # Check the ouput of the exported model
    torch_res = ort_session.run(None, input)[0]
    mus = res[:, :, :1]
    sigmas = res[:, :, 1:]
    print(
        torch.allclose(torch.tensor(torch_res[:, :, :1]), mus, rtol=1e-03, atol=1e-05)
    )
    print(
        torch.allclose(
            torch.tensor(torch_res[:, :, 1:]), sigmas, rtol=1e-03, atol=1e-05
        )
    )

    # Check that dynamic axes work well
    input = {
        "refls": torch.cat((data[0], data[0]), 0).numpy(),
        "angles": torch.cat((data[1], data[1]), 0).numpy(),
    }  # (B, N)
    torch_res = ort_session.run(None, input)[0]
    print(
        torch.allclose(torch.tensor(torch_res[:7, :, :1]), mus, rtol=1e-03, atol=1e-05)
    )
    print(
        torch.allclose(
            torch.tensor(torch_res[:7, :, 1:]), sigmas, rtol=1e-03, atol=1e-05
        )
    )


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
        # required=True,
        default="/work/scratch/data/kalinie/src/MMDC/checkpoint_pvae/last.ckpt",
    )
    parser.add_argument(
        "--config",
        "-cfg",
        type=str,
        help="Path to hydra config directory",
        # required=True,
        default="/work/scratch/data/kalinie/src/MMDC/checkpoint_pvae",
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
        "--output",
        type=str,
        help="Ouptut directory path",
        # required=True,
        default="./",
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
    onnx_export(net, output_dir, data, results)
