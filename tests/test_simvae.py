from pathlib import Path

import torch

from prosailvae.encoders import EncoderConfig
from prosailvae.loss import LossConfig
from prosailvae.prosail_vae import ProsailVAEConfig, get_prosail_vae
from prosailvae.utils.utils import load_standardize_coeffs

from .paths import PATCHES_DIR

SRC_DIR = Path(__file__).parent.parent
RSR_DIR = SRC_DIR / "data"
N_PROSAIL_VARS = 11
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_config(bands: int = 10, lat_idx: list[int] | None = None):
    if lat_idx is None:
        lat_idx = [6]
    io_coeffs = load_standardize_coeffs(PATCHES_DIR)
    n_idx = io_coeffs.idx.loc.size(0) if io_coeffs.idx.loc is not None else 0
    enc_conf = EncoderConfig(
        encoder_type="rnn",
        input_size=bands + 3 + n_idx,
        io_coeffs=io_coeffs,
        output_size=N_PROSAIL_VARS,
    )
    loss_conf = LossConfig(lat_idx=torch.tensor(lat_idx).int())
    pv_conf = ProsailVAEConfig(enc_conf, loss_conf, RSR_DIR, "/tmp", "/tmp")
    return pv_conf


def test_instantiate():
    config = generate_config()
    model = get_prosail_vae(config, DEVICE)
    assert model is not None


def test_forward():
    batch_size = 2
    patch_size = 4
    config = generate_config()
    model = get_prosail_vae(config, DEVICE)
    data = torch.rand(batch_size, 10, patch_size, patch_size).to(DEVICE)
    angles = torch.rand(batch_size, 3, patch_size, patch_size).to(DEVICE)
    dist_params, z, phi, rec = model.forward(data, angles=angles, n_samples=3)
    rec.sum().backward()
