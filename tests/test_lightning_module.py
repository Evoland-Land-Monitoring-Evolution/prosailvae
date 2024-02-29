from pathlib import Path

import torch

from prosailvae.encoders import EncoderConfig
from prosailvae.loss import LossConfig
from prosailvae.models.lightning_module import ProsailVAELightningModule
from prosailvae.prosail_vae import ProsailVAEConfig
from prosailvae.utils.utils import load_standardize_coeffs

SRC_DIR = Path(__file__).parent.parent
PATCHES_DIR = Path("/usr/local/stok/DATA/MMDC/ProsailVAE/PROSAILVAE/s2_patch_dataset/")
RSR_DIR = SRC_DIR / "data"
N_PROSAIL_VARS = 11


def instanciate(bands: int = 10, lat_idx: int = 6):
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
    mod = ProsailVAELightningModule(pv_conf)
    return mod


def test_lightning_instanciate(bands: int = 10, lat_idx: int = 6):
    instanciate()
