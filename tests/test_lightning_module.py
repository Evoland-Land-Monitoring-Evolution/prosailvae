from prosailvae.models.LightningModule import ProsailVAELightningModule
from prosailvae.encoders import EncoderConfig
from prosailvae.prosail_vae import ProsailVAEConfig
from prosailvae.loss import LossConfig
from prosailvae.utils.utils import load_standardize_coeffs
from pathlib import Path

PATCHES_DIR = Path("/usr/local/stok/DATA/MMDC/ProsailVAE/PROSAILVAE/s2_patch_dataset/")
RSR_DIR = "/home/inglada/Dev/MMDC/prosailvae/data/"


def test_lightning_instanciate():
    io_coefs = load_standardize_coeffs(PATCHES_DIR)
    enc_conf = EncoderConfig(io_coefs)
    loss_conf = LossConfig()
    pv_conf = ProsailVAEConfig(enc_conf, loss_conf, RSR_DIR, "/tmp", "/tmp")
    mod = ProsailVAELightningModule(pv_conf)
    return mod
