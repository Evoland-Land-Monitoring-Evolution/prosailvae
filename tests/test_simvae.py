from pathlib import Path

import torch

from prosailvae.encoders import EncoderConfig
from prosailvae.loss import LossConfig
from prosailvae.prosail_vae import ProsailVAEConfig, get_prosail_vae
from prosailvae.utils.utils import load_standardize_coeffs
import os
from .paths import PATCHES_DIR
from einops import rearrange
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

def test_regression_pvae_method(): 
    batch_size = 2
    patch_size = 4
    config = generate_config()
    model = get_prosail_vae(config, DEVICE)
    data = torch.rand(batch_size, 10, patch_size, patch_size).to(DEVICE)
    angles = torch.rand(batch_size, 3, patch_size, patch_size).to(DEVICE)
    dist_params_reg, z_reg, phi_reg, rec_reg = model.forward(data, angles=angles, n_samples=3)
    dist_params, z, phi, rec = model.pvae_method([data, angles] , n_samples=3)
    z_reg = rearrange(z_reg, 'batch samples bands w h -> (batch  w  h) bands samples').to(DEVICE)
    rec_reg = rearrange( rec_reg, 'batch samples bands w h -> (batch  w  h) bands samples').to(DEVICE)
    assert torch.eq(dist_params_reg, dist_params).all()
    assert torch.isclose(rec_reg, rec,rtol =5).all()

def test_pvae_loss(): 
    batch_size = 2
    patch_size = 4
    config = generate_config()
    model = get_prosail_vae(config, DEVICE)
    data = torch.rand(batch_size, 10, patch_size, patch_size).to(DEVICE)
    angles = torch.rand(batch_size, 3, patch_size, patch_size).to(DEVICE)
    dist_params, z, phi, rec = model.pvae_method([data, angles] , n_samples=20)
    dic= {}
    loss_sum, normalized_loss_dict = model.unsupervised_batch_loss([data,angles], dic, n_samples=20 )
    kl_loss = model.pvae_kl_term( data, angles, dist_params)
    assert kl_loss is not None 
    rec_loss = model.pvae_reconstruction_loss( rearrange( data, 'b bands p p2 -> (b  p  p2) bands'), angles, z, rec)
    assert torch.isclose(rec_loss.to(DEVICE),torch.tensor(normalized_loss_dict['rec_loss']).to(DEVICE),rtol=1e4 )