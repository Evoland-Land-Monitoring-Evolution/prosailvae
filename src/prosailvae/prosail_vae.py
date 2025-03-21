#!/usr/bin/env python3
"""
Created on Tue Nov  8 14:45:12 2022

@author: yoel
"""
from dataclasses import dataclass, field
from pathlib import Path

import torch

from .decoders import ProsailSimulatorDecoder
from .encoders import EncoderConfig, get_encoder

# from prosailvae.decoders import TSSimulatorDecoder
from .latentspace import TruncatedNormalLatent
from .loss import LossConfig, NLLLoss
from .ProsailSimus import ProsailSimulator, SensorSimulator
from .simspaces import LinearVarSpace
from .simvae import SimVAE, SimVAEConfig
from .utils.utils import load_dict


def load_params(config_dir, config_file, parser=None):
    """
    Load parameter dict form prosail vae and training, with default options
    """
    params = load_dict(config_dir + config_file)
    if params["supervised"]:
        params["simulated_dataset"] = True
    if "load_model" not in params.keys():
        params["load_model"] = False
    if "vae_load_dir_path" not in params.keys():
        params["vae_load_dir_path"] = None
    else:
        params["vae_load_dir_path"] = f"{Path( __file__).parent.absolute()}"
        f"/../../trained_models/{params['vae_load_dir_path']}"
    if "lr_recompute_mode" not in params.keys():
        params["lr_recompute_mode"] = False
    if "init_model" not in params.keys():
        params["init_model"] = False
    if parser is not None:
        params["k_fold"] = parser.n_xp
        params["n_fold"] = parser.n_fold if params["k_fold"] > 1 else None
    if "layer_sizes" not in params.keys():
        params["layer_sizes"] = [512, 512]
    if "kernel_sizes" not in params.keys():
        params["kernel_sizes"] = [3, 3]
    if "first_layer_kernel" not in params.keys():
        params["first_layer_kernel"] = 3
    if "first_layer_size" not in params.keys():
        params["first_layer_size"] = 128
    if "block_layer_sizes" not in params.keys():
        params["block_layer_sizes"] = [128, 128]
    if "block_layer_depths" not in params.keys():
        params["block_layer_depths"] = [2, 2]
    if "block_kernel_sizes" not in params.keys():
        params["block_kernel_sizes"] = [3, 1]
    if "block_n" not in params.keys():
        params["block_n"] = [1, 3]
    if "supervised_kl" not in params.keys():
        params["supervised_kl"] = False
    if "bvnet_bands" not in params.keys():
        params["bvnet_bands"] = False
    params["vae_save_file_path"] = None
    if "supervised_config_file" not in params.keys():
        params["supervised_config_file"] = None
    if "supervised_weight_file" not in params.keys():
        params["supervised_weight_file"] = None
    if "disabled_latent" not in params.keys():
        params["disabled_latent"] = []
    if "disabled_latent_values" not in params.keys():
        params["disabled_latent_values"] = []
    if "cycle_training" not in params.keys():
        params["cycle_training"] = False
    if "R_down" not in params.keys():
        params["R_down"] = 1
    if "n_init_models" not in params.keys():
        params["n_init_models"] = 10
    if "n_init_epochs" not in params.keys():
        params["n_init_epochs"] = 10
    if "init_lr" not in params.keys():
        params["init_lr"] = 5e-4
    if "break_init_at_rec_loss" not in params.keys():
        params["break_init_at_rec_loss"] = None
    if "rec_bands_loss_coeffs" not in params.keys():
        params["rec_bands_loss_coeffs"] = None
    if "deterministic" not in params.keys():
        params["deterministic"] = False
    if "accum_iter" not in params.keys():
        params["accum_iter"] = 1
    if "beta_cyclical" not in params.keys():
        params["beta_cyclical"] = 0
    if "snap_cyclical" not in params.keys():
        params["snap_cyclical"] = False
    if "lat_loss_type" not in params.keys():
        params["lat_loss_type"] = ""
    if "lrs_threshold" not in params.keys():
        params["lrs_threshold"] = 5e-3
    if "validation_at_every_epoch" not in params.keys():
        params["validation_at_every_epoch"] = None
    if "prosail_vars_dist_type" not in params.keys():
        params["prosail_vars_dist_type"] = "legacy"
    if "lat_idx" not in params.keys():
        params["lat_idx"] = []
    if "prospect_version" not in params.keys():
        params["prospect_version"] = "5"
    if "frm4veg_data_dir" not in params.keys():
        params[
            "frm4veg_data_dir"
        ] = "/work/scratch/zerahy/prosailvae/data/frm4veg_validation"
    if "frm4veg_2021_data_dir" not in params.keys():
        params[
            "frm4veg_2021_data_dir"
        ] = "/work/scratch/zerahy/prosailvae/data/frm4veg_2021_validation"
    if "belsar_dir" not in params.keys():
        params["belsar_dir"] = "/work/scratch/zerahy/prosailvae/data/belSAR_validation"
    if "cyclical_data_dir" not in params.keys():
        params[
            "cyclical_data_dir"
        ] = "/work/scratch/zerahy/prosailvae/data/projected_data"
    return params


@dataclass
class ProsailVAEConfig:
    """
    Dataclass to hold all of PROSAIL_VAE configurations
    """

    encoder_config: EncoderConfig
    loss_config: LossConfig
    rsr_dir: Path | str
    vae_load_file_path: str
    vae_save_file_path: str
    spatial_mode: bool = False
    load_vae: bool = False
    apply_norm_rec: bool = True
    inference_mode: bool = False
    prosail_bands: list[int] = field(
        default_factory=lambda: [1, 2, 3, 4, 5, 6, 7, 8, 11, 12]
    )
    disabled_latent: list[int] = field(default_factory=lambda: [])
    disabled_latent_values: list[int] = field(default_factory=lambda: [])
    R_down: int = 1
    deterministic: bool = False
    prosail_vars_dist_type: str = "legacy"
    prospect_version: str = "5"

    def __post_init__(self):
        self.rsr_dir = Path(self.rsr_dir)


def get_prosail_vae(
    pv_config: ProsailVAEConfig,
    device: torch.device | str = "cpu",
    logger_name: str = "",
    hyper_prior: SimVAE | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    load_simulator=True,
    freeze_weights=False,
):
    """
    Intializes an instance of prosail_vae
    """
    encoder = get_encoder(pv_config.encoder_config, device="cpu")
    if hyper_prior is not None:
        kl_type = "tntn"
    else:
        kl_type = "tnu"
    lat_space = TruncatedNormalLatent(
        device="cpu",
        latent_dim=pv_config.encoder_config.output_size,
        kl_type=kl_type,
        disabled_latent=pv_config.disabled_latent,
        disabled_latent_values=pv_config.disabled_latent_values,
    )

    reconstruction_loss = NLLLoss(
        loss_type=pv_config.loss_config.loss_type,
        feature_indexes=pv_config.loss_config.reconstruction_bands_coeffs,
    )

    prosail_var_space = LinearVarSpace(
        latent_dim=pv_config.encoder_config.output_size,
        #  z2sim_mat=z2sim_mat,
        #  z2sim_offset=z2sim_offset,
        #  sim_pdf_support_span=sim_pdf_support_span,
        device="cpu",
        var_bounds_type=pv_config.prosail_vars_dist_type,
    )
    psimulator = ProsailSimulator(
        device="cpu",
        R_down=pv_config.R_down,
        prospect_version=pv_config.prospect_version,
    )
    if load_simulator:
        ssimulator = SensorSimulator(
            pv_config.rsr_dir / "sentinel2.rsr",
            device="cpu",
            bands_loc=pv_config.encoder_config.io_coeffs.bands.loc,
            bands_scale=pv_config.encoder_config.io_coeffs.bands.scale,
            idx_loc=pv_config.encoder_config.io_coeffs.idx.loc,
            idx_scale=pv_config.encoder_config.io_coeffs.idx.scale,
            apply_norm=pv_config.apply_norm_rec,
            bands=pv_config.prosail_bands,
            R_down=pv_config.R_down,
        )
    else:
        ssimulator = SensorSimulator(
            pv_config.rsr_dir / "sentinel2.rsr",
            device="cpu",
            bands_loc=None,
            bands_scale=None,
            apply_norm=pv_config.apply_norm_rec,
            bands=pv_config.prosail_bands,
            R_down=pv_config.R_down,
        )

    decoder = ProsailSimulatorDecoder(
        prosailsimulator=psimulator,
        ssimulator=ssimulator,
        loss_type=pv_config.loss_config.loss_type,
    )

    prosail_vae = SimVAE(
        SimVAEConfig(
            encoder=encoder,
            decoder=decoder,
            lat_space=lat_space,
            sim_space=prosail_var_space,
            deterministic=pv_config.deterministic,
            reconstruction_loss=reconstruction_loss,
            supervised=pv_config.loss_config.supervised,
            device="cpu",
            beta_kl=pv_config.loss_config.beta_kl,
            beta_index=pv_config.loss_config.beta_index,
            beta_cyclical=pv_config.loss_config.beta_cyclical,
            snap_cyclical=pv_config.loss_config.snap_cyclical,
            logger_name=logger_name,
            inference_mode=pv_config.inference_mode,
            lat_nll=pv_config.loss_config.lat_loss_type,
            lat_idx=pv_config.loss_config.lat_idx,
        )
    )
    prosail_vae.set_hyper_prior(hyper_prior)
    if pv_config.load_vae is not None and pv_config.vae_load_file_path is not None:
        if pv_config.load_vae:
            _, _ = prosail_vae.load_ae(pv_config.vae_load_file_path, optimizer)

    prosail_vae.change_device(device)
    if freeze_weights:
        prosail_vae.freeze_weigths()
    return prosail_vae


def load_prosail_vae_with_hyperprior(
    logger_name: str,
    pv_config: ProsailVAEConfig,
    pv_config_hyper: ProsailVAEConfig | None = None,
):
    """
    Loads prosail vae with or without intializing weight, with optionnal hyperprior
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hyper_prior = None
    if pv_config_hyper is not None:
        hyper_prior = get_prosail_vae(
            pv_config_hyper,
            device=device,
            logger_name=logger_name,
            load_simulator=False,
            freeze_weights=True,
        )

    prosail_vae = get_prosail_vae(
        pv_config, device=device, logger_name=logger_name, hyper_prior=hyper_prior
    )
    return prosail_vae
