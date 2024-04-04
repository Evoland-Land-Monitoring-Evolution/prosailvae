import torch

from prosailvae.decoders import ProsailSimulatorDecoder
from prosailvae.encoders import ProsailRNNEncoder
from prosailvae.latentspace import TruncatedNormalLatent
from prosailvae.loss import NLLLoss
from prosailvae.models.lightning_module import ProsailVAELightningModule
from prosailvae.ProsailSimus import ProsailSimulator, SensorSimulator
from prosailvae.simspaces import LinearVarSpace
from prosailvae.simvae import SimVAE, SimVAEConfig

from .test_simvae import generate_config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def instanciate(
    bands: int = 10, lat_idx: list[int] | None = None
) -> ProsailVAELightningModule:
    if lat_idx is None:
        lat_idx = [6]
    pv_conf = generate_config(bands, lat_idx)
    encoder = ProsailRNNEncoder(pv_conf.encoder_config)
    lat_space = TruncatedNormalLatent(
        device=DEVICE,
        latent_dim=pv_conf.encoder_config.output_size,
        kl_type="tnu",
        disabled_latent=pv_conf.disabled_latent,
        disabled_latent_values=pv_conf.disabled_latent_values,
    )
    reconstruction_loss = NLLLoss(
        loss_type=pv_conf.loss_config.loss_type,
        feature_indexes=pv_conf.loss_config.reconstruction_bands_coeffs,
    )
    prosail_var_space = LinearVarSpace(
        latent_dim=pv_conf.encoder_config.output_size,
        device=DEVICE,
        var_bounds_type=pv_conf.prosail_vars_dist_type,
    )
    psimulator = ProsailSimulator(
        device=DEVICE,
        R_down=pv_conf.R_down,
        prospect_version=pv_conf.prospect_version,
    )
    ssimulator = SensorSimulator(
        pv_conf.rsr_dir / "sentinel2.rsr",
        device=DEVICE,
        bands_loc=None,
        bands_scale=None,
        apply_norm=pv_conf.apply_norm_rec,
        bands=pv_conf.prosail_bands,
        R_down=pv_conf.R_down,
    )

    decoder = ProsailSimulatorDecoder(
        prosailsimulator=psimulator,
        ssimulator=ssimulator,
        loss_type=pv_conf.loss_config.loss_type,
    )

    model = SimVAE(
        SimVAEConfig(
            encoder=encoder,
            decoder=decoder,
            lat_space=lat_space,
            sim_space=prosail_var_space,
            deterministic=pv_conf.deterministic,
            reconstruction_loss=reconstruction_loss,
            supervised=pv_conf.loss_config.supervised,
            device=DEVICE,
            beta_kl=pv_conf.loss_config.beta_kl,
            beta_index=pv_conf.loss_config.beta_index,
            beta_cyclical=pv_conf.loss_config.beta_cyclical,
            snap_cyclical=pv_conf.loss_config.snap_cyclical,
            logger_name="",
            inference_mode=pv_conf.inference_mode,
            lat_nll=pv_conf.loss_config.lat_loss_type,
            lat_idx=pv_conf.loss_config.lat_idx,
        )
    )
    mod = ProsailVAELightningModule(model)
    return mod


def test_lightning_instanciate(
    bands: int = 10, lat_idx: list[int] | None = None
) -> None:
    if lat_idx is None:
        lat_idx = [6]
    module = instanciate(bands, lat_idx)
    assert module is not None
