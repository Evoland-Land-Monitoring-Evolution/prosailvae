from prosailvae.models.lightning_module import ProsailVAELightningModule

from .test_simvae import generate_config


def instanciate(bands: int = 10, lat_idx: int = 6) -> ProsailVAELightningModule:
    pv_conf = generate_config()
    mod = ProsailVAELightningModule(pv_conf)
    return mod


def test_lightning_instanciate(bands: int = 10, lat_idx: int = 6) -> None:
    module = instanciate(bands, lat_idx)
    assert module is not None
