from prosailvae.datamodules.data_module import DataModuleConfig, ProsailVAEDataModule

from .paths import PATCHES_DIR


def instanciate(bands: int = 10) -> ProsailVAEDataModule:
    cfg = DataModuleConfig(PATCHES_DIR, list(range(bands)))
    dm = ProsailVAEDataModule(cfg)
    return dm


def test_instanciate_dm(bands: int = 10):
    instanciate(bands)


def test_get_dl():
    cfg = DataModuleConfig(PATCHES_DIR, 10)
    dm = ProsailVAEDataModule(cfg)
    trdl = dm.train_dataloader()
    assert trdl is not None
    vdl = dm.val_dataloader()
    assert vdl is not None
    tsdl = dm.test_dataloader()
    assert tsdl is not None


def test_get_data():
    cfg = DataModuleConfig(PATCHES_DIR, 10)
    dm = ProsailVAEDataModule(cfg)
    trdl = dm.train_dataloader()
    for batch, _ in zip(trdl, range(10), strict=False):
        print(batch)
