from pathlib import Path

from prosailvae.datamodules.data_module import DataModuleConfig, ProsailVAEDataModule

PATCHES_DIR = Path("/usr/local/stok/DATA/MMDC/ProsailVAE/PROSAILVAE/s2_patch_dataset/")


def test_instanciate_dm(bands: int = 10):
    cfg = DataModuleConfig(PATCHES_DIR, list(range(bands)))
    dm = ProsailVAEDataModule(cfg)
    return dm


def test_get_dl():
    cfg = DataModuleConfig(PATCHES_DIR, 10)
    dm = ProsailVAEDataModule(cfg)
    trdl = dm.train_dataloader()
    vdl = dm.validation_dataloader()
    tsdl = dm.test_dataloader()
    return (trdl, vdl, tsdl)


def test_get_data():
    cfg = DataModuleConfig(PATCHES_DIR, 10)
    dm = ProsailVAEDataModule(cfg)
    trdl = dm.train_dataloader()
    for batch, _ in zip(trdl, range(10), strict=False):
        print(batch)
