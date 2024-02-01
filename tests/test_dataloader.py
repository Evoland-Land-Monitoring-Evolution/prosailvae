from pathlib import Path

from prosailvae.datamodules.DataModule import ProsailVAEDataModule

PATCHES_DIR = Path("/usr/local/stok/DATA/MMDC/ProsailVAE/PROSAILVAE/s2_patch_dataset/")


def test_instanciate_dm(bands: int = 10):
    dm = ProsailVAEDataModule(PATCHES_DIR, list(range(bands)))
    return dm


def test_get_dl():
    dm = ProsailVAEDataModule(PATCHES_DIR, list(range(10)))
    trdl = dm.train_dataloader()
    vdl = dm.validation_dataloader()
    tsdl = dm.test_dataloader()
    return (trdl, vdl, tsdl)


def test_get_data():
    dm = ProsailVAEDataModule(PATCHES_DIR, list(range(10)))
    trdl = dm.train_dataloader()
    for batch, _ in zip(trdl, range(10), strict=False):
        print(batch)
