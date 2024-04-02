from pathlib import Path

import pytest
from mmdc_singledate.datamodules.mmdc_datamodule import (
    MMDCDataLoaderConfig,
    MMDCDataModule,
    MMDCDataPaths,
)
from torch.utils.data import DataLoader

from prosailvae.datamodules.data_module import DataModuleConfig, ProsailVAEDataModule
from prosailvae.datamodules.mmdc_interface import mmdc2pvae_batch

from .paths import (
    MMDC_DATA_COMPONENTS,
    MMDC_DATASET_DIR,
    MMDC_TILES_CONFIG_DIR,
    PATCHES_DIR,
)


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
        print(f"{len(batch)=}")
        print(f"{batch[0].shape=}")
        print(f"{batch[1].shape=}")


def build_datamodule(
    max_open_files=1,
    batch_size_train=1,
    batch_size_val=1,
    num_workers=1,
    pin_memory=False,
    patch_size=16,
) -> tuple[MMDCDataModule, MMDCDataLoaderConfig]:
    dlc = MMDCDataLoaderConfig(
        max_open_files=max_open_files,
        batch_size_train=batch_size_train,
        batch_size_val=batch_size_val,
        num_workers=num_workers,
        pin_memory=pin_memory,
        patch_size=patch_size,
    )
    dpth = MMDCDataPaths(
        tensors_dir=Path(MMDC_DATASET_DIR),
        train_rois=Path(f"{MMDC_TILES_CONFIG_DIR}/train_tiny.txt"),
        val_rois=Path(f"{MMDC_TILES_CONFIG_DIR}/val_tiny.txt"),
        test_rois=Path(f"{MMDC_TILES_CONFIG_DIR}/test_tiny.txt"),
    )
    print(dpth)
    dm = MMDCDataModule(
        data_paths=dpth,
        dl_config=dlc,
    )
    return dm, dlc


def build_data_loader(
    max_open_files=1,
    batch_size_train=1,
    batch_size_val=1,
    num_workers=1,
    pin_memory=False,
    patch_size=16,
) -> tuple[DataLoader, MMDCDataLoaderConfig, MMDCDataModule]:
    dm, dlc = build_datamodule(
        max_open_files,
        batch_size_train,
        batch_size_val,
        num_workers,
        pin_memory,
        patch_size,
    )
    dm.setup("fit")
    dl = dm.train_dataloader()
    print(f"{dm.data.train.max_open_files=}")
    return dl, dlc, dm


def test_iterablemmdcdataset():
    dl, dlc, _ = build_data_loader()
    print(f"{dlc=}")
    print(f"{dl=}")
    batch = next(iter(dl))
    assert len(batch) == dlc.batch_size_train
    assert len(batch[0]) == MMDC_DATA_COMPONENTS


@pytest.mark.parametrize("batch_size, patch_size", [(1, 32), (2, 16), (4, 16)])
def test_mmdc2pvae_batch(batch_size, patch_size):
    dl, dlc, _ = build_data_loader(batch_size_train=batch_size, patch_size=patch_size)
    print(f"{dlc=}")
    print(f"{dl=}")
    batch = next(iter(dl))
    pbatch = mmdc2pvae_batch(batch)
    print(f"{len(pbatch)}")
    print(f"{pbatch[0].shape=}")
    print(f"{pbatch[1].shape=}")
    assert len(pbatch[0].shape) == 4
    assert pbatch[0].shape[-1] == patch_size
    assert pbatch[0].shape[0] == batch_size
    assert len(pbatch[1].shape) == 4
    assert pbatch[1].shape[1] == 3
