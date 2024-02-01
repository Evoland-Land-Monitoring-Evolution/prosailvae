from pathlib import Path

import pytest
import torch
from pytorch_lightning import Trainer

from prosailvae.datamodules.data_module import DataModuleConfig, ProsailVAEDataModule

from .test_dataloader import PATCHES_DIR
from .test_lightning_module import instanciate

TMP_DIR = Path("/tmp")


def prepare_data():
    fname = "train_patches.pth"
    x = torch.load(PATCHES_DIR / fname)
    y = x[:10, :]
    torch.save(y, TMP_DIR / fname)
    torch.save(y, TMP_DIR / "valid_patches.pth")
    torch.save(y, TMP_DIR / "test_patches.pth")


@pytest.mark.slow
def test_train_pipeline():
    prepare_data()
    bands = 10
    lit_mod = instanciate(bands=bands)
    cfg = DataModuleConfig(TMP_DIR, list(range(bands)))
    data_mod = ProsailVAEDataModule(cfg)
    trainer = Trainer(num_sanity_val_steps=1, max_epochs=1)
    trainer.fit(model=lit_mod, datamodule=data_mod)
    optimized_metric = "train/loss"
    score = trainer.callback_metrics.get(optimized_metric)
    assert score is not None
