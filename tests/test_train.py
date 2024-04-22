from pathlib import Path

import pytest
import torch
from pytorch_lightning import Trainer

from prosailvae.datamodules.data_module import DataModuleConfig, ProsailVAEDataModule

from .paths import PATCHES_DIR, TMP_DIR
from .test_lightning_module import instanciate

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def prepare_data() -> None:
    fname = "train_patches.pth"
    if not Path(TMP_DIR / fname).exists():
        x = torch.load(PATCHES_DIR / "test_patches.pth")
        y = x[:10, :, :2, :2]
        torch.save(y, TMP_DIR / fname)
        torch.save(y, TMP_DIR / "valid_patches.pth")
        torch.save(y, TMP_DIR / "test_patches.pth")


@pytest.mark.slow
def test_train_pipeline() -> None:
    prepare_data()
    bands = 10
    lit_mod = instanciate(bands=bands)
    cfg = DataModuleConfig(TMP_DIR, list(range(bands)))
    data_mod = ProsailVAEDataModule(cfg)
    trainer = Trainer(num_sanity_val_steps=1, max_epochs=1, accelerator=DEVICE)
    trainer.fit(model=lit_mod, datamodule=data_mod)
    optimized_metric = "val/loss_sum"
    score = trainer.callback_metrics.get(optimized_metric)
    assert score is not None
    trainer.validate(model=lit_mod, datamodule=data_mod)
    trainer.test(model=lit_mod, datamodule=data_mod)
