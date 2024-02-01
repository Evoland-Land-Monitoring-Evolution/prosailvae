from pathlib import Path

import pytest
from pytorch_lightning import Trainer

from prosailvae.datamodules.data_module import DataModuleConfig, ProsailVAEDataModule

from .test_lightning_module import test_lightning_instanciate


@pytest.mark.slow
def test_train_pipeline():
    bands = 10
    lit_mod = test_lightning_instanciate(bands=bands)

    cfg = DataModuleConfig(Path("/tmp"), list(range(bands)))
    data_mod = ProsailVAEDataModule(cfg)
    trainer = Trainer(num_sanity_val_steps=1, max_epochs=1)
    trainer.fit(model=lit_mod, datamodule=data_mod)
    optimized_metric = "train/loss"
    score = trainer.callback_metrics.get(optimized_metric)
    return score
