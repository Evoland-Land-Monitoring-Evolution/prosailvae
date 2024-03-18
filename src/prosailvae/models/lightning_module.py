""" Pytorch Lightning Module for training a ProsailVAE model """

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from pytorch_lightning import LightningModule

from ..metrics.results import save_validation_results
from ..simvae import SimVAE

# Configure logging
NUMERIC_LEVEL = getattr(logging, "INFO", None)
logging.basicConfig(
    level=NUMERIC_LEVEL, format="%(asctime)-15s %(levelname)s: %(message)s"
)

logger = logging.getLogger(__name__)


@dataclass
class ValidationConfig:
    res_dir: str | Path
    frm4veg_data_dir: str | Path
    frm4veg_2021_data_dir: str | Path
    belsar_data_dir: str | Path
    model_name: str = "pvae"
    method: str = "simple_interpolate"
    mode: str = "sim_tg_mean"
    save_reconstruction: bool = True
    remove_files: bool = False
    plot_results: bool = False

    def __post_init__(self):
        self.res_dir = Path(self.res_dir)
        self.frm4veg_data_dir = Path(self.frm4veg_data_dir)
        self.frm4veg_2021_data_dir = Path(self.frm4veg_2021_data_dir)
        self.belsar_data_dir = Path(self.belsar_data_dir)


class ProsailVAELightningModule(LightningModule):  # pylint: disable=too-many-ancestors
    """Pytorch Lightning Module for training a ProsailVAE model"""

    def __init__(
        self,
        model: SimVAE,
        lr: float = 1e-3,
        val_config: ValidationConfig | None = None,
    ):
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.model = model
        self.n_samples = 1  # how many samples drawn for reconstruction
        self.learning_rate = lr
        self.val_config = val_config

    def step(self, batch: Any) -> Any:
        """Generic step of the model. Delegates to the pytorch model"""
        train_loss_dict: dict = {}
        loss_sum, _ = self.model.unsupervised_batch_loss(
            batch,
            train_loss_dict,
            n_samples=self.n_samples,
        )
        return loss_sum

    def training_step(  # pylint: disable=arguments-differ
        self,
        batch: Any,
        batch_idx: int,  # pylint: disable=unused-argument
    ) -> dict[str, Any]:
        """Training step. Step and return loss."""

        if batch_idx % 100 == 0:
            logger.info(f"Training step on batch {batch_idx}")
            logger.info(f"Device {self.model.device}")
        torch.autograd.set_detect_anomaly(True)
        loss = self.step(batch)

        # log training metrics
        self.log(
            "train/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

        return loss

    def validation_step(  # pylint: disable=arguments-differ
        self,
        batch: Any,
        batch_idx: int,  # pylint: disable=unused-argument
        prefix: str = "val",
    ) -> dict[str, Any]:
        """Validation step. Step and return loss and metrics."""
        loss = self.step(batch)

        self.log(
            f"{prefix}/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

        if self.val_config is not None:
            save_validation_results(
                self.model,
                self.val_config.res_dir / f"ep_{self.current_epoch}",
                self.frm4veg_data_dir,
                self.frm4veg_2021_data_dir,
                self.belsar_data_dir,
                model_name=f"pvae_{self.current_epoch}",
                method="simple_interpolate",
                mode="sim_tg_mean",
                remove_files=True,
            )

        return {"loss": loss}

    def test_step(  # pylint: disable=arguments-differ
        self,
        batch: Any,
        batch_idx: int,  # pylint: disable=unused-argument
    ) -> dict[str, Any]:
        """Test step. Step and return loss and metrics."""
        return self.validation_step(batch, batch_idx, prefix="test")

    def on_train_epoch_end(self) -> None:
        logger.info("Ended traning epoch %s", self.trainer.current_epoch)

    def on_validation_epoch_end(self) -> None:
        logger.info("Ended validation epoch %s", self.trainer.current_epoch)

    def configure_optimizers(self) -> dict[str, Any]:
        """A single optimizer with a LR scheduler"""
        optimizer = torch.optim.Adam(
            params=self.model.parameters(), lr=self.learning_rate
        )

        training_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=4, T_mult=2, eta_min=0, last_epoch=-1
        )
        scheduler = {
            "scheduler": training_scheduler,
            "interval": "epoch",
            "monitor": "val/loss",
            "frequency": 1,
        }
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }
