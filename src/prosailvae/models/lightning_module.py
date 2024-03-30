""" Pytorch Lightning Module for training a ProsailVAE model """

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pytorch_lightning import LightningModule

from ..metrics.results import save_results_on_s2_data, save_validation_results
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
        latent_samples: int = 10,
        val_config: ValidationConfig | None = None,
        resume_from_checkpoint: str | None = None,
    ):
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.model = model
        self.latent_samples = latent_samples
        self.learning_rate = lr
        self.val_config = val_config
        self.resume_from_checkpoint = resume_from_checkpoint

    def step(self, batch: Any) -> Any:
        """Generic step of the model. Delegates to the pytorch model"""
        train_loss_dict: dict = {}
        loss_sum, loss_dict = self.model.unsupervised_batch_loss(
            batch,
            train_loss_dict,
            n_samples=self.latent_samples,
        )
        return loss_sum, loss_dict

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
        opt_loss, loss_dict = self.step(batch)

        # log training metrics
        for loss_type, loss in loss_dict.items():
            loss_name = f"train/{loss_type}"
            self.log(
                loss_name,
                loss,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
            )
        return opt_loss

    def validation_step(  # pylint: disable=arguments-differ
        self,
        batch: Any,
        batch_idx: int,  # pylint: disable=unused-argument
        prefix: str = "val",
    ) -> dict[str, Any]:
        """Validation step. Step and return loss and metrics."""
        opt_loss, loss_dict = self.step(batch)

        # log validation metrics
        for loss_type, loss in loss_dict.items():
            loss_name = f"{prefix}/{loss_type}"
            self.log(
                loss_name,
                loss,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
            )

        if self.val_config is not None and prefix == "val" and batch_idx == 0:
            logger.info(f"Validation config {self.val_config}")
            save_results_on_s2_data(
                self.model,
                self.trainer.val_dataloaders,
                self.val_config.res_dir / f"ep_{self.current_epoch}_{self.global_step}",
                logger.name,
                self.val_config.plot_results,
                info_test_data=np.load(
                    Path(self.val_config.belsar_data_dir).parent.parent
                    / "s2_patch_dataset/test_info.npy"
                ),
            )
            save_validation_results(
                self.model,
                self.val_config.res_dir / f"ep_{self.current_epoch}_{self.global_step}",
                self.val_config.frm4veg_data_dir,
                self.val_config.frm4veg_2021_data_dir,
                self.val_config.belsar_data_dir,
                model_name=f"pvae_{self.current_epoch}",
                method="simple_interpolate",
                mode="sim_tg_mean",
                remove_files=self.val_config.remove_files,
                plot_results=self.val_config.plot_results,
            )

        return {"loss": opt_loss}

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
            "interval": "step",
            "monitor": "val/loss_sum",
            "frequency": 100,
        }
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }
