import logging
from typing import Any

import torch
from pytorch_lightning import LightningModule

from ..prosail_vae import ProsailVAEConfig, get_prosail_vae

# Configure logging
NUMERIC_LEVEL = getattr(logging, "INFO", None)
logging.basicConfig(
    level=NUMERIC_LEVEL, format="%(asctime)-15s %(levelname)s: %(message)s"
)

logger = logging.getLogger(__name__)


class ProsailVAELightningModule(LightningModule):  # pylint: disable=too-many-ancestors
    def __init__(self, config: ProsailVAEConfig, lr: float = 1e-3):
        super().__init__()
        self.model = get_prosail_vae(config, device=self.device)
        self.n_samples = 1  # how many samples drawn for reconstruction
        self.learning_rate = lr

    def step(self, batch):
        train_loss_dict = {}
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

        return {"loss": loss}

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
