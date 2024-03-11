""" Training pipeline to be called by hydra experiments"""

import os

import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers import Logger

from .hydra_utils import finish, get_logger, log_hyperparameters

log = get_logger(__name__)


def setup(
    config: DictConfig,
) -> tuple[
    LightningModule,
    LightningDataModule,
    Trainer,
    list[Callback],
    list[Logger],
]:
    """Setup the elements of the training pipeline"""

    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed"):
        seed_everything(config.seed, workers=True)

    # Handle tensor cores configuration
    if config.get("mat_mul_precision"):
        torch.set_float32_matmul_precision(config.get("mat_mul_precision"))

    # Convert relative ckpt path to absolute path if necessary
    ckpt_path = config.model.get("resume_from_checkpoint")
    if ckpt_path and not os.path.isabs(ckpt_path):
        config.model.resume_from_checkpoint = os.path.join(
            hydra.utils.get_original_cwd(), ckpt_path
        )

    # Init lightning model
    # log.info("Instantiating model <%s>", config.model._target_)
    # pylint: disable=W0212
    pl_module: LightningModule = hydra.utils.instantiate(config.model)

    # Init lightning datamodule
    log.info(
        "Instantiating datamodule <%s>",
        config.datamodule._target_,  # pylint: disable=W0212
    )
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)

    # Init lightning callbacks
    callbacks: list[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                log.info(
                    "Instantiating callback <%s>",
                    cb_conf._target_,  # pylint: disable=W0212
                )
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init lightning loggers
    logger: list[Logger] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(
                    "Instantiating logger <%s>",
                    lg_conf._target_,  # pylint: disable=W0212
                )
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init lightning trainer
    log.info(
        "Instantiating trainer <%s>",
        config.trainer._target_,  # pylint: disable=W0212
    )
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )
    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters!")
    log_hyperparameters(
        config=config,
        model=pl_module,
        trainer=trainer,
    )

    return pl_module, datamodule, trainer, callbacks, logger


def train(config: DictConfig) -> torch.Tensor | None:
    """
    Contains the training pipeline.
    Can additionally evaluate model on a testset, using best
    weights achieved during training.

    :param config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    model, datamodule, trainer, _, logger = setup(config)

    # Train the model
    if config.get("train"):
        log.info("Starting training!")
        ckpt_path = config.model.get("resume_from_checkpoint")
        if ckpt_path is not None:
            log.info("Training from checkpoint %s", ckpt_path)
        else:
            log.info("Training from scratch")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    # Get metric score for hyperparameter optimization
    optimized_metric = config.get("optimized_metric")
    if optimized_metric and optimized_metric not in trainer.callback_metrics:
        raise RuntimeError(
            "Metric for hyperparameter optimization not found! "
            "Make sure the `optimized_metric` in"
            "`hparams_search` config is correct!"
        )
    score = trainer.callback_metrics.get(optimized_metric)

    # Test the model
    if config.get("test"):
        test_ckpt_path: str | None = "best"
        if not config.get("train") or config.trainer.get("fast_dev_run"):
            test_ckpt_path = None
        log.info("Starting testing with model %s", test_ckpt_path)
        trainer.test(model=model, datamodule=datamodule, ckpt_path=test_ckpt_path)

    # Make sure everything closed properly
    log.info("Finalizing!")
    finish(logger=logger)

    # Print path to best checkpoint
    if (
        ckpt_path is not None
        and trainer.checkpoint_callback is not None
        and hasattr(trainer.checkpoint_callback, "best_model_path")
    ):
        log.info(
            "Best model ckpt at %s",
            trainer.checkpoint_callback.best_model_path,
        )

    # Return metric score for hyperparameter optimization
    return score
