"""
ProsailVAE Lightning Data Module
"""
from dataclasses import dataclass
from pathlib import Path

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from ..dataset.loaders import get_loader_from_patches


@dataclass
class DataModuleConfig:
    """Configuration for the ProsailVAEDataModule"""

    patches_dir: Path
    bands: list[int]
    batch_size: int = 1
    num_workers: int = 0
    max_valid_samples: int = 50
    concat: bool = False


class ProsailVAEDataModule(LightningDataModule):
    """ProsailVAE Lightning Data Module"""

    def __init__(self, config: DataModuleConfig):
        super().__init__()
        self.patches_dir = config.patches_dir
        self.bands = torch.as_tensor(config.bands)
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.max_valid_samples = config.max_valid_samples
        self.concat = config.concat

    def train_dataloader(self) -> DataLoader:
        return self.instanciate_dataloader("train_patches.pth")

    def validation_dataloader(self) -> DataLoader:
        """Instanciate and return a validation data loader"""
        return self.instanciate_dataloader("valid_patches.pth")

    def test_dataloader(self) -> DataLoader:
        """Instanciate and return a testing data loader"""
        return self.instanciate_dataloader("test_patches.pth")

    def instanciate_dataloader(self, fname: str) -> DataLoader:
        """Instanciate and return a data loader using patches from the given file"""
        return get_loader_from_patches(
            path_to_patches=self.patches_dir / fname,
            # bands=self.bands,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            concat=self.concat,
        )
