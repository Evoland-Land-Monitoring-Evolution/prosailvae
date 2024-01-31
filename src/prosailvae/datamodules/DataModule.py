"""
ProsailVAE Lightning Data Module
"""

from pytorch_lightning import LightningDataModule
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from ..dataset.loaders import get_loader_from_patches


class ProsailVAEDataModule(LightningDataModule):
    def __init__(
        self,
        patches_dir: Path,
        bands: list[int],
        batch_size: int = 1,
        num_workers: int = 0,
        max_valid_samples: int = 50,
        concat: bool = False,
    ):
        super().__init__()
        self.patches_dir = patches_dir
        self.bands = torch.as_tensor(bands)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_valid_samples = max_valid_samples
        self.concat = concat

    def train_dataloader(self) -> DataLoader:
        return self.instanciate_dataloader("train_patches.pth")

    def validation_dataloader(self) -> DataLoader:
        return self.instanciate_dataloader("valid_patches.pth")

    def test_dataloader(self) -> DataLoader:
        return self.instanciate_dataloader("test_patches.pth")

    def instanciate_dataloader(self, fname: str) -> DataLoader:
        return get_loader_from_patches(
            path_to_patches=self.patches_dir / fname,
            # bands=self.bands,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            concat=self.concat,
        )
