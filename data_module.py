import os

import pytorch_lightning as pl
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from util.crop import center_crop_arr


class ImageNetDataModule(pl.LightningDataModule):
    """ImageNet data module for JiT training.

    PL automatically wraps the DataLoader with a DistributedSampler when
    running in multi-GPU/multi-node mode, so no manual sampler setup is needed.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.train_dataset = None

    def setup(self, stage: str | None = None) -> None:
        img_size = self.cfg.model.img_size
        transform = transforms.Compose([
            transforms.Lambda(lambda img: center_crop_arr(img, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.PILToTensor(),
        ])
        train_dir = os.path.join(self.cfg.data.path, "train")
        self.train_dataset = datasets.ImageFolder(train_dir, transform=transform)
        print(f"Train dataset: {self.train_dataset}")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.training.batch_size,
            num_workers=self.cfg.hardware.num_workers,
            pin_memory=self.cfg.hardware.pin_memory,
            drop_last=True,
            shuffle=True,
        )
