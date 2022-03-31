# -*- coding: utf-8 -*-
import logging
import os
from typing import Callable, List, Optional, Union

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST, CelebA

from src import _PATH_DATA


class DebuggedCelebA(CelebA):
    def __init__(
        self,
        root: str = _PATH_DATA,
        split: str = "train",
        target_type: Union[List[str], str] = "attr",
        transform: Optional[Callable] = None,
        download: bool = False,
    ):
        def target_transform(target):
            col_idx_1 = self.attr_names.index("Male")
            col_idx_2 = self.attr_names.index('Blond_Hair')
            col_idx_3 = self.attr_names.index('Brown_Hair')
            col_idx_4 = self.attr_names.index('Black_Hair')
            return target[[col_idx_1,col_idx_2,col_idx_3,col_idx_4]]

            # if target[col_idx_1] == 0:
            #     if target[col_idx_2] == 0:
            #         return torch.tensor([1, 0, 0, 0])
            #     else:
            #         return torch.tensor([0, 1, 0, 0])
            # else:
            #     if target[col_idx_2] == 0:
            #         return torch.tensor([0, 0, 1, 0])
            #     else:
            #         return torch.tensor([0, 0, 0, 1])

        super().__init__(
            root, split, target_type, transform, target_transform, download
        )

    def _check_integrity(self):
        return True


class CelebADataModule(pl.LightningDataModule):
    def __init__(
        self, data_dir: str = _PATH_DATA, batch_size: int = 64, num_workers: int = 0
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        # download
        DebuggedCelebA(self.data_dir, split="train", download=False)
        DebuggedCelebA(self.data_dir, split="test", download=False)

    def setup(self):
        transforms_seq = transforms.Compose(
            [transforms.Resize((128, 128)), transforms.ToTensor()]
        )
        self.celeb_test = DebuggedCelebA(
            self.data_dir, split="test", transform=transforms_seq
        )

        self.celeb_train = DebuggedCelebA(
            self.data_dir, split="train", transform=transforms_seq
        )

        self.celeb_val = DebuggedCelebA(
            self.data_dir, split="valid", transform=transforms_seq
        )

    def train_dataloader(self):
        return DataLoader(
            self.celeb_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.celeb_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.celeb_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    # def predict_dataloader(self):
    #     return DataLoader(
    #         self.celeb_predict,
    #         batch_size=self.batch_size,
    #         num_workers=self.num_workers,
    #         pin_memory=True,
    #     )


class MNISTDataModule(pl.LightningDataModule):
    def __init__(
        self, data_dir: str = _PATH_DATA, batch_size: int = 64, num_workers: int = 0
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self):
        mnist_test = MNIST(self.data_dir, train=False)
        mnist_test.data = mnist_test.data.to(float) / 255
        self.mnist_test = dataset(mnist_test.data, mnist_test.targets)
        self.mnist_predict = dataset(mnist_test.data, mnist_test.targets)

        mnist_full = MNIST(self.data_dir, train=True)
        mnist_full.data = mnist_full.data.to(float) / 255
        mnist_train, mnist_val = random_split(
            mnist_full, [55000, 5000], generator=torch.Generator().manual_seed(42)
        )

        self.mnist_train = dataset(
            mnist_train.dataset.data[mnist_train.indices],
            mnist_train.dataset.targets[mnist_train.indices],
        )
        self.mnist_val = dataset(
            mnist_val.dataset.data[mnist_val.indices],
            mnist_val.dataset.targets[mnist_val.indices],
        )

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.mnist_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.mnist_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.mnist_predict,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class CIFARDataModule(pl.LightningDataModule):
    def __init__(
        self, data_dir: str = _PATH_DATA, batch_size: int = 64, num_workers: int = 0
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        # download
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self):
        cifar_test = CIFAR10(
            self.data_dir, train=False, transform=transforms.ToTensor()
        )
        self.cifar_test = cifar_test

        cifar_full = CIFAR10(self.data_dir, train=True, transform=transforms.ToTensor())
        self.cifar_train, self.cifar_val = random_split(
            cifar_full, [45000, 5000], generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(
            self.cifar_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.cifar_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.cifar_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.cifar_predict,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )


def main():
    """Runs data processing scripts to turn raw data into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("Downloading and loading raw data from torchvision")
    mnist = MNISTDataModule()
    mnist.prepare_data()
    cifar = CIFARDataModule()
    cifar.prepare_data()


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
