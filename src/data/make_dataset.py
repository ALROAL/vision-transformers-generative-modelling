# -*- coding: utf-8 -*-
import logging

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST

from src import _PATH_DATA


class dataset:
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        X = torch.bernoulli(self.data[idx])
        y = self.target[idx]

        return X, y


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = _PATH_DATA, batch_size: int = 64):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

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
        return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=4)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=self.batch_size, num_workers=4)


class CIFARDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = _PATH_DATA, batch_size: int = 64):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

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
        return DataLoader(self.cifar_train, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.cifar_val, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=self.batch_size, num_workers=4)

    def predict_dataloader(self):
        return DataLoader(self.cifar_predict, batch_size=self.batch_size, num_workers=4)


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
