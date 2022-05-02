# -*- coding: utf-8 -*-
import logging
import os
from typing import Callable, List, Optional, Union

import PIL
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import  MNIST, CelebA

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

            if torch.equal(target[[col_idx_1,col_idx_2,col_idx_3,col_idx_4]],torch.tensor([1,1,0,0],dtype=torch.int64)): # Blond Male
                return torch.tensor([1,0,0,0,0,0])
            elif torch.equal(target[[col_idx_1,col_idx_2,col_idx_3,col_idx_4]],torch.tensor([1,0,1,0],dtype=torch.int64)): # Brown Male
                return torch.tensor([0,1,0,0,0,0])
            elif torch.equal(target[[col_idx_1,col_idx_2,col_idx_3,col_idx_4]],torch.tensor([1,0,0,1],dtype=torch.int64)): # Black Male
                return torch.tensor([0,0,1,0,0,0])
            elif torch.equal(target[[col_idx_1,col_idx_2,col_idx_3,col_idx_4]],torch.tensor([0,1,0,0],dtype=torch.int64)): # Blond Female
                return torch.tensor([0,0,0,1,0,0])
            elif torch.equal(target[[col_idx_1,col_idx_2,col_idx_3,col_idx_4]],torch.tensor([0,0,1,0],dtype=torch.int64)): # Brown Female
                return torch.tensor([0,0,0,0,1,0])
            elif torch.equal(target[[col_idx_1,col_idx_2,col_idx_3,col_idx_4]],torch.tensor([0,0,0,1],dtype=torch.int64)): # Black Female
                return torch.tensor([0,0,0,0,0,1])
            else:
                if target[col_idx_1] == 1:
                    if target[col_idx_3] == 1:
                        return torch.tensor([0,1,0,0,0,0]) # Brown Male
                    elif target[col_idx_4] == 1:
                        return torch.tensor([0,0,1,0,0,0]) # Black Male
                    else:
                        return torch.tensor([1,0,0,0,0,0]) # Blond Male
                if target[col_idx_1] == 0:
                    if target[col_idx_3] == 1:
                        return torch.tensor([0,0,0,0,1,0]) # Brown Female
                    elif target[col_idx_4] == 1:
                        return torch.tensor([0,0,0,0,0,1]) # Black Female
                    else:
                        return torch.tensor([0,0,0,1,0,0]) # Blond Female
                    
        
        super().__init__(
            root, split, target_type, transform, target_transform, download
        )
        self.first_getitem = True

    def _check_integrity(self):
        return True

    def __len__(self):
        if self.first_getitem == True:
            idx_blond = self.attr_names.index('Blond_Hair')
            idx_brown = self.attr_names.index('Brown_Hair')
            idx_black = self.attr_names.index('Black_Hair')
            
            idxs = torch.where(torch.sum(self.attr[:,[idx_blond,idx_brown,idx_black]],axis=1) != 0)[0]
            self.filename = list(np.array(self.filename)[idxs])
            self.attr = self.attr[idxs,:]
            self.first_getitem = False
        
        return len(self.attr)
        

class CelebADataModule(pl.LightningDataModule):
    def __init__(
        self, data_dir: str = _PATH_DATA, batch_size: int = 64, num_workers: int = 0, classify = False, transforms = None
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.classify = classify
        self.transforms = transforms

    def prepare_data(self):
        # download
        DebuggedCelebA(self.data_dir, split="train", download=False)
        DebuggedCelebA(self.data_dir, split="test", download=False)

    def setup(self, stage=None):
        if self.transforms is None:
            transforms_seq = transforms.Compose(
                [transforms.Resize((128, 128)), transforms.ToTensor()]
            )
        else:
            transforms_seq = self.transforms
        if self.classify:
            transforms_seq = transforms.Compose(
                [transforms.Resize((128, 128)), transforms.ToTensor(),
                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])]
            )
        
        if stage == "test" or stage is None:
            self.celeb_test = DebuggedCelebA(
                self.data_dir, split="test", transform=transforms_seq
            )

        if stage == "fit" or stage is None:
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


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    print("nothing")
