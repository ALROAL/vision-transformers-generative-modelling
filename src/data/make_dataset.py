# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import torchvision.datasets as datasets
from torch.utils.data import Dataset
from torch import save

from src import _PATH_DATA

class dataset:
    def __init__(self,data,target):
        self.data = data
        self.target = target
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        X = torch.bernoulli(self.data[idx])   
        y = self.target[idx]
        
        return X,y


def main():
    """ Runs data processing scripts to turn raw data into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Downloading and loading raw data from torchvision')
    train_dataset = datasets.MNIST(root=_PATH_DATA,download=True,train=True)
    test_dataset = datasets.MNIST(root=_PATH_DATA,download=True,train=False)

    logger.info('making final data set from raw data')

    train_data = train_dataset.data.to(float)/255
    train_target = train_dataset.targets
    test_data = test_dataset.data.to(float)/255
    test_target = test_dataset.targets

    train_dataset = dataset(train_data, train_target)
    test_dataset = dataset(test_data, test_target)

    save(train_dataset,"{}/train_dataset.pt".format(_PATH_DATA))
    save(test_dataset,"{}/test_dataset.pt".format(_PATH_DATA))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
