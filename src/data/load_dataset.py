from torch.utils.data import DataLoader
from torch import load
from src import _PATH_DATA


def make_dataloader(train=True,batch_size=64):
    if train:
        train_dataset = load('{}/train_dataset.pt'.format(_PATH_DATA))
        train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=4)
        return train_loader
    else:
        test_dataset = load('{}/test_dataset.pt'.format(_PATH_DATA))
        test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False,num_workers=4)
        return test_loader
