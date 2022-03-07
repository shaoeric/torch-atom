try:
    from .cifar import *
except:
    from sys import path
    path.append('../datasets')
    from cifar import *

import os

import yaml
from easydict import EasyDict
import re

__all__ = [
    "DatasetBuilder"
]


DATASET_CONFIG = os.path.join(os.path.dirname(__file__), "dataset_config.yml")


def parse_dataset_config():
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
        [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))
    with open(DATASET_CONFIG, 'r') as f:
        data = yaml.load(f, Loader=loader)
    return EasyDict(data)


class DatasetBuilder:
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def load(dataset_name: str = 'CIFAR10',
             *args, **kwargs):
        config = parse_dataset_config()[dataset_name]
        config.update(kwargs)
        dataset = globals()[dataset_name](*args, **config)
        return dataset, {dataset_name: config}


if __name__ == '__main__':
    from torchvision.transforms import transforms
    from torch.utils.data import DataLoader

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    trainset, trainset_config = DatasetBuilder.load(dataset_name="CIFAR10", transform=train_transform, train=True)
    valset, valset_config = DatasetBuilder.load(dataset_name="CIFAR10", transform=val_transform, train=False)
    print(trainset_config)
    print(valset_config)
    val_loader = DataLoader(valset, batch_size=16, shuffle=False, num_workers=4)
    for img, label in val_loader:
        print(img.shape, label.shape)  # torch.Size([16, 3, 32, 32]) torch.Size([16])
        break