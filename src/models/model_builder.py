try:
    from .resnet import *
except:
    from sys import path
    path.append('../models')
    from resnet import *
import torch.nn as nn


class ModelBuilder:
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def load(model_name, *args, **kwargs) -> nn.Module:
        model = globals()[model_name]
        return model(*args, **kwargs)


if __name__ == '__main__':
    model = ModelBuilder.load("resnet34", num_classes=10)
    print(model)