import torch.nn as nn

try:
    from .resnet import *
    from .vgg import *
    from .shufflenet import *
    from .shufflenetv2 import *
    from .mobilenetv2 import *
except:
    from sys import path
    path.append('../models')
    from resnet import *
    from vgg import *
    from shufflenet import *
    from shufflenetv2 import *
    from mobilenetv2 import *




class ModelBuilder:
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def load(model_name, *args, **kwargs) -> nn.Module:
        model = globals()[model_name]
        return model(*args, **kwargs)


if __name__ == '__main__':
    model = ModelBuilder.load("resnet32", num_classes=10)
    print(model)