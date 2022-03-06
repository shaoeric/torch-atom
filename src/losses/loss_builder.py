try:
    from .classification import *
except:
    from sys import path
    path.append("../losses")
    from classification import *

import torch.nn as nn

__all__ = [
    "LossBuilder"
]

class LossBuilder:
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def load(loss_name, *args, **kwargs) -> nn.Module:
        loss_func = globals()[loss_name]
        return loss_func(*args, **kwargs)


if __name__ == '__main__':
    import torch
    loss_func = LossBuilder.load("CrossEntropyLoss", weight=torch.tensor(list(range(10))).float())
    x = torch.randn(size=(2, 10))  # input
    y = torch.tensor([0, 1]).long()  # label
    loss = loss_func.forward(x, y)
    print(loss)