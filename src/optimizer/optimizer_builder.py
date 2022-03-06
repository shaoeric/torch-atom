try:
    from .optimizers import *
except:
    from sys import path
    path.append("../")
    from optimizers import *
import os
import yaml
from easydict import EasyDict
from typing import Tuple, Iterator
from torch import optim
import torch.nn as nn
import re

__all__ = ["OptimizerBuilder"]


OPTIMIZER_CONFIG = os.path.join(os.path.dirname(__file__), "optimizer_config.yml")


def parse_optimizer_config():
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
    with open(OPTIMIZER_CONFIG, 'r') as f:
        data = yaml.load(f, Loader=loader)
    return EasyDict(data)


class OptimizerBuilder:
    def __init__(self) -> None:
        pass

    @staticmethod
    def load(optimizer_name: str, parameters: Iterator[nn.Parameter], lr: float) -> Tuple[optim.Optimizer, dict]:
        config = parse_optimizer_config()
        optimizer_param = config[optimizer_name]
        optimizer = globals()[optimizer_name](parameters=parameters, lr=lr, **optimizer_param)
        return optimizer, {optimizer_name: optimizer_param}


if __name__ == '__main__':
    import torch.nn as nn
    model = nn.Linear(3, 2)
    optimizer, param = OptimizerBuilder.load('Adam', model.parameters(), 0.1)
    print(param)