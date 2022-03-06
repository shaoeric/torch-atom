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


__all__ = ["OptimizerBuilder"]


OPTIMIZER_CONFIG = os.path.join(os.path.dirname(__file__), "optimizer_config.yml")


def parse_optimizer_config():
        with open(OPTIMIZER_CONFIG, 'r') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
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


