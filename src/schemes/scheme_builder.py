try:
    from .lr_schemes import *
except:
    from sys import path
    path.append('../schemes')
    from lr_schemes import *

from torch import optim
import os
import yaml
from easydict import EasyDict
import re

__all__ = [
    "SchedulerBuilder"
]


SCHEME_CONFIG = os.path.join(os.path.dirname(__file__), "scheme_config.yml")


def parse_scheduler_config():
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
    with open(SCHEME_CONFIG, 'r') as f:
        data = yaml.load(f, Loader=loader)
    return EasyDict(data)


class SchedulerBuilder:
    def __init__(self):
        pass
        
    @staticmethod
    def load(scheduler_func_name: str, optimizer: optim, max_epoch: int):
        scheduler_func = globals()[scheduler_func_name]
        config = parse_scheduler_config()
        scheduler_params = config[scheduler_func_name]
        scheduler = scheduler_func(optimizer, max_epoch, **scheduler_params)
        return scheduler, {scheduler_func_name: scheduler_params}


if __name__ == "__main__":
    from torch import optim
    from torch import nn
    import matplotlib.pyplot as plt
    epochs = 100
    m = nn.Linear(3, 5)
    optimizer = optim.SGD(m.parameters(), lr=0.001)
    scheduler, scheduler_param = SchedulerBuilder.load("cosine_annealing_lr", optimizer, epochs)
    lrs = []
    for i in range(epochs):
        scheduler.step()
        lr = scheduler.get_lr()
        lrs.append(lr)
    plt.plot(lrs)
    plt.show()