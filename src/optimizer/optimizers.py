from torch import optim

__all__ = [
    "SGD", "Adam"
]


def SGD(parameters, lr, *args, **kwargs) -> optim.Optimizer:
    optimizer = optim.SGD(parameters, lr=lr, *args, **kwargs)
    return optimizer


def Adam(parameters, lr, *args, **kwargs) -> optim.Optimizer:
    optimizer = optim.Adam(parameters, lr=lr, *args, **kwargs)
    return optimizer