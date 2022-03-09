try:
    from .transforms import *
except:
    from transforms import *


class TransformBuilder:
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def load(name: str = 'cifar100_transform'):
        transform_func_name = globals()[name]
        train_transform, val_transform = transform_func_name()
        return train_transform, val_transform


if __name__ == '__main__':
    train_transform, val_transform = TransformBuilder.load('cifar100_transform')
    print(train_transform)
    print(val_transform)