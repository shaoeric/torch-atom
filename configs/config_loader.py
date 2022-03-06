import os
import yaml
from easydict import EasyDict


__all__ = [
    "ConfigLoader"
]


class ConfigLoader:
    def __init__(self) -> None:
        self.PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

    @staticmethod
    def load(config_filepath: str) -> dict:
        with open(config_filepath, 'r') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        return EasyDict(data)


if __name__ == '__main__':
    config = ConfigLoader.load("configs/20220223_cifar100.yml")
    print(config)
    print(config['train']['criterion']['weights'])