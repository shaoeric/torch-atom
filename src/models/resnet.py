import torch.nn as nn
from torchvision.models import resnet


__all__ = [
    'resnet18'
]


class ResNet(nn.Module):
    def __init__(self) -> None:
        super(ResNet, self).__init__()
        self.backbone = resnet.resnet18()

    def forward(self, x):
        return self.backbone(x)



def resnet18():
    return ResNet()