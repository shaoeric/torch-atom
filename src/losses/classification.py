import torch
import torch.nn as nn


__all__ = [
    "CrossEntropyLoss"
]

class CrossEntropyLoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(CrossEntropyLoss, self).__init__()
        self.loss_func = nn.CrossEntropyLoss(*args, **kwargs)

    def forward(self, input, label):
        loss = self.loss_func(input, label)
        return loss