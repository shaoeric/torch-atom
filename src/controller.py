import torch
import torch.nn as nn
from torch import optim
from src.losses import LossWrapper
from typing import List


__all__ = ["Controller"]


class Controller(object):
    def __init__(self, 
        loss_wrapper: LossWrapper, 
        model: nn.Module,
        optimizer: optim.Optimizer
        ) -> None:
        self.loss_wrapper = loss_wrapper
        self.model = model
        self.optimizer = optimizer


    def train_step(self, input: torch.Tensor, label: torch.Tensor, *args, **kwargs):
        """
        Define the training process for the model, easy for extension for multiple models

        Args:
            input (torch.Tensor): input tensor of the model
            label (torch.Tensor): ground truth of the input tensor

        Returns:
            loss (torch.FloatTensor): train loss
            loss_tuple (tuple[torch.FloatTensor]): a tuple of loss item
            output_no_grad (torch.FloatTensor): model output without grad
        """
        self.optimizer.zero_grad()
        output = self.model(input)
        loss, loss_tuple, output_no_grad = self.loss_wrapper(output, [label])
        loss.backward()
        self.optimizer.step()
        return loss, loss_tuple, output_no_grad


    def validate_step(self, input: torch.Tensor, label: torch.Tensor, *args, **kwargs):
        """
        Define the validation process for the model

        Args:
            input (torch.Tensor): input tensor for the model
            label (torch.Tensor): ground truth for the input tensor

        Returns:
            loss (torch.FloatTensor): validation loss item, without grad
            loss_tuple (tuple[torch.FloatTensor]): a tuple of loss item
            output_no_grad (torch.FloatTensor): model output without grad
        """
        output = self.model(input)
        loss, loss_tuple, output_no_grad = self.loss_wrapper(output, [label])
        return loss.detach(), loss_tuple, output_no_grad