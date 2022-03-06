import torch
import torch.nn as nn
from typing import List


class LossWrapper(nn.Module):
    def __init__(self, loss_func_list: List[nn.Module], loss_weight_list: List[float], *args, **kwargs) -> None:
        super(LossWrapper, self).__init__()
        self.loss_func_list = loss_func_list
        self.loss_weight_list = loss_weight_list
        assert len(self.loss_func_list) == len(self.loss_weight_list), "length of loss function list should match the length of loss weight list"
        self.num_meter = len(self.loss_func_list)
        if len(self.loss_func_list) == 1:
            self.loss_weight_list = [1.0]

    def forward(self, pred: torch.Tensor, targets: List[torch.Tensor], *args, **kwargs):
        """
        Calculate the total loss between model prediction and target list
        
        Args:
            pred (torch.Tensor): model prediction
            targets (List[torch.Tensor]): a list of targets for multi-task / multi loss training

        Returns:
            loss (torch.FloatTensor): a weighted loss tensor
            loss_list (tuple[torch.FloatTensor]): a tuple of loss item
            pred (torch.FloatTensor): model output without grad
        """
        assert len(self.loss_func_list) == len(targets), "length of loss function list should match the length of targets"

        loss = 0.0
        loss_list = []
        for loss_func, loss_weight, target in zip(self.loss_func_list, self.loss_weight_list, targets):
            loss_item = loss_func(pred, target) * loss_weight
            loss += loss_item
            loss_list.append(loss_item.detach().item())

        return loss, tuple(loss_list), pred.detach()


if __name__ == '__main__':
    from loss_builder import LossBuilder
    model = nn.Linear(3, 5)
    x = torch.randn(2, 3)
    y = torch.randint(0, 5, size=(2, ))

    # loss = CrossEntropyLoss * 1.0
    ce_loss = LossBuilder.load("CrossEntropyLoss")
    wrapper = LossWrapper([ce_loss], [1.0])
    out = model(x)
    loss, loss_list, output = wrapper.forward(out, [y])
    print("loss: {} loss_list: {}, pred: {}".format(loss, loss_list, output.max(dim=1)[1]))