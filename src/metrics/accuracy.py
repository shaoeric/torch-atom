import torch


__all__ = [
    "Accuracy"
]


class Accuracy:
    def __init__(self) -> None:
        pass

    def __call__(self, pred: torch.Tensor, label: torch.Tensor) -> float:
        """
        Calculate the accuracy of predictions

        Args:
            pred (torch.Tensor): shape: [N, C]
            label (torch.Tensor): shape: [N, ]
        Return:
            accuracy (float): range from 0 to 100
        """
        with torch.no_grad():
            assert len(pred.shape) == 2
            assert pred.size(0) == label.size(0)
            total = pred.size(0)
            correct = (pred.argmax(dim=1) == label).sum().float().item()
            accuracy = correct / total
            return accuracy * 100.0


if __name__ == '__main__':
    pred = torch.randn(size=(3, 5))
    label = torch.randint(low=0, high=5, size=(3, ))
    metric = Accuracy()
    acc = metric(pred, label)
    print(acc)