import torch


__all__ = [
    "Accuracy"
]

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


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
            res = accuracy(pred, label)[0]
            return res.item()


if __name__ == '__main__':
    pred = torch.randn(size=(3, 5))
    label = torch.randint(low=0, high=5, size=(3, ))
    print(pred.argmax(1))
    print(label)
    metric = Accuracy()
    acc = metric(pred, label)
    print(acc)