try:
    from .accuracy import Accuracy
except:
    from sys import path
    path.append("../metric")
    from accuracy import Accuracy


__all__ = [
    "MetricBuilder"
]


class MetricBuilder:
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def load(metric_name: str, *args, **kwargs) -> object:
        metric_func = globals()[metric_name]
        return metric_func(*args, **kwargs)



if __name__ == '__main__':
    import torch
    pred = torch.randn(size=(30, 2))
    label = torch.randint(low=0, high=2, size=(30, ))
    metric = MetricBuilder.load("Accuracy")
    acc = metric(pred, label)
    print(acc)