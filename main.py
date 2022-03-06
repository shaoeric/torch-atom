import logging
from turtle import width
import torch
from src.models import ModelBuilder
from src.losses import LossBuilder, LossWrapper
from src.optimizer import OptimizerBuilder
from src.schemes import SchedulerBuilder
from configs import ConfigLoader
from datetime import datetime
from src.utils import NetIO, LoggerBuilder
from src.metrics import MetricBuilder


date = datetime.now().strftime("%Y%m%d")
config  = ConfigLoader.load("./configs/20220223_cifar100.yml")

metric_func = MetricBuilder.load(config.train.metric["names"][0])

config.output["save_dir"] = "{}_{}".format(date, config.output["save_dir"])
netio = NetIO(config)


# logger, summary = LoggerLoader(config).load()


epochs = 10
x = torch.randn(size=(2, 3, 224, 224))
y = torch.randint(0, 1000, size=(2,))

loss_func = LossBuilder.load("CrossEntropyLoss", weight=torch.tensor([1] * 1000).float())

model = ModelBuilder.load("resnet18")
optimizer, optimizer_config = OptimizerBuilder.load("SGD", parameters=model.parameters(), lr=0.01)

scheduler, scheduler_config = SchedulerBuilder.load("multi_step_lr", optimizer, epochs)

import matplotlib.pyplot as plt
trains = []
vals = []
trains_acc = []
vals_acc = []
for epoch in range(epochs):
    # scheduler.step()
    model.train()
    out = model(x)
    loss = loss_func(out, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    trains.append(loss.item())
    train_acc = metric_func(out, y)
    trains_acc.append(train_acc)

    model.eval()
    with torch.no_grad():
        out = model(x)
        val_loss = loss_func(out, y)
        vals.append(val_loss)
        val_acc = metric_func(out, y)
        vals_acc.append(val_acc)

plt.plot(list(range(epochs)), trains)
plt.show()
plt.plot(list(range(epochs)), vals)
plt.show()
plt.plot(list(range(epochs)), trains_acc)
plt.show()
plt.plot(list(range(epochs)), vals_acc)
plt.show()
