import torch
from configs import ConfigLoader
from datetime import datetime
from src import DatasetBuilder, ModelBuilder, LossBuilder, LossWrapper, OptimizerBuilder, SchedulerBuilder, MetricBuilder, NetIO, Trainer
import argparse
import numpy as np
import os
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def prepare_environment(config_path: str):
    config  = ConfigLoader.load(config_path)
    date = datetime.now().strftime("%Y%m%d")
    config.output["save_dir"] = "{}_{}".format(date, config.output["save_dir"])
    
    seed = config.environment['seed']
    set_seed(seed)

    if config.environment.cuda.flag:
        os.environ['CUDA_VISIBLE_DEVICES'] = config.environment.cuda.devices
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
    
    return config

def build_dataloader(config):
    batch_size = config.train['batch_size']
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    trainset, trainset_config = DatasetBuilder.load(dataset_name=config.dataset['name'], transform=train_transform, train=True)
    valset, valset_config = DatasetBuilder.load(dataset_name=config.dataset['name'], transform=val_transform, train=False)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=4)
    return (train_loader, trainset_config), (val_loader, valset_config)


def build_trainer(config):
    netio = NetIO(config)

    model = ModelBuilder.load(config.model['name'], num_classes=config.model['num_classes'])
    if config.model['resume']:
        model = netio.load_file(model, config.model['ckpt'])

    loss_func1 = LossBuilder.load("CrossEntropyLoss")
    loss_wrapper = LossWrapper([loss_func1], [config.train.criterion['loss_weights']])

    if config.environment.cuda.flag:
        model = model.cuda()
        loss_wrapper = loss_wrapper.cuda()
    
    trainer = Trainer(config=config, model=model, wrapper=loss_wrapper, ioer=netio)
    return trainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default="./configs/20220223_cifar100.yml")
    args = parser.parse_args()

    config = prepare_environment(config_path=args.config_path)

    start_epoch = config.train['start_epoch']
    max_epoch = config.train['epochs'] + 1

    trainer = build_trainer(config)
    (train_loader, trainset_config), (val_loader, valset_config) = build_dataloader(config)

    if trainer.logger is not None:
        trainer.logger.info(trainset_config)
        trainer.logger.info(valset_config)
        trainer.logger.info(config.train)
        trainer.logger.info(config.output)

    for epoch in range(start_epoch, max_epoch):
        trainer.train(epoch, train_loader)
        trainer.validate(epoch, val_loader)
        break


if __name__ == '__main__':
    main()

