import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import argparse
import numpy as np
import os
from configs import ConfigLoader
from datetime import datetime
from src import DatasetBuilder, TransformBuilder, ModelBuilder, LossBuilder, LossWrapper, NetIO, Trainer


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_environment(args):
    config = ConfigLoader.load(args.config_path.replace('\n', '').replace('\r', ''))
    date = datetime.now().strftime("%Y%m%d")
    if args.save_dir is not None:
        config.output["save_dir"] = args.save_dir
    config.output["save_dir"] = "{}_{}".format(date, config.output["save_dir"])

    config.model['name'] = args.model
    if args.seed is not None:
        config.environment['seed'] = args.seed
    seed = config.environment['seed']
    set_seed(seed)

    config.environment.local_rank = args.local_rank
    config.environment.num_gpu = torch.cuda.device_count()
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    return config

def build_dataloader(config):
    batch_size = config.train['batch_size']  # 128
    num_gpu = config.environment['num_gpu']
    num_workers = config.train['workers']
    num_workers = int((num_workers + num_gpu - 1) / num_gpu)

    transform_name = config.dataset['transform_name']
    dataset_name = config.dataset['name']
    train_transform, val_transform = TransformBuilder.load(transform_name)
    
    trainset, trainset_config = DatasetBuilder.load(dataset_name=dataset_name, transform=train_transform, train=True)
    valset, valset_config = DatasetBuilder.load(dataset_name=dataset_name, transform=val_transform, train=False)

    train_sampler = DistributedSampler(trainset)
    val_sampler = DistributedSampler(valset)
    print("build dataloader", config.environment.local_rank, batch_size)
    
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, sampler=train_sampler)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, sampler=val_sampler)
    return (train_loader, trainset_config), (val_loader, valset_config)


def build_trainer(config):
    local_rank = config.environment.local_rank
    netio = NetIO(config) if local_rank == 0 else None
    model = ModelBuilder.load(config.model['name'], num_classes=config.model['num_classes'])
    if config.model['resume']:
        model = netio.load_file(model, config.model['ckpt'])

    loss_func1 = LossBuilder.load("CrossEntropyLoss")
    loss_wrapper = LossWrapper([loss_func1], [config.train.criterion['loss_weights']])

    # 
    model = model.cuda(local_rank)
    loss_wrapper = loss_wrapper.cuda(local_rank)

    trainer = Trainer(config=config, model=model, wrapper=loss_wrapper, ioer=netio, ddp=True)
    return trainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default="./configs/20220223_cifar100.yml")
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    args = parser.parse_args()

    config = prepare_environment(args)
    num_gpu = config.environment.num_gpu
    main_worker(args.local_rank, num_gpu, args, config)


def main_worker(local_rank, num_gpus, args, config):
    start_epoch = config.train['start_epoch']
    max_epoch = config.train['epochs'] + 1
    # config.train['batch_size'] = config.train['batch_size'] // num_gpus
    # config.train['lr'] = config.train['lr'] / num_gpus
    # config.train['lr'] = config.train['lr'] * num_gpus
    
    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(args.local_rank)
    
    trainer = build_trainer(config)
    (train_loader, trainset_config), (val_loader, valset_config) = build_dataloader(config)

    if trainer.logger is not None and local_rank == 0:
        trainer.logger.info(trainset_config)
        trainer.logger.info(valset_config)
        trainer.logger.info(config.model)
        trainer.logger.info(config.train)
        trainer.logger.info(config.output)
        trainer.logger.info(config.environment)

    for epoch in range(start_epoch, max_epoch):
        train_loader.sampler.set_epoch(epoch)

        trainer.train(epoch, train_loader)
        trainer.validate(epoch, val_loader)

    if trainer.logger is not None and local_rank == 0:
        trainer.logger.info("best metric: {}".format(trainer.ioer.get_best_score()))

if __name__ == '__main__':
    main()

