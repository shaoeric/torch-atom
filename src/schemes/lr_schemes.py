from torch.optim import lr_scheduler


__all__ = [
    'constant_lr', 
    'multi_step_lr',
    'cosine_annealing_lr',
    'warmup_cosine_annealing_lr'
]


def constant_lr(optimizer, epochs, *args, **kwargs):
    gamma = 1.0
    last_epoch = kwargs["last_epoch"] if "last_epoch" in kwargs else -1
    scheduler = lr_scheduler.StepLR(optimizer, step_size=epochs, gamma=gamma, last_epoch=last_epoch)
    return scheduler


def multi_step_lr(optimizer, epochs, *args, **kwargs):
    milestones = kwargs["milestones"] if "milestones" in kwargs else [60, 120, 160]
    gamma = kwargs["gamma"] if "gamma" in kwargs else 0.1
    last_epoch = kwargs["last_epoch"] if "last_epoch" in kwargs else -1
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones, gamma=gamma, last_epoch=last_epoch)
    return scheduler


def cosine_annealing_lr(optimizer, epochs, *args, **kwargs):
    T_max = kwargs["T_max"] if "T_max" in kwargs else epochs // 5
    eta_min = kwargs["eta_min"] if "eta_min" in kwargs else 1e-6
    last_epoch = kwargs["last_epoch"] if "last_epoch" in kwargs else -1
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min, last_epoch=last_epoch)
    return scheduler


def warmup_cosine_annealing_lr(optimizer, epochs, *args, **kwargs):
    T_0 = kwargs["T_0"] if "T_0" in kwargs else 0
    T_mult = kwargs["T_mult"] if "T_mult" in kwargs else 1
    eta_min = kwargs["eta_min"] if "eta_min" in kwargs else 1e-6
    last_epoch = kwargs["last_epoch"] if "last_epoch" in kwargs else -1
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min, last_epoch=last_epoch)
    return scheduler