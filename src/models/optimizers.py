from typing import Iterable

from torch import optim
from torch.nn import Parameter


class InitOptimizer:
    def __init__(self, optimizer: str, lr,):
        self.optimizer = optimizer
        self.lr = lr


    def __call__(self, parameters: Iterable[Parameter]):
        if self.optimizer == 'adam':
            return optim.Adam(parameters, lr=self.lr,)
        elif self.optimizer == 'sgd':
            return optim.SGD(parameters, lr=self.lr,)


class InitLearningRateScheduler:

    def __init__(self, scheduler: str, init_optimizer: InitOptimizer, lr_scheduler_config: dict):
        self.scheduler = scheduler
        self.init_optimizer = init_optimizer
        self.lr_scheduler_config = lr_scheduler_config

    def __call__(self, parameters: Iterable[Parameter]):
        optimizer = self.init_optimizer(parameters)
        lr_scheduler = self.init_lr_scheduler(optimizer)
        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler,
        }

    def init_lr_scheduler(self, optimizer):
        if self.scheduler == 'step':
            return optim.lr_scheduler.StepLR(optimizer, **self.lr_scheduler_config)
        elif self.scheduler == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(optimizer, **self.lr_scheduler_config)
        else:
            raise ValueError(f'Unknown scheduler: {self.scheduler}')