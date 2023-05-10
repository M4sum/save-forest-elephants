import numpy as np
from pytorch_lightning import Callback

from src.models.utils import save_model


class SaveModelCheckpoint(Callback):

    def __init__(self, config, model, save_dir, monitor='val_loss', mode='min'):
        self.config = config
        self.model = model
        self.save_dir = save_dir
        self.monitor = monitor
        self.mode = mode
        self.best = None

        self.model_name = config['model_name']

        self.path = f'{save_dir}/{self.model_name}.pt'

        if self.mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif self.mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            raise ValueError(f'Unknown mode {self.mode}')

    def __call__(self, epoch, logs):
        current = logs.get(self.monitor).cpu().numpy()
        if current is None:
            raise ValueError(f'Monitor {self.monitor} is not in the logs')

        if self.monitor_op(current, self.best):
            self.best = current
            save_model(self.config, self.model, self.path)
            print(f'Epoch {epoch}: Saving model with {self.monitor} of {current}')
        else:
            print(f'Epoch {epoch}: Not saving model with {self.monitor} of {current}, best is {self.best}')

    def on_validation_epoch_end(self, trainer, pl_module):
        self(trainer.current_epoch, trainer.callback_metrics)
