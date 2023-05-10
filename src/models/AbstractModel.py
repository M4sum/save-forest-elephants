from typing import Dict, Any

import pytorch_lightning as pl
import torch
from torch import nn


class AbstractModel(pl.LightningModule):
    '''
        Abstract class for all models
        Takes care of:
            - Basic steps
            - Logging
    '''

    def __init__(self, train_metrics: Dict[str, Any] = None, val_metrics: Dict[str, Any] = None):
        super().__init__()
        self.train_metrics = train_metrics
        self.val_metrics = val_metrics

        # For each metric we add it as an attribute to this module
        # Prevents a weird bug where the metrics are not logged
        self.register_metrics(self.train_metrics)
        self.register_metrics(self.val_metrics)

        self.loss = nn.BCELoss()

    def register_metrics(self, metrics: Dict[str, Any]):
        if metrics:
            for name, metric in metrics.items():
                self.add_module(name, metric)

    def process_batch(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)

        loss = self.loss(y_hat, y)

        return loss, y, y_hat

    def training_step(self, batch, batch_idx):
        loss, y, y_hat = self.process_batch(batch, batch_idx)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        if self.train_metrics:
            self.log_metrics(y, y_hat, self.train_metrics)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        loss, y, y_hat = self.process_batch(batch, batch_idx)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        if self.val_metrics:
            self.log_metrics(y, y_hat, self.val_metrics)

        return loss

    def log_metrics(self, y, y_hat, metrics: Dict[str, Any], ):

        for name, metric in metrics.items():
            metric(y_hat, y)
            self.log(name, metric, on_step=False, on_epoch=True, prog_bar=True, logger=True)
