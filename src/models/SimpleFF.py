from typing import Dict, Any

import torch
from torch import nn

from src.data.dataset import MelSpectogramPipeline
from src.models.AbstractModel import AbstractModel


class SimpleFF(AbstractModel):

    def __init__(self, train_metrics: Dict[str, Any] = None, val_metrics: Dict[str, Any] = None):
        super().__init__(train_metrics, val_metrics)

        self.feature_extractor = MelSpectogramPipeline()
        self.input_size = 256 * self.sequence_length

        self.model = torch.nn.Sequential(
            nn.Linear(self.input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 20),
            nn.Sigmoid()
        )

    def forward(self, audio: torch.IntTensor):
        features = self.feature_extractor(audio)

        probs = self.model(features.reshape(-1, self.input_size)).reshape(-1, 2, 10)

        return probs

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
