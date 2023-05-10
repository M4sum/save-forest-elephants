from typing import Dict, Any

import numpy as np
import pytorch_lightning as pl
import torch
from transformers import ASTConfig, ASTModel, ASTFeatureExtractor
from torch import nn

from src.data.dataset import MelSpectogramPipeline
from src.models.AbstractModel import AbstractModel

'''
    NOTE Does not work at the moment!
'''

class ASTPL(AbstractModel):
    '''
        Pytorch lightning wrapper for the audio spectogram transformer.

        Warning: This model is only working in a very hacky way.

    '''

    def __init__(self, train_metrics: Dict[str, Any] = None, val_metrics: Dict[str, Any] = None):
        super(ASTPL, self).__init__(train_metrics, val_metrics)

        self.max_length = 79

        self.feature_extractor = MelSpectogramPipeline()
        configuration = ASTConfig(max_length=self.max_length, num_mel_bins=128, time_stride=7)

        model = ASTModel(configuration)
        self.model = model

        # Final predictions by using 2d convolutions with kernel size 1
        self.final_conv = torch.nn.Conv1d(768, 2, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()

        self.loss = nn.BCELoss()

    def forward(self, audio: torch.IntTensor):
        features = self.feature_extractor(audio)
        x_hat = self.model(features.permute(0,2,1))


        # We need to aggregate the hidden states.

        probs = self.sigmoid(self.final_conv(x_hat['last_hidden_state'].permute(0, 2, 1)))

        # Final predictions need to come from hidden state, but each second has more than one prediction.
        # For now we randomly take 10. Next up we need to average them somehow.
        return probs[:, :, 10:20]


    def pool(self, last_hidden_state: torch.Tensor) -> torch.Tensor:
        '''
            Pooling function for the hidden states.

            The logic is a bit complicated. But the idea is that we take the hidden states corresponding to a certain second and pool that one
            into a single hidden state.

            See https://huggingface.co/docs/transformers/model_doc/audio-spectrogram-transformer.
            We map the hidden states to their temporal place and pool those.
        '''

        # Remove the cls token and distilation token
        true_hidden_states = last_hidden_state[:, 2:, :]
        # Every second has 12 hidden states, For example the first


    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-5)
        return optimizer
