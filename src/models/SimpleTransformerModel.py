import math
from typing import Dict, Any, Callable, List, Iterator

import torch
from torch import nn
from torch.optim.lr_scheduler import LRScheduler

from src.data.dataset import MelSpectogramPipeline
from src.metrics import get_metrics
from src.models.AbstractModel import AbstractModel
from src.models.modules import Conv2dEmbeddingLayer, MultipleMultiHeadTransformer, \
    ACTIVATION_FUNCTIONS
from src.models.optimizers import InitOptimizer, InitLearningRateScheduler


class SimpleTransformerModel(AbstractModel):
    '''
    A simple transformer model, loosely based on the Audio Spectogram Transformer model

    First we extract the mel spectogram features from the audio samples
    Then we apply a convolutional layer to get embeddings and apply positional encoding
    Then we apply multiple self attention layers
    Finally we apply a final convolutional that maps each embedding to a prediction containing the probability of each of the sound events for each second


    '''

    def __init__(self, feature_extractor: nn.Module,
                 embedding_layer: nn.Module,
                 self_attention_layers: nn.Module,
                 final_layer: nn.Module,
                 learning_rate_scheduler: Callable[[Iterator], LRScheduler],
                 train_metrics: Dict[str, Any] = None,
                 val_metrics: Dict[str, Any] = None):
        super().__init__(train_metrics, val_metrics)

        self.feature_extractor = feature_extractor
        self.embedding = embedding_layer
        self.self_attention_layers = self_attention_layers
        self.final_conv = final_layer

        self.feature_extractor = MelSpectogramPipeline()

        self.learning_rate_scheduler = learning_rate_scheduler

    def forward(self, audio: torch.FloatTensor):
        '''

        :param audio: A batch of audio samples
        :return: A batch of predictions
        '''
        features = self.feature_extractor(audio)
        # Check if features contain nan
        assert not torch.isnan(features).any(), f"Features contain nan, {features}"

        embeddings = self.embedding(features)

        # Apply self attention
        transformer_out, att = self.self_attention_layers(embeddings, )

        transformer_out = transformer_out.permute(1, 2, 0)
        # Apply a 1by1 convolution to get the final predictions
        predictions = self.final_conv(transformer_out)

        # Create probs by applying sigmoid
        probs = torch.sigmoid(predictions)

        #check if the output is between 0 and 1
        assert torch.all(probs >= 0) and torch.all(probs <= 1), f"The output of the model is not between 0 and 1, {probs}"
        
        return probs

    def configure_optimizers(self) -> LRScheduler:
        schedular = self.learning_rate_scheduler(self.parameters())
        return schedular


def create_simple_transformer_model(config: dict) -> SimpleTransformerModel:
    '''
    Creates a simple transformer model
    :param config: The config dict
    :return: A simple transformer model
    '''

    # Read out some parameters from the config
    architecture_config = config['architecture']
    embedding_size = architecture_config['embedding_size']
    activation_function = ACTIVATION_FUNCTIONS[architecture_config['activation_function']]

    feature_extractor = MelSpectogramPipeline()

    embedding_layer = Conv2dEmbeddingLayer(embedding_size, positional_encoding=True, max_len=80,
                                           activation_function=activation_function,
                                           dropout=architecture_config["dropout"],
                                           )
    transformer_layers = []
    for layer in range(architecture_config['num_hidden_layers']):
        transformer_layer = nn.MultiheadAttention(embed_dim=embedding_size, num_heads=architecture_config["num_heads"],
                                                  dropout=architecture_config["dropout"],

                                                  )
        transformer_layers.append(transformer_layer,
                                  )

    transformer_layer = MultipleMultiHeadTransformer(transformer_layers,
                                                     activation_function=activation_function,
                                                     )

    # We finish with a 1by1 convolution to get the final predictions
    final_conv = torch.nn.Conv1d(embedding_size, 2, kernel_size=1, stride=1)

    # Get the metrics
    train_metrics, val_metrics = get_metrics()

    init_optimizer = InitOptimizer(config['optimizer']['type'], config['optimizer']['lr'])
    init_lr_scheduler = InitLearningRateScheduler(config['lr_scheduler']['type'], init_optimizer,
                                                  config['lr_scheduler']['config'])

    model = SimpleTransformerModel(feature_extractor, embedding_layer, transformer_layer, final_conv,
                                   init_lr_scheduler, train_metrics,val_metrics)

    return model

