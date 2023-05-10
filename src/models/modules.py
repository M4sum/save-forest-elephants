from typing import List

from torch import nn
import torch
from torch import Tensor

# Possible activation functions
ACTIVATION_FUNCTIONS = {
    'relu': nn.ReLU(),
    'elu': nn.ELU(),
    'gelu': nn.GELU(),

}


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int,  max_len: int = 100):
        super().__init__()


        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(Tensor([10000.0])) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return x


class Conv2dEmbeddingLayer(nn.Module):

    def __init__(self, embedding_size: int, max_len: int = 80, positional_encoding: bool = True,
                 activation_function: nn.Module = nn.ReLU(), dropout: float = 0.1):
        super().__init__()

        self.conv_embedding = nn.Conv2d(1, embedding_size, kernel_size=(128, 8), stride=8, )
        self.activation_function = activation_function
        self.positional_encoding = None
        if positional_encoding:
            self.positional_encoding = PositionalEncoding(embedding_size, max_len=max_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, mel_spectogram: Tensor) -> Tensor:

        # Add a channel dimension
        embeddings = self.conv_embedding(mel_spectogram.unsqueeze(1))

        embeddings = self.activation_function(embeddings)

        # Make it a one dimensional sequence
        embeddings = embeddings.squeeze(2)

        embeddings = embeddings.permute(2, 0, 1, )

        # Add the positional encoding
        if self.positional_encoding:
            embeddings = self.positional_encoding(embeddings)

        embeddings = self.dropout(embeddings)
        return embeddings


class MultipleMultiHeadTransformer(nn.Module):

    def __init__(self, attention_layers: List[nn.MultiheadAttention], activation_function: nn.Module = nn.ReLU()):
        super().__init__()
        self.attention_layers = nn.ModuleList(attention_layers)
        self.activation_function = activation_function

    def forward(self, x: Tensor):
        for attention_layer in self.attention_layers:
            x, _ = attention_layer(x, x, x)
            x = self.activation_function(x)
        return x, None
