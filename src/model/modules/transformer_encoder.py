import torch
from torch import Tensor
import torch.nn as nn


import pytorch_lightning as pl
import math


class TransformerEncoder(pl.LightningModule):
    """
    Transformer Object

    Resolves all padding issues before using the pytorch native module


    """

    def __init__(self, d_model,
                 nhead,
                 num_encoder_layers,
                 dim_feedforward,
                 encoder_dropout,
                 input_dropout,
                 activation):
        """
        Initialises the encoder object.
        :param hid_dim: hidden dimensions in each layer
        :param n_layers: number of layers (same for encoder and decoder)
        :param n_features: number of features from the dataset
        :param dropout: dropout ratio for encoder
        """
        super().__init__()
        # self.device = torch.device(
        #     "cuda" if torch.cuda.is_available() else "cpu")

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, encoder_dropout, activation, batch_first=True).double()

        encoder_norm = nn.LayerNorm(d_model).double()

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm)

        self.positional_encoder = PositionalEncoding(
            d_model, input_dropout)

    def forward(self, x, seq_len):
        """
        Forward propagation.
        :param x: features of dataset at every timestamp [batch_size,sequence_len,feature_dim]
        :param seq_len: actual length of each data sequence [batch_size]
        :return: hidden state of the last layer in the encoder;
                 outputs the outputs of last layer at every timestamps

        hidden: [batch_size, n_layer, hid_dim]
        cell: [batch_size, n_layer, hid_dim]
        lstm_output [batch_size, sequence_len, n_directions*hid_dim] , note n_direction = 2 for bidirectional-lstm 
        output [batch_size, sequence_len, n_directions*hid_dim] 
        """

        # create source padding mask
        source_mask = self.get_source_mask(seq_len)

        # add positional encoding to source
        x = self.positional_encoder(x)
        memory = self.transformer_encoder(x, src_key_padding_mask=source_mask)

        return memory

    def get_source_mask(self, lengths):
        return self.get_pad_mask(lengths)

    def get_pad_mask(self, lengths):

        max_len = lengths.max()
        row_vector = torch.arange(0, max_len, 1)  # size (seq_len)
        # matrix = torch.unsqueeze(lengths, dim=-1)  # size = N ,1
        mask = row_vector.to('cuda') >= lengths
        # mask = row_vector.to('cpu') >= lengths
        mask.type(torch.bool)
        return mask


class PositionalEncoding(nn.Module):

    '''
    d_model : size of input embedding
    '''

    def __init__(self, d_model: int, dropout: float, max_len: int = 50000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)

        a = torch.sin(position * div_term)

        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1), ...]
        return self.dropout(x)
