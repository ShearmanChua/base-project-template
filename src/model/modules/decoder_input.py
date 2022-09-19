import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torch import Tensor
from typing import Tuple


class DecoderInput(pl.LightningModule):
    """
    Decoder Object

    Note that the forward method only steps forward in time ONCE. 
    It expects an input of [batch]
    """

    def __init__(self, output_dim, embedding_dim, labels_map):
        """
        Initialises the decoder object.
        :param output_dim: number of classes to predict
        :param hid_dim: hidden dimensions in each layer
        """
        super().__init__()
        self.output_dim = output_dim
        self.embedding_dim = embedding_dim
        self.labels_map = labels_map
        # we have two special tokens
        self.embedding = nn.Embedding(output_dim, embedding_dim)

    def forward(self, decoder_input):
        """
        Forward propagation.
        :param input: label of dataset at each timestamp (tensor) [batch_size]

        :param encoder_output: used to measure similiarty in states in attention [batch_size,sequence_len,hid_dim]
        :param hidden_cell: hidden state from previous timestamp (tensor) ([batch_size,n_layer,hid_dim],[batch_size,n_layer,hid_dim]) 
        :param mask: mask to filter out the paddings in attention object [batch_size,sequence_len]
        :return: normalized output probabilities for each timestamp - softmax (tensor) [batch_size,sequence_len,num_outputs]
        """
        # hidden = hidden_cell[0]
        # last_hidden = hidden[:,-1,:] #batch_size,hid_dim

        # decoder input is [batch]
        # decoder_input = decoder_input.unsqueeze(1) #[batch,1]
        embedded = self.embedding(decoder_input)  # [batch,hid_dim]
        # embedded = self.dropout(embedded)

        return embedded
