import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torch import Tensor
from typing import Tuple


class DecoderOutput(pl.LightningModule):
    """
    Decoder Object

    Note that the forward method only steps forward in time ONCE. 
    It expects an input of [batch]
    """

    def __init__(self, decoder_trans_output_dim, output_dim, mode3_encoder, callsign_encoder):
        """
        Initialises the decoder object.
        :param output_dim: number of classes to predict
        :param hid_dim: hidden dimensions in each layer
        :param n_layers: number of layers (same for encoder and decoder)
        :param dropout: dropout ratio for decoder
        :param attention: attention object to used (initialized in seq2seq)
        """
        super().__init__()
        # self.device = torch.device(
        #     "cuda" if torch.cuda.is_available() else "cpu")
        self.output_dim = output_dim

        self.mode3_encoder = mode3_encoder
        self.callsign_encoder = callsign_encoder

        self.fc_out1 = nn.Linear(
            decoder_trans_output_dim + self.mode3_encoder.id_embed_dim + self.callsign_encoder.id_embed_dim, 4*output_dim).double()
        self.fc_out2 = nn.Linear(4*output_dim, 2*output_dim).double()
        self.fc_out3 = nn.Linear(2*output_dim, output_dim).double()

        self.activation = nn.LeakyReLU()

    def forward(self, decoder_trans_output, mode3_input, callsign_input):
        """
        Forward propagation.
        :param input: label of dataset at each timestamp (tensor) [batch_size]

        :param encoder_output: used to measure similiarty in states in attention [batch_size,sequence_len,hid_dim]
        :param hidden_cell: hidden state from previous timestamp (tensor) ([batch_size,n_layer,hid_dim],[batch_size,n_layer,hid_dim]) 
        :param mask: mask to filter out the paddings in attention object [batch_size,sequence_len]
        :return: normalized output probabilities for each timestamp - softmax (tensor) [batch_size,sequence_len,num_outputs]
        """

        # we can also add information on the callsign here
        mode3_embedding = self.mode3_encoder(mode3_input)
        callsign_embedding = self.callsign_encoder(callsign_input)

        if len(decoder_trans_output.shape) == 2:
            cat = torch.cat((decoder_trans_output, mode3_embedding,
                            callsign_embedding), dim=1)
        else:
            seq_len = decoder_trans_output.shape[1]

            # print(decoder_trans_output.shape)
            # print(mode3_embedding.shape)
            # print(callsign_embedding.shape)
            cat = torch.cat((decoder_trans_output, mode3_embedding.unsqueeze(1).repeat(1, seq_len, 1),
                            callsign_embedding.unsqueeze(1).repeat(1, seq_len, 1)), dim=2)

        prediction = self.activation(self.fc_out1(
            cat))
        prediction = self.activation(self.fc_out2(prediction))
        prediction = self.fc_out3(prediction)
        return F.softmax(prediction, dim=-1)
