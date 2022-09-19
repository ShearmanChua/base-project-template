import torch
from torch import Tensor
import torch.nn as nn


import pytorch_lightning as pl
import math


class TransformerDecoder(pl.LightningModule):
    """
    Transformer Object

    Resolves all padding issues before using the pytorch native module


    """

    def __init__(self, decoder_input,
                 decoder_output,
                 d_model,
                 nhead,
                 num_decoder_layers,
                 dim_feedforward,
                 decoder_dropout,
                 activation):
        """
        Initialises the encoder object.
        :param hid_dim: hidden dimensions in each layer
        :param n_layers: number of layers (same for encoder and decoder)
        :param n_features: number of features from the dataset
        :param dropout: dropout ratio for encoder
        """
        super().__init__()

        decoder_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, decoder_dropout, activation, batch_first=True)
        decoder_norm = nn.LayerNorm(d_model)
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_decoder_layers, decoder_norm)

        self.decoder_input = decoder_input

        self.decoder_output = decoder_output

    def forward(self, memory, mode3, callsign, seq_len, y=None):
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
        memory_mask = self.get_memory_mask(seq_len)
        sequence_len = memory.shape[1]
        batch_size = memory.shape[0]

        # initialize start token , [0,n_class-1] are actual classes , start token is n_class
        decoder_start_token = torch.ones(
            [batch_size], dtype=torch.long) * self.decoder_input.labels_map['start']

        decoder_inputs = torch.zeros(
            batch_size, sequence_len + 1, self.decoder_input.embedding_dim).cuda()

        decoder_inputs[:, 0, :] = self.decoder_input(
            decoder_start_token.cuda())
        # if y in given, do teacher forcing
        if y != None:
            mode3_input = mode3[:, 0, :]
            callsign_input = callsign[:, 0, :]
            decoder_inputs[:, 1:, :] = self.decoder_input(y)
            target_mask = self.get_target_mask(sequence_len)
            decoder_trans_output = self.transformer_decoder(
                decoder_inputs[:, :sequence_len, :], memory, tgt_mask=target_mask, memory_key_padding_mask=memory_mask)
            decoder_outputs = self.decoder_output(
                decoder_trans_output, mode3_input, callsign_input)

        else:
            decoder_outputs = torch.zeros(
                batch_size, sequence_len, self.decoder_output.output_dim).cuda()

            for t in range(sequence_len):
                mode3_input = mode3[:, t, :]
                callsign_input = callsign[:, t, :]

                target_mask = self.get_target_mask(t+1)

                # [batch,embedding]
                decoder_trans_output = self.transformer_decoder(
                    decoder_inputs[:, :t+1, :], memory, tgt_mask=target_mask, memory_key_padding_mask=memory_mask)[:, -1, :]

                decoder_outputs[:, t, :] = self.decoder_output(
                    decoder_trans_output, mode3_input, callsign_input)

                target_token = decoder_outputs[:, t, :].argmax(-1)
                decoder_inputs[:, t+1, :] = self.decoder_input(target_token)

        return decoder_outputs

    def get_pad_mask(self, lengths, max_len=None, dtype=torch.bool):
        if max_len is None:
            max_len = lengths.max()
        row_vector = torch.arange(0, max_len, 1)  # size (seq_len)
        matrix = torch.unsqueeze(lengths, dim=-1)  # size = N ,1
        mask = row_vector.cuda() >= matrix
        mask.type(dtype)
        return mask

    def get_target_mask(self, sz):
        return generate_square_subsequent_mask(sz).cuda()

    def get_memory_mask(self, lengths, max_len=None):
        return self.get_pad_mask(lengths, max_len)


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
