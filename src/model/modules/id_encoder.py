import torch
import torch.nn as nn
import pytorch_lightning as pl


class IdEncoder(pl.LightningModule):
    """
    Id Encoder Object

    This encodes any identifiers (callsign, modes) by using character level tokens into a dense vector.
    Bidirectional LSTM is used as the encoder.
    Forward and Backwards direction of each layer of Hidden and Cell are combined 
    using a linear layer

    """

    def __init__(self, n_tokens, n_token_embedding, id_embed_dim, n_layers):
        """
        Initialises the encoder object.
        :param hid_dim: hidden dimensions in each layer
        :param n_layers: number of layers (same for encoder and decoder)
        :param n_features: number of features from the dataset
        """
        super().__init__()
        # self.device = torch.device(
        #     "cuda" if torch.cuda.is_available() else "cpu")
        self.token_embedding = nn.Embedding(
            n_tokens, n_token_embedding)
        self.id_embed_dim = id_embed_dim
        self.n_layers = n_layers
        self.rnn = nn.LSTM(n_token_embedding, id_embed_dim, n_layers,
                           bidirectional=True, batch_first=True)
        self.linear = nn.Linear(id_embed_dim*2, id_embed_dim)
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        """
        Forward propagation.
        :param x: features of dataset at every timestamp [batch_size,max_word_len] -> padded to the longest word 
        :param seq_len: actual length of each data sequence [batch_size]
        :return: hidden state of the last layer in the encoder;
                 outputs the outputs of last layer at every timestamps

        hidden: [batch_size, n_layer, hid_dim]
        cell: [batch_size, n_layer, hid_dim]
        output [batch_size, sequence_len, n_directions*hid_dim] , note n_direction = 2 for bidirectional-lstm 
        """

        # [batch_size,max_word_en,n_token_embedding]
        x_embedding = self.token_embedding(x)

        output, (hidden, cell) = self.rnn(x_embedding)

        output = output.view(
            output.shape[0], output.shape[1], 2, self.id_embed_dim)

        last_forward = output[:, -1, 0, :]  # [batch_size,hid_dim]
        first_backwards = output[:, 0, 1, :]  # [batch_size,hid_dim]
        id_embedding = self.activation(
            (self.linear(torch.cat([last_forward, first_backwards], -1))))
        return id_embedding
