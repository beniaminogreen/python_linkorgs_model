
import torch.nn as nn
import torch.nn.functional as F
import torch

import torchtune
import math

# Code taken from
# https://medium.com/@hunter-j-phillips/positional-encoding-7a93db4109e6
class PositionalEncoding(nn.Module):
  def __init__(self, d_model: int, dropout: float = 0.1, max_length: int = 5000):
    """
    Args:
      d_model:      dimension of embeddings
      dropout:      randomly zeroes-out some of the input
      max_length:   max sequence length
    """
    # inherit from Module
    super().__init__()

    # initialize dropout
    self.dropout = nn.Dropout(p=dropout)

    # create tensor of 0s
    pe = torch.zeros(max_length, d_model)

    # create position column
    k = torch.arange(0, max_length).unsqueeze(1)

    # calc divisor for positional encoding
    div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
    )

    # calc sine on even indices
    pe[:, 0::2] = torch.sin(k * div_term)

    # calc cosine on odd indices
    pe[:, 1::2] = torch.cos(k * div_term)

    # add dimension
    pe = pe.unsqueeze(0)

    # buffers are saved in state_dict but not trained by the optimizer
    self.register_buffer("pe", pe)

  def forward(self, x):
    """
    Args:
      x:        embeddings (batch_size, seq_length, d_model)

    Returns:
                embeddings + positional encodings (batch_size, seq_length, d_model)
    """
    # add positional encoding to the embeddings
    x = x + self.pe[:, : x.size(1)].requires_grad_(False)

    # perform dropout
    return self.dropout(x)

class TransformerEmbedding(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size=128,
                 vocab_size = 256,
                 embedding_dim = 256,
                 depth = 1):

        super(TransformerEmbedding, self).__init__()

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

        self.pos_encoding_layer = PositionalEncoding(embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=4, batch_first  = True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)

        self.linear = nn.Linear(input_size*embedding_dim, output_size)

    def forward(self, input):
        embeddings = self.embedding(input)

        embeddings = self.pos_encoding_layer(embeddings)

        output = self.transformer_encoder(embeddings)

        flattened_output = output.reshape(input.size()[0], -1)


        output = self.linear(flattened_output)

        norm = torch.norm(output, p=2, dim=-1, keepdim=True)
        output = output / torch.clamp(norm, min=1.0)  # Ensures norm <= 1

        return(output)


if __name__ == "__main__":
    batch_size = 3
    sequence_length = 11
    num_tokens = 5  # Number of token IDs in each sequence
    vocab_size = 100

    model = TransformerEmbedding(
            input_size = 11,
            hidden_size = 6,
            output_size = 100,
            vocab_size = vocab_size,
            embedding_dim = 256)

    token_ids = torch.randint(0, vocab_size, (batch_size, sequence_length))

    out = model(token_ids)

