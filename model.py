import torch.nn as nn
import torch.nn.functional as F
import torch

class LSTMEmbedding(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size=128,
                 vocab_size = 256,
                 embedding_dim = 256,
                 bidirectional = False,
                 depth = 1):

        super(LSTMEmbedding, self).__init__()

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

        self.lstm = nn.LSTM(
                input_size = embedding_dim,
                hidden_size =  hidden_size,
                batch_first = True,
                num_layers = depth,
                bidirectional = bidirectional
                            )

        self.relu = nn.ReLU()

        if bidirectional:
            self.D = 2
        else:
            self.D = 1

        self.linear = nn.Linear(input_size*hidden_size, output_size)

    def forward(self, input):
        embeddings = self.embedding(input)

        h0 = torch.zeros(())
        output, (h0, c0) = self.lstm(embeddings)

        flattened_output = output.reshape(input.size()[0], -1)

        output = self.relu(flattened_output)

        output = self.linear(flattened_output)

        norm = torch.norm(output, p=2, dim=-1, keepdim=True)

        output = output / torch.clamp(norm, min=1.0)  # Ensures norm <= 1

        return(output)


if __name__ == "__main__":
    batch_size = 2
    sequence_length = 11
    num_tokens = 5  # Number of token IDs in each sequence
    vocab_size = 100

    model = LSTMEmbedding(
            input_size = 11,
            hidden_size = 6,
            output_size = 100,
            vocab_size = vocab_size,
            embedding_dim = 256)

    token_ids = torch.randint(0, vocab_size, (batch_size, sequence_length))

    out = model(token_ids)
