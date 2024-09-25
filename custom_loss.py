import torch.nn as nn
import torch.nn.functional as F
import torch

from math import sqrt

from dataset import OrganizationDataset
from model import LSTMEmbedding

# FOR DEVELOPMENT ONLY:
from utils import n_letters, lineToTensor

class HardTripletLoss(nn.Module):
    def __init__(self, margin):
        super(HardTripletLoss, self).__init__()

        self.margin = margin

    def forward(self, anchor_embedding, positive_embeddings, negative_embeddings):
        # anchor is 1 x embedding_dim
        # positive_embeddings is n x embedding_dim
        # negative_embeddings is n x embedding_dim


        # Step 1, find the row that is furthest away from anchor among positive
        # embeddings
        embedding_dim = anchor_embedding.shape[1]


        max_dist = float('-inf')
        max_dist_index = 0
        with torch.no_grad():
            for i in range(positive_embeddings.shape[0]):
                dist = torch.norm(anchor_embedding - positive_embeddings[i, : ], p = 2).item()
                if dist > max_dist and dist < self.margin:
                    max_dist = dist
                    max_dist_index = i

        min_dist = float('inf')
        min_dist_index = None
        with torch.no_grad():
            for i in range(negative_embeddings.shape[0]):
                dist = torch.norm(anchor_embedding - negative_embeddings[i, : ], p = 2).item()
                if dist < min_dist:
                    min_dist = dist
                    min_dist_index = i


        max_dist_to_positive = torch.norm(anchor_embedding - positive_embeddings[max_dist_index, : ], p = 2)
        min_dist_to_negative = torch.norm(anchor_embedding - negative_embeddings[min_dist_index, : ], p = 2)

        return(
                torch.clamp(max_dist_to_positive - min_dist_to_negative + self.margin, min=0.0)
                )


if __name__ == "__main__":

    dataset = OrganizationDataset()
    examples = dataset.random_set(3)

    loss = HardTripletLoss(.2)

    model = LSTMEmbedding(40, 2)


    anchor = model(examples[0])
    positives = model(examples[1])
    negatives = model(examples[2])


    print(loss(anchor, positives[0, :].unsqueeze(0), positives[0, :].unsqueeze(0)).item())
    print(anchor.shape[1])



