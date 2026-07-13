# models/hyperedge.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveHyperedgeGenerator(nn.Module):
    def __init__(
        self,
        num_nodes,
        embed_dim,
        num_hyperedges
    ):
        super().__init__()

        self.hyperedge_embeddings = nn.Parameter(
            torch.randn(num_hyperedges, embed_dim)
        )

    def forward(self, x):

        self.node_encoder = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1)
        )

        node_repr = x.mean(dim=-1)
        node_repr = self.node_encoder(node_repr)

        scores = torch.einsum(
            "bcn,ec->bne",
            node_repr,
            self.hyperedge_embeddings
        )

        H = torch.softmax(scores, dim=-1)

        Dv = H.sum(dim=-1, keepdim=True) + 1e-6

        return H / Dv