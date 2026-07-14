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

        self.node_encoder = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, 1),
            nn.ReLU(),
            nn.Conv1d(embed_dim, embed_dim, 1)
        )

        self.hyperedge_embeddings = nn.Parameter(
            torch.randn(num_hyperedges, embed_dim)
        )

    def forward(self, x):


        node_repr = x.mean(dim=-1)

        node_repr = self.node_encoder(
            node_repr
        )

        scores = torch.einsum(
            "bcn,ec->bne",
            node_repr,
            self.hyperedge_embeddings
        )

        H = torch.softmax(scores, dim=-1)

        topk = 5

        values, indices = torch.topk(
            H,
            topk,
            dim=-1
        )

        mask = torch.zeros_like(H)

        mask.scatter_(
            -1,
            indices,
            1
        )

        H = H * mask

        H = H / (
            H.sum(-1, keepdim=True)
            + 1e-6
        )

        Dv = H.sum(dim=-1, keepdim=True) + 1e-6

        return H / Dv