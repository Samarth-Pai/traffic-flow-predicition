# models/hyperedge.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveHyperedgeGenerator(nn.Module):
    def __init__(
        self,
        num_nodes,
        embed_dim=32,
        num_hyperedges=64
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.num_hyperedges = num_hyperedges

        self.node_embeddings = nn.Parameter(
            torch.randn(num_nodes, embed_dim)
        )

        self.hyperedge_embeddings = nn.Parameter(
            torch.randn(num_hyperedges, embed_dim)
        )

    def forward(self):

        scores = torch.matmul(
            self.node_embeddings,
            self.hyperedge_embeddings.T
        )

        H = F.softmax(scores, dim=-1)

        return H
    
