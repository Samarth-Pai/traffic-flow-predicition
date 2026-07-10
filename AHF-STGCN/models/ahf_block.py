# models/ahf_block.py

import torch
import torch.nn as nn

from models.temporal import MultiScaleTemporalConv
from models.hyperedge import AdaptiveHyperedgeGenerator
from models.hypergraph_attention import HypergraphAttention
from models.fusion import SpatialTemporalFusionGate


class AHFBlock(nn.Module):

    def __init__(
        self,
        num_nodes,
        channels=64,
        num_hyperedges=64,
        embed_dim=32
    ):
        super().__init__()

        self.temporal = MultiScaleTemporalConv(
            channels,
            channels
        )

        self.hyperedge_generator = (
            AdaptiveHyperedgeGenerator(
                num_nodes=num_nodes,
                embed_dim=embed_dim,
                num_hyperedges=num_hyperedges
            )
        )

        self.hypergraph_attention = (
            HypergraphAttention(
                in_channels=channels,
                out_channels=channels
            )
        )

        self.fusion = (
            SpatialTemporalFusionGate(
                channels
            )
        )

        self.norm = nn.LayerNorm(channels)

    def forward(self, x):

        residual = x

        temporal_features = self.temporal(x)

        H = self.hyperedge_generator()

        spatial_features = self.hypergraph_attention(
            temporal_features,
            H
        )

        out = self.fusion(
            spatial_features,
            temporal_features
        )

        out = out + residual

        B, C, N, T = out.shape

        out = out.permute(
            0,
            2,
            3,
            1
        )

        out = self.norm(out)

        out = out.permute(
            0,
            3,
            1,
            2
        )

        return out