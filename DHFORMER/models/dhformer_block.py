# models/ahf_block.py

import torch
import torch.nn as nn

from models.temporal import MultiScaleTemporalConv
from models.hyperedge import AdaptiveHyperedgeGenerator
from models.hypergraph_attention import HypergraphAttention
from models.dhformer_fusion import DHFusion
from models.spatial_transformer import SpatialTransformer
from models.temporal_transformer import TemporalTransformer


class DHFormerBlock(nn.Module):

    def __init__(
        self,
        num_nodes,
        channels=64,
        num_hyperedges=64,
        embed_dim=64
    ):
        super().__init__()

        self.temporal = MultiScaleTemporalConv(
            channels,
            channels
        )

        self.hyperedge_generator = AdaptiveHyperedgeGenerator(
            num_nodes=num_nodes,
            embed_dim=channels,
            num_hyperedges=num_hyperedges
        )

        self.hypergraph_attention = HypergraphAttention(
            in_channels=channels,
            out_channels=channels
        )

        self.spatial_transformer = SpatialTransformer(
            channels=channels,
            heads=4
        )

        self.temporal_transformer = TemporalTransformer(
            channels=channels,
            heads=4
        )

        self.fusion = DHFusion(channels)

        self.norm = nn.LayerNorm(channels)

    # OUTSIDE __init__
    def forward(self, x):

        residual = x

        temporal = self.temporal(x)

        H = self.hyperedge_generator(
            temporal
        )

        hypergraph = self.hypergraph_attention(
            temporal,
            H
        )

        spatial = self.spatial_transformer(
            hypergraph
        )

        temporal_attn = self.temporal_transformer(
            spatial
        )

        out = self.fusion(
            temporal,
            hypergraph,
            temporal_attn
        )

        out = out + residual

        B, C, N, T = out.shape

        out = out.permute(
            0, 2, 3, 1
        )

        out = self.norm(out)

        out = out.permute(
            0, 3, 1, 2
        )

        return out