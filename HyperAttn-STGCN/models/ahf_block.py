# models/ahf_block.py

import torch
import torch.nn as nn

from models.temporal import MultiScaleTemporalConv
from models.hyperedge import AdaptiveHyperedgeGenerator
from models.hypergraph_attention import HypergraphAttention
from models.fusion import SpatialTemporalFusionGate, TripleFusion
from models.frequency import FrequencyBranch


class AHFBlock(nn.Module):

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

        self.frequency = FrequencyBranch(
            channels
        )

        self.hyperedge_generator = (
            AdaptiveHyperedgeGenerator(
                num_nodes=num_nodes,
                embed_dim=channels,
                num_hyperedges=num_hyperedges
            )
        )

        self.hypergraph_attention = (
            HypergraphAttention(
                in_channels=channels,
                out_channels=channels
            )
        )

        self.fusion = TripleFusion(
            channels
        )

        self.norm = nn.LayerNorm(channels)

    def forward(self, x):

        residual = x

        temporal_features = self.temporal(x)

        frequency_features = self.frequency(
            temporal_features
        )

        H = self.hyperedge_generator(
            temporal_features
        )

        spatial_features = self.hypergraph_attention(
            temporal_features,
            H
        )

        out = self.fusion(
            temporal_features,
            spatial_features,
            frequency_features
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