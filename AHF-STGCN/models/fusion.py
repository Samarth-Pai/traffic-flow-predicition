# models/fusion.py

import torch
import torch.nn as nn


class SpatialTemporalFusionGate(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.spatial_proj = nn.Conv2d(
            channels,
            channels,
            kernel_size=1
        )

        self.temporal_proj = nn.Conv2d(
            channels,
            channels,
            kernel_size=1
        )

        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        spatial_features,
        temporal_features
    ):
        """
        spatial_features:
        [B,C,N,T]

        temporal_features:
        [B,C,N,T]
        """

        gate = self.sigmoid(
            self.spatial_proj(spatial_features)
            +
            self.temporal_proj(temporal_features)
        )

        output = (
            gate * spatial_features
            +
            (1 - gate) * temporal_features
        )

        return output