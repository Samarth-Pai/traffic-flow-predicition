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
    
class TripleFusion(nn.Module):
    def __init__(self,c):
        super().__init__()

        self.weight_generator = nn.Conv2d(
            c*3,
            3,
            kernel_size=1
        )

    def forward(
        self,
        temporal,
        hypergraph,
        frequency
    ):

        weights = self.weight_generator(
            torch.cat(
                [
                    temporal,
                    hypergraph,
                    frequency
                ],
                dim=1
            )
        )

        weights = torch.softmax(
            weights,
            dim=1
        )

        return (
            weights[:,0:1]*temporal
            +
            weights[:,1:2]*hypergraph
            +
            weights[:,2:3]*frequency
        )