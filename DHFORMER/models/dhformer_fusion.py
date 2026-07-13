import torch
import torch.nn as nn

class DHFusion(nn.Module):

    def __init__(self, channels):

        super().__init__()

        self.gate = nn.Sequential(
            nn.Conv2d(
                channels*3,
                channels,
                1
            ),
            nn.Sigmoid()
        )

    def forward(
        self,
        temporal,
        hypergraph,
        transformer
    ):

        fusion = torch.cat(
            [
                temporal,
                hypergraph,
                transformer
            ],
            dim=1
        )

        alpha = self.gate(fusion)

        return (
            alpha * temporal
            + (1-alpha)/2 * hypergraph
            + (1-alpha)/2 * transformer
        )