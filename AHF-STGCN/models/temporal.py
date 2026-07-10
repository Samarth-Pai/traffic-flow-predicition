import torch
import torch.nn as nn


class MultiScaleTemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.k2 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, 2),
            padding=(0, 1)
        )

        self.k3 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, 3),
            padding=(0, 1)
        )

        self.k5 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, 5),
            padding=(0, 2)
        )

        self.fusion = nn.Conv2d(
            out_channels * 3,
            out_channels,
            kernel_size=1
        )

        self.relu = nn.ReLU()

    def forward(self, x):

        x1 = self.k2(x)
        x2 = self.k3(x)
        x3 = self.k5(x)

        min_t = min(
            x1.size(-1),
            x2.size(-1),
            x3.size(-1)
        )

        x1 = x1[..., :min_t]
        x2 = x2[..., :min_t]
        x3 = x3[..., :min_t]

        x = torch.cat([x1, x2, x3], dim=1)

        x = self.fusion(x)

        return self.relu(x)