# models/ahf_stgcn.py

import torch
import torch.nn as nn

from models.ahf_block import AHFBlock


class AHF_STGCN(nn.Module):

    def __init__(
        self,
        num_nodes=207,
        input_dim=1,
        hidden_dim=64,
        horizon=12
    ):
        super().__init__()

        self.input_proj = nn.Conv2d(
            input_dim,
            hidden_dim,
            kernel_size=1
        )

        self.block1 = AHFBlock(
            num_nodes=num_nodes,
            channels=hidden_dim
        )

        self.block2 = AHFBlock(
            num_nodes=num_nodes,
            channels=hidden_dim
        )

        self.forecast_head = nn.Sequential(
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                hidden_dim,
                horizon,
                kernel_size=(1, 12)
            )
        )

    def forward(
        self,
        source,
        target=None,
        teacher_forcing_ratio=0
    ):
        """
        source:
        [B,T,N,F]

        output:
        [B,12,N,1]
        """

        x = source.permute(
            0,
            3,
            2,
            1
        )

        x = self.input_proj(x)

        x = self.block1(x)

        x = self.block2(x)

        x = self.forecast_head(x)

        x = x.permute(
            0,
            1,
            2,
            3
        )

        return x