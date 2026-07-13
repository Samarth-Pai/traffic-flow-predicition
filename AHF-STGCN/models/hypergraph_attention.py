# models/hypergraph_attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class HypergraphAttention(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels
    ):
        super().__init__()

        self.query = nn.Linear(
            in_channels,
            out_channels
        )

        self.key = nn.Linear(
            in_channels,
            out_channels
        )

        self.value = nn.Linear(
            in_channels,
            out_channels
        )

        self.scale = out_channels ** 0.5

    def forward(self, x, H):
        """
        x: [B,C,N,T]
        H: [N,E]

        output:
        [B,out_channels,N,T]
        """

        B, C, N, T = x.shape

        x = x.permute(
            0,
            3,
            2,
            1
        )

        hyper_nodes = torch.einsum(
            "btnc,bne->btec",
            x,
            H
        )

        Q = self.query(hyper_nodes)
        K = self.key(hyper_nodes)
        V = self.value(hyper_nodes)

        scores = torch.matmul(
            Q,
            K.transpose(-1, -2)
        ) / self.scale

        attn = F.softmax(
            scores,
            dim=-1
        )

        hyper_out = torch.matmul(
            attn,
            V
        )

        node_out = torch.einsum(
            "btec,bne->btnc",
            hyper_out,
            H
        )

        node_out = node_out.permute(
            0,
            3,
            2,
            1
        )

        return node_out