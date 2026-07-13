import torch
import torch.nn as nn

class SpatialTransformer(nn.Module):

    def __init__(self, channels, heads=4):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=heads,
            batch_first=True
        )

        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        # x : [B,C,N,T]

        B,C,N,T = x.shape

        x = x.permute(0,3,2,1)

        x = x.reshape(B*T, N, C)

        out,_ = self.attn(
            x,
            x,
            x
        )

        out = self.norm(out + x)

        out = out.reshape(B,T,N,C)

        return out.permute(
            0,3,2,1
        )