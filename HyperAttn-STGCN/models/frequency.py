import torch.nn as nn
import torch

class FrequencyBranch(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):

        freq = torch.fft.rfft(
            x,
            dim=-1
        )

        freq = freq * self.gate(
            freq.abs()
        )

        out = torch.fft.irfft(
            freq,
            n=x.shape[-1],
            dim=-1
        )

        return out