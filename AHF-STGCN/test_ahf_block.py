# test_ahf_block.py

import torch

from models.ahf_block import AHFBlock

x = torch.randn(
    8,
    64,
    207,
    12
)

model = AHFBlock(
    num_nodes=207,
    channels=64
)

out = model(x)

print(out.shape)