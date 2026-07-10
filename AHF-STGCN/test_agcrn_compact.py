# test_agcrn_compat.py

import torch

from models.ahf_stgcn import AHF_STGCN

x = torch.randn(
    64,
    12,
    207,
    1
)

model = AHF_STGCN(
    num_nodes=207,
    input_dim=1,
    hidden_dim=64,
    horizon=12
)

out = model(x)

print("input :", x.shape)
print("output:", out.shape)