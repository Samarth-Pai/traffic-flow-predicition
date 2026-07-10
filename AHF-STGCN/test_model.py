# test_model.py

import torch

from models.ahf_stgcn import AHF_STGCN

x = torch.randn(
    8,
    1,
    207,
    12
)

model = AHF_STGCN(
    num_nodes=207,
    input_dim=1,
    hidden_dim=64,
    horizon=12
)

out = model(x)

print(out.shape)