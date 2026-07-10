# test_fusion.py

import torch

from models.fusion import (
    SpatialTemporalFusionGate
)

spatial = torch.randn(
    8,
    64,
    207,
    12
)

temporal = torch.randn(
    8,
    64,
    207,
    12
)

fusion = SpatialTemporalFusionGate(
    channels=64
)

out = fusion(
    spatial,
    temporal
)

print(out.shape)