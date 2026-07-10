# test_attention.py

import torch

from models.hyperedge import (
    AdaptiveHyperedgeGenerator
)

from models.hypergraph_attention import (
    HypergraphAttention
)

x = torch.randn(
    8,
    64,
    207,
    12
)

gen = AdaptiveHyperedgeGenerator(
    num_nodes=207,
    embed_dim=32,
    num_hyperedges=64
)

H = gen()

layer = HypergraphAttention(
    in_channels=64,
    out_channels=64
)

out = layer(
    x,
    H
)

print(out.shape)