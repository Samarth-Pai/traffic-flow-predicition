from models.hyperedge import AdaptiveHyperedgeGenerator

gen = AdaptiveHyperedgeGenerator(
    num_nodes=207,
    embed_dim=32,
    num_hyperedges=64
)

H = gen()

print(H.shape)