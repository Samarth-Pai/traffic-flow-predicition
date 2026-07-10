# count_params.py

from models.ahf_stgcn import AHF_STGCN

model = AHF_STGCN()

total = sum(
    p.numel()
    for p in model.parameters()
)

print(f"Parameters: {total:,}")