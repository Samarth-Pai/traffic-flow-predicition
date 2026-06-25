import torch

# 1. Check if CUDA (GPU support) is available
gpu_available = torch.cuda.is_available()
print(f"Is GPU available? {gpu_available}")

if gpu_available:
    # 2. Get the name of your GPU
    print(f"GPU Device Name: {torch.cuda.get_device_name(0)}")
    
    # 3. Check the current active device index
    print(f"Current Device Index: {torch.cuda.current_device()}")
