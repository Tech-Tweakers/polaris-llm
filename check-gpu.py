import torch

# Check if CUDA is available
cuda_available = torch.cuda.is_available()

# Print the result
if cuda_available:
    print("CUDA (GPU support) is available in PyTorch!")
    # Get the number of GPUs available
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
    # Get the name of the first GPU
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU Name: {gpu_name}")
else:
    print("CUDA (GPU support) is not available in PyTorch. Only CPU is available.")

