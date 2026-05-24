# Conversational Graph Memory (CGM) Package
import sys
import os

# Add custom PyTorch CUDA path if it exists
cuda_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".torch_cuda")
if os.path.exists(cuda_path) and cuda_path not in sys.path:
    sys.path.insert(0, cuda_path)
    # Only print once to avoid polluting imports
    if "torch" not in sys.modules:
        print(f"[CGM] Custom path prepended: {cuda_path} (loading CUDA-enabled PyTorch)")
