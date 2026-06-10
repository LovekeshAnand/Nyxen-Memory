# Conversational Graph Memory (CGM) Package
import sys
import os

# Redirect Hugging Face cache to D: drive if it exists and is a local fixed disk to prevent C: drive exhaustion
if "HF_HOME" not in os.environ:
    if os.path.exists("D:\\"):
        is_local_fixed = True
        if os.name == 'nt':
            import ctypes
            try:
                drive_type = ctypes.windll.kernel32.GetDriveTypeW("D:\\")
                is_local_fixed = (drive_type == 3) # 3 = DRIVE_FIXED
            except Exception:
                is_local_fixed = False
        if is_local_fixed:
            os.environ["HF_HOME"] = "D:\\huggingface_cache"

# Add custom PyTorch CUDA path if it exists
cuda_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".torch_cuda")
if os.path.exists(cuda_path) and cuda_path not in sys.path:
    sys.path.insert(0, cuda_path)
    if "torch" not in sys.modules:
        print(f"[CGM] Custom path prepended: {cuda_path} (loading CUDA-enabled PyTorch)")

# Expose core classes at package level
from cgm.core.pipeline import CGMPipeline
from cgm.database.schema import SQLiteGraphStore, HybridMemoryObject
from cgm.safety.safety import GPULockManager, GPUMemoryGuard, ThermalGuard, InputBufferGuard
from cgm.retrieval.rag_retriever import RAGRetriever, TurnMemory
