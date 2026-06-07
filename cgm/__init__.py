# Conversational Graph Memory (CGM) Package
import sys
import os

# Redirect Hugging Face cache to D: drive if it exists to prevent C: drive exhaustion
if os.path.exists("D:\\"):
    os.environ["HF_HOME"] = "d:/huggingface_cache"

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
