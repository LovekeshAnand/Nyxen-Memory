import time
import logging
import threading
import os
import ctypes
import numpy as np
from typing import Tuple

logger = logging.getLogger("cgm.safety")

# Global Re-entrant Thread Lock to serialize GPU access
_gpu_lock = threading.RLock()

# Load Rust dynamic library for high-performance memory queries, similarity, and thermals
_dll = None
_dll_name = "cgm_rust_speedups.dll"
if os.name != "nt":
    import platform
    if platform.system() == "Darwin":
        _dll_name = "libcgm_rust_speedups.dylib"
    else:
        _dll_name = "libcgm_rust_speedups.so"

_dll_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), _dll_name)

if os.path.exists(_dll_path):
    try:
        _dll = ctypes.CDLL(_dll_path)
        _dll.get_system_ram.argtypes = [ctypes.POINTER(ctypes.c_ulonglong), ctypes.POINTER(ctypes.c_ulonglong)]
        _dll.get_system_ram.restype = ctypes.c_int
        _dll.get_gpu_vram.argtypes = [ctypes.POINTER(ctypes.c_ulonglong), ctypes.POINTER(ctypes.c_ulonglong)]
        _dll.get_gpu_vram.restype = ctypes.c_int
        _dll.compute_cosine_similarity.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        _dll.compute_cosine_similarity.restype = ctypes.c_float
        _dll.get_gpu_temperature.argtypes = [ctypes.POINTER(ctypes.c_uint)]
        _dll.get_gpu_temperature.restype = ctypes.c_int
        logger.info(f"[Safety] Loaded Rust speedups library: {_dll_path}")
    except Exception as e:
        logger.error(f"[Safety] Failed to load Rust speedups library: {e}")
else:
    logger.info("[Safety] Rust speedups library not found. Using pure Python fallback implementations.")

def get_system_ram_info() -> Tuple[int, int]:
    """
    Returns (free_bytes, total_bytes) of system physical RAM.
    """
    if _dll is not None:
        free_bytes = ctypes.c_ulonglong(0)
        total_bytes = ctypes.c_ulonglong(0)
        res = _dll.get_system_ram(ctypes.byref(free_bytes), ctypes.byref(total_bytes))
        if res == 0:
            return free_bytes.value, total_bytes.value
    try:
        import psutil
        vmem = psutil.virtual_memory()
        return vmem.available, vmem.total
    except Exception:
        pass
    return 0, 0

def get_gpu_vram_info() -> Tuple[int, int]:
    """
    Returns (free_bytes, total_bytes) of GPU VRAM.
    """
    if _dll is not None:
        free_bytes = ctypes.c_ulonglong(0)
        total_bytes = ctypes.c_ulonglong(0)
        res = _dll.get_gpu_vram(ctypes.byref(free_bytes), ctypes.byref(total_bytes))
        if res == 0:
            return free_bytes.value, total_bytes.value
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.mem_get_info()
    except Exception:
        pass
    return 0, 0

def get_gpu_temperature_info() -> int:
    """
    Returns current GPU temperature in Celsius, or 0 if query is unsupported/fails.
    """
    if _dll is not None:
        temp = ctypes.c_uint(0)
        res = _dll.get_gpu_temperature(ctypes.byref(temp))
        if res == 0:
            return temp.value
    return 0

class GPULockManager:
    """
    Ensures thread-safe serialized access to the GPU/CUDA device.
    Uses a re-entrant mutual exclusion lock to serialize concurrent thread execution
    on the RTX A2000 while allowing safe recursive acquisitions from the same thread.
    """
    @staticmethod
    def acquire():
        _gpu_lock.acquire()

    @staticmethod
    def release():
        _gpu_lock.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class GPUMemoryGuard:
    """
    Monitored memory allocator guards to prevent Out-Of-Memory (OOM) crashes on the GPU.
    """
    def __init__(self, warning_threshold_mb: float = 800.0, critical_threshold_mb: float = 300.0):
        self.warning_threshold = warning_threshold_mb * 1024 * 1024
        self.critical_threshold = critical_threshold_mb * 1024 * 1024

    def check_vram(self) -> Tuple[int, int]:
        """
        Queries available and total GPU VRAM via Rust/NVML, falling back to PyTorch.
        """
        return get_gpu_vram_info()

    def check_system_ram(self) -> Tuple[int, int]:
        """
        Queries available and total system physical RAM.
        """
        return get_system_ram_info()

    def enforce_safety(self, model_size_est_mb: float = 0.0):
        """
        Checks available VRAM. Clears PyTorch cache if memory is low.
        Raises an exception if available VRAM is critically low.
        """
        try:
            import torch
            if not torch.cuda.is_available():
                return
                
            free_bytes, total_bytes = self.check_vram()
            free_mb = free_bytes / (1024 * 1024)
            logger.info(f"[Safety] Available GPU VRAM: {free_mb:.1f} MB / {total_bytes / (1024 * 1024):.1f} MB")
            
            # Flush cache if below warning threshold
            if free_bytes < self.warning_threshold:
                logger.warning(f"[Safety Warning] Low VRAM ({free_mb:.1f} MB free). Flushing PyTorch CUDA cache...")
                torch.cuda.empty_cache()
                free_bytes, _ = self.check_vram()
                free_mb = free_bytes / (1024 * 1024)
                logger.info(f"[Safety] VRAM after cache flush: {free_mb:.1f} MB")
                
            # Hard crash prevention if below critical threshold or model estimation is too large
            req_bytes = model_size_est_mb * 1024 * 1024
            if free_bytes < self.critical_threshold or (req_bytes > 0 and free_bytes < req_bytes):
                raise RuntimeError(
                    f"CRITICAL: Insufficient VRAM to run operation. "
                    f"Required est: {model_size_est_mb:.1f} MB, Free: {free_mb:.1f} MB. "
                    f"Execution aborted to prevent hardware OOM crash."
                )
        except ImportError:
            logger.info("[Safety] PyTorch not installed yet, skipping CUDA VRAM checks.")


class ThermalGuard:
    """
    Prevents laptop workstation GPUs from overheating under sustained heavy training load.
    Instead of hard pauses (which cause sudden temperature drops and thermal cycling stresses
    due to mismatched coefficients of thermal expansion), it implements a dynamic, proportional
    pacing control loop that introduces tiny, smooth micro-delays to keep the temperature stable.
    """
    def __init__(self, target_temp_c: float = 75.0, max_temp_c: float = 85.0, base_pacing_s: float = 0.005, *args, **kwargs):
        self.target_temp = target_temp_c
        self.max_temp = max_temp_c
        self.base_pacing = base_pacing_s
        logger.info(f"[Safety] ThermalGuard initialized. Target Temp: {target_temp_c}°C, Max Temp: {max_temp_c}°C.")

    def cycle(self):
        """
        Queries the GPU temperature and applies proportional pacing delay.
        """
        current_temp = get_gpu_temperature_info()
        
        # If temp query is unavailable (returns 0), fall back to a small base pacing sleep
        if current_temp == 0:
            try:
                import torch
                if torch.cuda.is_available():
                    time.sleep(self.base_pacing)
            except ImportError:
                pass
            return

        # Proportional thermal throttling pacing
        if current_temp >= self.target_temp:
            excess = current_temp - self.target_temp
            range_temp = max(1.0, self.max_temp - self.target_temp)
            ratio = min(1.0, excess / range_temp)
            
            # Quadratic scaling of delay from 10ms to 300ms
            pacing_delay = 0.01 + 0.29 * (ratio ** 2)
            
            logger.warning(
                f"[Safety Warning] GPU temperature at {current_temp}°C (target: {self.target_temp}°C). "
                f"Applying thermal pacing delay of {pacing_delay*1000:.1f}ms."
            )
            time.sleep(pacing_delay)
        else:
            if self.base_pacing > 0:
                time.sleep(self.base_pacing)


class InputBufferGuard:
    """
    Sanitizes string inputs and validates sizes before they enter vectorization and model parsing.
    """
    @staticmethod
    def sanitize_string(text: str, max_chars: int = 10000) -> str:
        """
        Prevents stack overflow or extremely long string allocations.
        """
        if not text:
            return ""
        # Truncate
        if len(text) > max_chars:
            logger.warning(f"[Safety] String input too long ({len(text)} chars). Truncating to {max_chars} chars.")
            text = text[:max_chars]
        return text

    @staticmethod
    def validate_tensor_bounds(sequence_length: int, max_bound: int = 2048):
        """
        Ensures model doesn't allocate massive attention matrices.
        """
        if sequence_length > max_bound:
            raise ValueError(f"Input sequence length {sequence_length} exceeds hardware safety bound of {max_bound} tokens.")

def c_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calls the C++ AVX2-optimized cosine similarity function.
    """
    if _dll is None:
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))
        
    a_f32 = a.astype(np.float32)
    b_f32 = b.astype(np.float32)
    dim = a_f32.shape[-1]
    
    a_ptr = a_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    b_ptr = b_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    
    return float(_dll.compute_cosine_similarity(a_ptr, b_ptr, dim))
