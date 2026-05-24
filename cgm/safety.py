import time
import logging
import threading
from typing import Tuple

logger = logging.getLogger("cgm.safety")
logging.basicConfig(level=logging.INFO)

# Global Thread Lock & Semaphore to serialize GPU access
_gpu_lock = threading.Lock()
_gpu_semaphore = threading.BoundedSemaphore(value=1)

class GPULockManager:
    """
    Ensures thread-safe serialized access to the GPU/CUDA device.
    Uses mutual exclusion locks and semaphores to block multiple threads
    from concurrently launching heavy kernels on the RTX A2000.
    """
    @staticmethod
    def acquire():
        _gpu_semaphore.acquire()
        _gpu_lock.acquire()

    @staticmethod
    def release():
        _gpu_lock.release()
        _gpu_semaphore.release()

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
        Queries the GPU driver via PyTorch to get (free_bytes, total_bytes).
        If PyTorch is not compiled with CUDA or no GPU is available, returns (0, 0).
        """
        try:
            import torch
            if torch.cuda.is_available():
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                return free_bytes, total_bytes
        except ImportError:
            pass
        return 0, 0

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
    Inserts periodic cooling rests during iterative runs.
    """
    def __init__(self, batch_limit_before_cooldown: int = 50, cooldown_seconds: float = 3.0):
        self.batch_limit = batch_limit_before_cooldown
        self.cooldown_seconds = cooldown_seconds
        self.batch_count = 0
        self.last_cooldown_time = time.time()

    def cycle(self):
        """
        Tracks batches processed. Sleep if limit is reached to allow hardware to cool down.
        """
        self.batch_count += 1
        if self.batch_count >= self.batch_limit:
            logger.info(f"[Safety] ThermalGuard: Processing limit reached ({self.batch_count} batches). Pausing for {self.cooldown_seconds}s for GPU cool down...")
            time.sleep(self.cooldown_seconds)
            self.batch_count = 0
            self.last_cooldown_time = time.time()


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
