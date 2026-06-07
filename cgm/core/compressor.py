import time
import torch
import logging
import threading
from typing import Dict, Any, List, Optional, Tuple
from transformers.cache_utils import DynamicCache
from cgm.safety.safety import GPULockManager, GPUMemoryGuard

logger = logging.getLogger("cgm.compressor")

def _get_layer_k_v(cache: DynamicCache, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if hasattr(cache, "key_cache") and cache.key_cache is not None:
        return cache.key_cache[layer_idx], cache.value_cache[layer_idx]
    else:
        return cache.layers[layer_idx].keys, cache.layers[layer_idx].values

def _set_layer_k_v(cache: DynamicCache, layer_idx: int, k_state: torch.Tensor, v_state: torch.Tensor):
    if hasattr(cache, "key_cache") and cache.key_cache is not None:
        cache.key_cache[layer_idx] = k_state
        cache.value_cache[layer_idx] = v_state
    else:
        cache.layers[layer_idx].keys = k_state
        cache.layers[layer_idx].values = v_state

def _get_num_layers(cache: DynamicCache) -> int:
    if hasattr(cache, "key_cache") and cache.key_cache is not None:
        return len(cache.key_cache)
    else:
        return len(cache.layers)

def _get_device(cache: DynamicCache) -> Any:
    if hasattr(cache, "key_cache") and cache.key_cache is not None and len(cache.key_cache) > 0:
        return cache.key_cache[0].device
    elif hasattr(cache, "layers") and len(cache.layers) > 0:
        return cache.layers[0].keys.device
    return "cpu"

class KVCompressor:
    """
    Double-buffered, in-flight KV cache compressor.
    Evaluates key-value token importance based on attention history, and dynamically
    evicts/clusters cold-zone entries under strict safety locks and CUDA streams.
    """
    def __init__(self, hot_window: int = 64, kl_threshold: float = 0.05, check_interval_sec: float = 2.0):
        self.hot_window = hot_window
        self.kl_threshold = kl_threshold
        self.check_interval_sec = check_interval_sec
        self.running = False
        self._thread = None
        self._cuda_stream = None
        
    def start_background_monitor(self, pipeline):
        """
        Starts the background monitor thread to proactively compress KV cache when VRAM is low.
        """
        if self.running:
            return
        self.running = True
        self._cuda_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        
        def monitor_loop():
            logger.info("[Compressor] Asynchronous KV Cache Monitor thread started.")
            while self.running:
                try:
                    # Enforce thermal rest check or sleep
                    time.sleep(self.check_interval_sec)
                    
                    # Query GPU memory
                    free_bytes, total_bytes = pipeline.mem_guard.check_vram()
                    if total_bytes == 0:
                        continue # CUDA not active
                        
                    free_mb = free_bytes / (1024 * 1024)
                    
                    # Trigger if free VRAM drops below 800 MB warning threshold
                    if free_mb < 800.0:
                        logger.warning(f"[Compressor] VRAM is low ({free_mb:.1f} MB free). Initiating async compression...")
                        
                        # Get active cache from pipeline
                        active_cache = getattr(pipeline, "active_cache", None)
                        if active_cache is not None and isinstance(active_cache, DynamicCache):
                            seq_len = active_cache.get_seq_length()
                            if seq_len > self.hot_window + 10:
                                # Run compression in our background CUDA stream
                                self.compress_in_stream(pipeline, active_cache)
                except Exception as e:
                    logger.error(f"[Compressor] Error in monitor loop: {e}")
                    
        self._thread = threading.Thread(target=monitor_loop, daemon=True)
        self._thread.start()

    def stop_background_monitor(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None

    def compress_in_stream(self, pipeline, cache: DynamicCache, attention_history: Optional[List[torch.Tensor]] = None):
        """
        Runs compression in the background CUDA stream under the GPULockManager lock.
        """
        if self._cuda_stream is None:
            # CPU fallback
            self.compress(pipeline, cache, attention_history)
            return

        with GPULockManager():
            with torch.cuda.stream(self._cuda_stream):
                self.compress(pipeline, cache, attention_history)

    def compute_salience(self, cache: DynamicCache, attentions: Optional[List[torch.Tensor]]) -> torch.Tensor:
        """
        Computes salience score per position using attention history.
        If history is not provided, defaults to linear decay (more recent = more important).
        """
        seq_len = cache.get_seq_length()
        device = _get_device(cache)
        
        if attentions:
            salience = torch.zeros(seq_len, device=device)
            # Accumulate attention scores
            for step_attn in attentions:
                summed = step_attn.sum(dim=(0, 1, 2))
                key_len = summed.shape[0]
                if key_len <= seq_len:
                    salience[:key_len] += summed
            return salience
        else:
            # Fallback: linear decay scoring (recency bias)
            return torch.linspace(0.1, 1.0, steps=seq_len, device=device)

    def compress(self, pipeline, cache: DynamicCache, attentions: Optional[List[torch.Tensor]] = None) -> bool:
        """
        Performs in-flight compression on the key and value caches.
        Returns True if compression was applied and accepted by the fidelity gate, False otherwise.
        """
        num_layers = _get_num_layers(cache)
        if num_layers == 0:
            return False
            
        seq_len = cache.get_seq_length()
        if seq_len <= self.hot_window:
            return False # Sequence is too short to compress
            
        device = _get_device(cache)
        
        # 1. Compute salience
        salience = self.compute_salience(cache, attentions)
        
        # 2. Identify cold zone indices
        cold_len = seq_len - self.hot_window
        cold_salience = salience[:cold_len]
        
        # 3. Create a snapshot/copy of the original cache for the fidelity check
        original_cache = DynamicCache()
        for idx in range(num_layers):
            k_orig, v_orig = _get_layer_k_v(cache, idx)
            original_cache.update(k_orig.clone(), v_orig.clone(), idx)
            
        # 4. Perform compression (pruning/eviction + clustering of low-salience cold positions)
        threshold = torch.median(cold_salience)
        
        keep_indices = []
        cluster_groups = []
        
        for i in range(cold_len):
            if cold_salience[i] >= threshold:
                keep_indices.append(i)
            else:
                if not cluster_groups or len(cluster_groups[-1]) >= 4:
                    cluster_groups.append([i])
                else:
                    cluster_groups[-1].append(i)
                    
        # Apply changes layer by layer
        for layer_idx in range(num_layers):
            k_state, v_state = _get_layer_k_v(cache, layer_idx)
            
            k_hot = k_state[:, :, cold_len:, :]
            v_hot = v_state[:, :, cold_len:, :]
            
            k_cold_kept = k_state[:, :, keep_indices, :] if keep_indices else torch.empty(k_state.shape[0], k_state.shape[1], 0, k_state.shape[3], device=device)
            v_cold_kept = v_state[:, :, keep_indices, :] if keep_indices else torch.empty(v_state.shape[0], v_state.shape[1], 0, v_state.shape[3], device=device)
            
            k_clustered_list = []
            v_clustered_list = []
            for group in cluster_groups:
                k_group_avg = k_state[:, :, group, :].mean(dim=2, keepdim=True)
                v_group_avg = v_state[:, :, group, :].mean(dim=2, keepdim=True)
                
                k_scale = k_group_avg.abs().max() + 1e-6
                v_scale = v_group_avg.abs().max() + 1e-6
                
                k_quant = (k_group_avg / k_scale * 127).round().clamp(-128, 127).to(torch.int8)
                v_quant = (v_group_avg / v_scale * 127).round().clamp(-128, 127).to(torch.int8)
                
                k_dequant = k_quant.to(torch.float32) / 127 * k_scale
                v_dequant = v_quant.to(torch.float32) / 127 * v_scale
                
                k_clustered_list.append(k_dequant)
                v_clustered_list.append(v_dequant)
                
            if k_clustered_list:
                k_cold_clustered = torch.cat(k_clustered_list, dim=2)
                v_cold_clustered = torch.cat(v_clustered_list, dim=2)
            else:
                k_cold_clustered = torch.empty(k_state.shape[0], k_state.shape[1], 0, k_state.shape[3], device=device)
                v_cold_clustered = torch.empty(v_state.shape[0], v_state.shape[1], 0, v_state.shape[3], device=device)
                
            new_k = torch.cat([k_cold_kept, k_cold_clustered, k_hot], dim=2)
            new_v = torch.cat([v_cold_kept, v_cold_clustered, v_hot], dim=2)
            
            _set_layer_k_v(cache, layer_idx, new_k, new_v)
            
        # 5. Fidelity Gate (KL Divergence check)
        kl_div = self._check_fidelity(pipeline, original_cache, cache)
        logger.info(f"[Compressor] Compression evaluated. Sequence length: {seq_len} -> {cache.get_seq_length()}. KL Divergence: {kl_div:.4f}")
        
        if kl_div > self.kl_threshold:
            # Revert to original cache
            logger.warning(f"[Compressor] KL Divergence ({kl_div:.4f}) exceeds threshold ({self.kl_threshold}). Reverting cache...")
            for idx in range(num_layers):
                k_orig, v_orig = _get_layer_k_v(original_cache, idx)
                _set_layer_k_v(cache, idx, k_orig, v_orig)
            
            try:
                from cgm.visualization.visualize import log_pipeline_event
                log_pipeline_event("compress", {
                    "pre_len": seq_len,
                    "post_len": seq_len,
                    "kl_div": kl_div,
                    "accepted": False,
                    "message": f"Cache compression REJECTED: KL divergence ({kl_div:.4f}) exceeded threshold ({self.kl_threshold})"
                })
            except Exception:
                pass
            return False
            
        logger.info(f"[Compressor] Compression accepted. Saved {seq_len - cache.get_seq_length()} KV positions.")
        
        try:
            from cgm.visualization.visualize import log_pipeline_event
            log_pipeline_event("compress", {
                "pre_len": seq_len,
                "post_len": cache.get_seq_length(),
                "savings": seq_len - cache.get_seq_length(),
                "kl_div": kl_div,
                "accepted": True,
                "message": f"Cache compressed: {seq_len} -> {cache.get_seq_length()} slots. KL divergence: {kl_div:.4f} (Accepted)"
            })
        except Exception:
            pass
            
        return True

    def _check_fidelity(self, pipeline, original_cache: DynamicCache, compressed_cache: DynamicCache) -> float:
        """
        Runs a forward pass on a small dummy input and computes the KL divergence of top-50 logits.
        """
        model = getattr(pipeline, "model", None)
        tokenizer = getattr(pipeline, "tokenizer", None)
        if model is None or tokenizer is None:
            return 0.0
            
        try:
            device = model.device
            dummy_ids = torch.tensor([[tokenizer.eos_token_id or 50256]], dtype=torch.long, device=device)
            
            with GPULockManager():
                test_orig_cache = DynamicCache()
                num_layers_orig = _get_num_layers(original_cache)
                for idx in range(num_layers_orig):
                    k_orig, v_orig = _get_layer_k_v(original_cache, idx)
                    test_orig_cache.update(k_orig.clone(), v_orig.clone(), idx)
                
                outputs_orig = model(dummy_ids, past_key_values=test_orig_cache, use_cache=True)
                logits_orig = outputs_orig.logits[:, -1, :]
                
                test_comp_cache = DynamicCache()
                num_layers_comp = _get_num_layers(compressed_cache)
                for idx in range(num_layers_comp):
                    k_comp, v_comp = _get_layer_k_v(compressed_cache, idx)
                    test_comp_cache.update(k_comp.clone(), v_comp.clone(), idx)
                    
                outputs_comp = model(dummy_ids, past_key_values=test_comp_cache, use_cache=True)
                logits_comp = outputs_comp.logits[:, -1, :]
                
            top_k = min(50, logits_orig.shape[-1])
            vals_orig, idxs_orig = torch.topk(logits_orig, top_k, dim=-1)
            
            vals_comp = torch.gather(logits_comp, -1, idxs_orig)
            
            p = torch.softmax(vals_orig, dim=-1)
            log_p = torch.log_softmax(vals_orig, dim=-1)
            log_q = torch.log_softmax(vals_comp, dim=-1)
            
            kl = torch.sum(p * (log_p - log_q), dim=-1).mean().item()
            return float(kl)
        except Exception as e:
            logger.error(f"[Compressor] Error in fidelity check: {e}")
            return 999.0
