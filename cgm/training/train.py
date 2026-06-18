import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from transformers.cache_utils import DynamicCache
from cgm.database.schema import SQLiteGraphStore
from cgm.retrieval.retriever import SubgraphRetriever
from cgm.core.model import MemoryEncoderNetwork
from cgm.core.pipeline import CGMPipeline
from cgm.safety.safety import GPULockManager, GPUMemoryGuard, ThermalGuard
from cgm.training.train_data import (
    TripleTrainingSample, encode_triple, build_training_samples, split_train_eval
)
from cgm.eval.prompts import wrap_training_prompt


class MEGATrainer:
    """
    Trains the Memory Encoder Network (MEN) adapters to map
    retrieved graph triples to the attention space of the target LLM.
    Keeps the target LLM frozen.
    
    Supports two loss modes:
    - 'generation': Cross-entropy next-token prediction only (original).
    - 'distillation': Generation loss + auxiliary MSE loss anchoring MEN KV outputs
                      to the frozen LLM's own KV states (KV-distillation).
    """
    def __init__(self, pipeline: CGMPipeline, lr: float = 1e-4, distill_lambda: float = 0.5):
        self.pipeline = pipeline
        self.lr = lr
        self.distill_lambda = distill_lambda
        self.thermal_guard = ThermalGuard(target_temp_c=75.0, max_temp_c=85.0)
        self.mem_guard = GPUMemoryGuard()
        self.default_checkpoint_path = "data/checkpoints/men_state_dict.pt"

    def _save_men_checkpoint(self, path: str = None) -> None:
        """Persist trained MEN weights so inference can load the same architecture."""
        checkpoint_path = path or self.default_checkpoint_path
        os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
        torch.save({
            "state_dict": self.pipeline.men.state_dict(),
            "model_name": self.pipeline.model_name,
            "rank": self.pipeline.rank,
            "num_layers": self.pipeline.men.num_layers,
            "num_heads": self.pipeline.men.num_heads,
            "head_dim": self.pipeline.men.head_dim,
            "use_routing": self.pipeline.men.use_routing,
        }, checkpoint_path)

    def _build_men_cache(self, x_triples: torch.Tensor) -> Tuple[DynamicCache, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Runs MEN forward and wraps output into a DynamicCache with position-appropriate RoPE rotation."""
        raw_kv_tuples = self.pipeline.men(x_triples)
        past_key_values = DynamicCache()
        
        # Dynamically find the target attention dtype
        if hasattr(self.pipeline.model, "model") and hasattr(self.pipeline.model.model, "layers") and len(self.pipeline.model.model.layers) > 0:
            attn_dtype = self.pipeline.model.model.layers[0].self_attn.q_proj.weight.dtype
        else:
            attn_dtype = next(self.pipeline.model.parameters()).dtype
        
        # Apply RoPE to key states if model has rotary embeddings
        memory_length = x_triples.shape[1]
        if hasattr(self.pipeline.model.model, "rotary_emb"):
            pos_ids = torch.arange(memory_length, dtype=torch.long, device=self.pipeline.device).unsqueeze(0)
            dummy_x = torch.zeros(1, 1, memory_length, self.pipeline.model.config.hidden_size // self.pipeline.model.config.num_attention_heads).to(self.pipeline.device)
            cos, sin = self.pipeline.model.model.rotary_emb(dummy_x, pos_ids)
            cos_u = cos.unsqueeze(1).to(dtype=attn_dtype) # (1, 1, memory_length, head_dim)
            sin_u = sin.unsqueeze(1).to(dtype=attn_dtype) # (1, 1, memory_length, head_dim)
        else:
            cos_u, sin_u = None, None
            
        for idx, (k, v) in enumerate(raw_kv_tuples):
            k_cast = k.to(dtype=attn_dtype)
            v_cast = v.to(dtype=attn_dtype)
            
            if cos_u is not None:
                k1 = k_cast[..., : k_cast.shape[-1] // 2]
                k2 = k_cast[..., k_cast.shape[-1] // 2 :]
                rot_k = torch.cat((-k2, k1), dim=-1)
                k_cast = (k_cast * cos_u) + (rot_k * sin_u)
                
            past_key_values.update(k_cast, v_cast, idx)
        return past_key_values, raw_kv_tuples

    def _extract_teacher_kv(self, triple_text: str) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Runs the triple's natural language text through the frozen LLM to extract
        the 'teacher' KV states — and unrotates them if the model uses RoPE.
        """
        teacher_ids = self.pipeline.tokenizer(triple_text, return_tensors="pt").input_ids.to(self.pipeline.device)
        with torch.no_grad():
            teacher_out = self.pipeline.model(teacher_ids, use_cache=True)
        teacher_kv = teacher_out.past_key_values
        
        # Extract as list of (K, V) tuples
        result = []
        if isinstance(teacher_kv, DynamicCache):
            if hasattr(teacher_kv, "key_cache") and teacher_kv.key_cache is not None:
                for idx in range(len(teacher_kv.key_cache)):
                    result.append((teacher_kv.key_cache[idx], teacher_kv.value_cache[idx]))
            else:
                for idx in range(len(teacher_kv.layers)):
                    result.append((teacher_kv.layers[idx].keys, teacher_kv.layers[idx].values))
        else:
            # Tuple of tuples format
            for kv_pair in teacher_kv:
                result.append((kv_pair[0], kv_pair[1]))
                
        # Unrotate keys if model uses RoPE
        if hasattr(self.pipeline.model.model, "rotary_emb"):
            T = teacher_ids.shape[1]
            pos_ids = torch.arange(T, dtype=torch.long, device=self.pipeline.device).unsqueeze(0)
            dummy_x = torch.zeros(1, 1, T, self.pipeline.model.config.hidden_size // self.pipeline.model.config.num_attention_heads).to(self.pipeline.device)
            cos, sin = self.pipeline.model.model.rotary_emb(dummy_x, pos_ids)
            cos_u = cos.unsqueeze(1)
            sin_u = sin.unsqueeze(1)
            
            unrotated_result = []
            for idx, (k, v) in enumerate(result):
                k1 = k[..., : k.shape[-1] // 2]
                k2 = k[..., k.shape[-1] // 2 :]
                rot_k = torch.cat((-k2, k1), dim=-1)
                k_unrot = (k * cos_u) - (rot_k * sin_u)
                unrotated_result.append((k_unrot, v))
            return unrotated_result
            
        return result


    def _compute_distillation_loss(
        self, 
        men_kv: List[Tuple[torch.Tensor, torch.Tensor]], 
        teacher_kv: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> torch.Tensor:
        """
        MSE loss between MEN-projected KV states and the teacher LLM's own KV states.
        We compare the last token of the sequence (T-1) to map the summary representation.
        
        Uses variance normalization to scale losses equally across all layers and K/V.
        """
        mse_losses = []
        num_layers = min(len(men_kv), len(teacher_kv))
        
        for layer_idx in range(num_layers):
            men_k, men_v = men_kv[layer_idx]
            teach_k, teach_v = teacher_kv[layer_idx]
            
            # Cast to float32 for stable loss computation
            men_k_f = men_k.float()
            men_v_f = men_v.float()
            teach_k_f = teach_k.float()
            teach_v_f = teach_v.float()
            
            # Use the last token of the sequence: (batch, heads, seq, dim) -> (batch, heads, dim)
            men_k_last = men_k_f[:, :, -1, :]
            men_v_last = men_v_f[:, :, -1, :]
            teach_k_last = teach_k_f[:, :, -1, :]
            teach_v_last = teach_v_f[:, :, -1, :]
            
            # Compute variances of the teacher states to normalize scale differences across layers
            var_k = teach_k_f.var().clamp(min=1e-6)
            var_v = teach_v_f.var().clamp(min=1e-6)
            
            mse_k = F.mse_loss(men_k_last, teach_k_last) / var_k
            mse_v = F.mse_loss(men_v_last, teach_v_last) / var_v
            
            mse_losses.append(mse_k + mse_v)
        
        return torch.stack(mse_losses).mean()

    def train_step(self, x_triples: torch.Tensor, prompt_ids: torch.Tensor, 
                   target_ids: torch.Tensor, optimizer: optim.Optimizer,
                   triple_text: str = None, triple_idx: Optional[int] = None) -> Dict[str, float]:
        """
        Runs a single backpropagation training step on the MEN adapters.
        
        If triple_text is provided and distill_lambda > 0, computes the combined
        generation + distillation loss.
        """
        # Enforce memory safety
        self.mem_guard.enforce_safety()

        with GPULockManager():
            optimizer.zero_grad()

            # 1. MEN forward pass
            past_key_values, raw_kv_tuples = self._build_men_cache(x_triples)

            # 2. Prepare inputs for LLM forward pass
            inputs = torch.cat([prompt_ids, target_ids], dim=-1).to(self.pipeline.device)
            
            memory_length = x_triples.shape[1]
            memory_mask = torch.ones(inputs.shape[0], memory_length, dtype=torch.long, device=self.pipeline.device)
            inputs_mask = torch.ones_like(inputs)
            attention_mask = torch.cat([memory_mask, inputs_mask], dim=-1)
            
            position_ids = torch.arange(
                memory_length, 
                memory_length + inputs.shape[-1], 
                dtype=torch.long, 
                device=self.pipeline.device
            ).unsqueeze(0).expand(inputs.shape[0], -1)
            
            # 3. Forward pass through frozen LLM
            outputs = self.pipeline.model(
                inputs, 
                past_key_values=past_key_values, 
                attention_mask=attention_mask,
                position_ids=position_ids
            )
            
            # 4. Compute generation loss (cross-entropy)
            logits = outputs.logits
            prompt_len = prompt_ids.shape[-1]
            target_len = target_ids.shape[-1]
            
            shift_logits = logits[..., prompt_len - 1 : prompt_len + target_len - 1, :].contiguous()
            shift_labels = target_ids.contiguous()
            
            loss_gen = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_labels.view(-1)
            )
            
            # 5. Compute distillation loss if enabled
            loss_distill = torch.tensor(0.0, device=self.pipeline.device)
            if triple_text is not None and self.distill_lambda > 0:
                teacher_kv = self._extract_teacher_kv(triple_text)
                if triple_idx is not None:
                    # Slice raw_kv_tuples to get only the active triple's KVs
                    sliced_men_kv = []
                    for k, v in raw_kv_tuples:
                        sliced_men_kv.append((k[:, :, triple_idx:triple_idx+1, :], v[:, :, triple_idx:triple_idx+1, :]))
                    loss_distill = self._compute_distillation_loss(sliced_men_kv, teacher_kv)
                else:
                    loss_distill = self._compute_distillation_loss(raw_kv_tuples, teacher_kv)
            
            # Compute routing regularization loss if routing is active
            loss_routing_prior = torch.tensor(0.0, device=self.pipeline.device)
            if self.pipeline.men.use_routing:
                gates = self.pipeline.men.routing_gate(x_triples)
                gates = gates.view(-1, self.pipeline.men.num_layers, self.pipeline.men.num_heads)
                gates_mean_per_layer = gates.mean(dim=[0, 2])
                
                mu = (self.pipeline.men.num_layers - 1) / 2.0
                sigma = 4.0
                prior_target = []
                for l in range(self.pipeline.men.num_layers):
                    prior_target.append(np.exp(-((l - mu) ** 2) / (2.0 * (sigma ** 2))))
                prior_target = torch.tensor(prior_target, dtype=torch.float32, device=self.pipeline.device)
                loss_routing_prior = F.mse_loss(gates_mean_per_layer, prior_target)

            # 6. Combined loss
            loss_total = loss_gen + self.distill_lambda * loss_distill + 0.1 * loss_routing_prior

            # 7. Backpropagation
            loss_total.backward()
            # Gradient clipping to prevent exploding gradients from distillation
            torch.nn.utils.clip_grad_norm_(self.pipeline.men.parameters(), max_norm=1.0)
            optimizer.step()

        return {
            "loss_total": loss_total.item(),
            "loss_gen": loss_gen.item(),
            "loss_distill": loss_distill.item(),
            "loss_routing_prior": loss_routing_prior.item(),
        }

    def evaluate_recall(self, eval_samples: List[TripleTrainingSample], x_all_tensor: torch.Tensor, max_new_tokens: int = 30) -> Dict[str, Any]:
        """
        Runs the eval split through the full pipeline and measures factual recall.
        Does NOT train — pure inference evaluation.
        
        Returns:
            Dict with 'recall_rate', 'total', 'recalled', 'details'.
        """
        self.pipeline.men.eval()
        recalled = 0
        details = []
        
        for sample in eval_samples:
            # Extract the expected keyword from the target
            obj = sample.triple[2].lower()
            
            # Run MEN + generation using the full x_all_tensor
            with torch.no_grad():
                raw_kv = self.pipeline.men(x_all_tensor)
                past_kv = DynamicCache()
                
                # Dynamically find the target attention dtype
                if hasattr(self.pipeline.model, "model") and hasattr(self.pipeline.model.model, "layers") and len(self.pipeline.model.model.layers) > 0:
                    attn_dtype = self.pipeline.model.model.layers[0].self_attn.q_proj.weight.dtype
                else:
                    attn_dtype = next(self.pipeline.model.parameters()).dtype
                
                memory_length = x_all_tensor.shape[1]
                # Apply RoPE to key states if model has rotary embeddings
                if hasattr(self.pipeline.model.model, "rotary_emb"):
                    pos_ids = torch.arange(memory_length, dtype=torch.long, device=self.pipeline.device).unsqueeze(0)
                    dummy_x = torch.zeros(1, 1, memory_length, self.pipeline.model.config.hidden_size // self.pipeline.model.config.num_attention_heads).to(self.pipeline.device)
                    cos, sin = self.pipeline.model.model.rotary_emb(dummy_x, pos_ids)
                    cos_u = cos.unsqueeze(1).to(dtype=attn_dtype) # (1, 1, memory_length, head_dim)
                    sin_u = sin.unsqueeze(1).to(dtype=attn_dtype) # (1, 1, memory_length, head_dim)
                else:
                    cos_u, sin_u = None, None
                    
                for idx, (k, v) in enumerate(raw_kv):
                    k_cast = k.to(dtype=attn_dtype)
                    v_cast = v.to(dtype=attn_dtype)
                    
                    if cos_u is not None:
                        k1 = k_cast[..., : k_cast.shape[-1] // 2]
                        k2 = k_cast[..., k_cast.shape[-1] // 2 :]
                        rot_k = torch.cat((-k2, k1), dim=-1)
                        k_cast = (k_cast * cos_u) + (rot_k * sin_u)
                        
                    past_kv.update(k_cast, v_cast, idx)
                
                prompt_ids = self.pipeline.tokenizer(
                    wrap_training_prompt(sample.prompt_text), return_tensors="pt"
                ).input_ids.to(self.pipeline.device)
                
                memory_length = x_all_tensor.shape[1]
                memory_mask = torch.ones(1, memory_length, dtype=torch.long, device=self.pipeline.device)
                prompt_mask = torch.ones_like(prompt_ids)
                full_mask = torch.cat([memory_mask, prompt_mask], dim=-1)
                
                position_ids = torch.arange(
                    memory_length, memory_length + prompt_ids.shape[1],
                    dtype=torch.long, device=self.pipeline.device
                ).unsqueeze(0)
                
                outputs = self.pipeline.model.generate(
                    prompt_ids,
                    past_key_values=past_kv,
                    attention_mask=full_mask,
                    position_ids=position_ids,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.pipeline.tokenizer.pad_token_id,
                    do_sample=False,  # Greedy for deterministic eval
                )
            
            response = self.pipeline.tokenizer.decode(outputs[0][prompt_ids.shape[1]:], skip_special_tokens=True).strip()
            hit = obj in response.lower()
            if hit:
                recalled += 1
            details.append({
                "triple": sample.triple,
                "expected": obj,
                "response": response,
                "recalled": hit,
            })
        
        self.pipeline.men.train()
        
        total = len(eval_samples)
        return {
            "recall_rate": (recalled / total * 100) if total > 0 else 0.0,
            "total": total,
            "recalled": recalled,
            "details": details,
        }

    def fit(
        self, 
        train_samples: List[TripleTrainingSample],
        eval_samples: List[TripleTrainingSample],
        epochs: int = 200,
        patience: int = 15,
        lr: float = None,
        retrieval_aware: bool = True,
        max_cache_triples: int = 20,
        num_distractors: int = 15,
    ) -> Dict[str, Any]:
        """
        Full training loop with KV-distillation, multi-triple data, and eval-recall early stopping.
        Accumulates gradients over all samples using leaf key-value tensors for optimal memory usage
        on limited VRAM systems.
        
        Args:
            train_samples: Training set of TripleTrainingSample.
            eval_samples: Eval set of TripleTrainingSample (disjoint from train).
            epochs: Maximum number of epochs.
            patience: Early stop after this many epochs without eval recall improvement.
            lr: Learning rate override.
            
        Returns:
            Dict with training history.
        """
        if lr is None:
            lr = self.lr
            
        print("\n=== STARTING MEN TRAINING WITH KV-DISTILLATION ===")
        print(f"  Train samples : {len(train_samples)}")
        print(f"  Eval samples  : {len(eval_samples)}")
        print(f"  Max epochs    : {epochs}")
        print(f"  Patience      : {patience}")
        print(f"  Distill λ     : {self.distill_lambda}")
        print(f"  Learning rate : {lr}")
        print(f"  Retrieval-aware: {retrieval_aware} (max_cache={max_cache_triples})")
        
        # Freeze LLM, unfreeze MEN
        for param in self.pipeline.model.parameters():
            param.requires_grad = False
        for param in self.pipeline.men.parameters():
            param.requires_grad = True
        self.pipeline.men.train()
        
        optimizer = optim.AdamW(self.pipeline.men.parameters(), lr=lr, weight_decay=0.01)
        
        # Build the combined x_all_tensor containing all triples (train + eval)
        # to match the inference sequence length distribution. Ensure only unique triples are used.
        unique_triples = []
        unique_x_vectors = []
        seen_triples = set()
        for sample in train_samples + eval_samples:
            triple_key = f"{sample.triple[0]}|{sample.triple[1]}|{sample.triple[2]}"
            if triple_key not in seen_triples:
                seen_triples.add(triple_key)
                unique_triples.append(sample.triple)
                unique_x_vectors.append(sample.x_vector)
                
        x_all_arr = np.stack(unique_x_vectors)
        x_all_tensor = torch.tensor(x_all_arr, dtype=torch.float32).unsqueeze(0).to(self.pipeline.device)
        
        # Pre-tokenize all samples and pre-extract teacher KVs to speed up training
        tokenized_samples = []
        print("  Pre-tokenizing samples and pre-extracting teacher KVs...")
        for sample in train_samples:
            # Find the index of this sample in the unique triples list
            triple_idx = -1
            for idx, trip in enumerate(unique_triples):
                if trip == sample.triple:
                    triple_idx = idx
                    break
            
            prompt_ids = self.pipeline.tokenizer(
                wrap_training_prompt(sample.prompt_text), return_tensors="pt"
            ).input_ids.to(self.pipeline.device)
            target_ids = self.pipeline.tokenizer(sample.target_text, return_tensors="pt").input_ids.to(self.pipeline.device)
            
            teacher_kv = None
            if self.distill_lambda > 0 and sample.triple_text is not None:
                teacher_kv = self._extract_teacher_kv(sample.triple_text)
                
            tokenized_samples.append((prompt_ids, target_ids, sample.triple_text, triple_idx, teacher_kv, sample.triple))
        
        # Dynamic Magnitude Calibration
        print("  Running dynamic magnitude calibration...")
        with torch.no_grad():
            # Extract teacher KVs for all unique training triples
            all_teacher_kvs = []
            seen_texts = set()
            for sample in train_samples:
                if sample.triple_text not in seen_texts:
                    seen_texts.add(sample.triple_text)
                    teacher_kv = self._extract_teacher_kv(sample.triple_text)
                    all_teacher_kvs.append(teacher_kv)
            
            # For each layer, compute mean and std across all samples
            num_layers = len(all_teacher_kvs[0])
            for layer_idx in range(num_layers):
                layer_ks = []
                layer_vs = []
                for tkvs in all_teacher_kvs:
                    k, v = tkvs[layer_idx]
                    layer_ks.append(k.cpu())
                    layer_vs.append(v.cpu())
                
                # Concatenate along the sequence dimension (dim=2) to get global stats
                all_k = torch.cat(layer_ks, dim=2)
                all_v = torch.cat(layer_vs, dim=2)
                
                mean_k = all_k.mean().item()
                std_k = all_k.std().item()
                mean_v = all_v.mean().item()
                std_v = all_v.std().item()
                
                # Initialize LayerNorm parameters in MEN to match teacher distribution
                self.pipeline.men.k_norms[layer_idx].weight.data.fill_(std_k)
                self.pipeline.men.k_norms[layer_idx].bias.data.fill_(mean_k)
                self.pipeline.men.v_norms[layer_idx].weight.data.fill_(std_v)
                self.pipeline.men.v_norms[layer_idx].bias.data.fill_(mean_v)
                
                if layer_idx in [0, num_layers // 2, num_layers - 1]:
                    print(f"    Layer {layer_idx:2d} | Key std={std_k:.4f}, mean={mean_k:.4f} | Val std={std_v:.4f}, mean={mean_v:.4f}")
        
        trainable_params = sum(p.numel() for p in self.pipeline.men.parameters() if p.requires_grad)
        print(f"  Trainable params: {trainable_params:,}")
        print("  Starting optimization loop...\n")
        
        history = {"train_loss": [], "train_loss_gen": [], "train_loss_distill": [], "eval_recall": []}
        best_recall = -1.0
        best_loss = 999.0
        best_epoch = 0
        epochs_without_improvement = 0
        best_state_dict = None
        
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            self.pipeline.men.train()
            optimizer.zero_grad()
            
            # 1. Forward MEN once per epoch to get KVs for all triples
            raw_kv_tuples = self.pipeline.men(x_all_tensor)
            
            # 2. Clone each KV tensor and set requires_grad=True to create leaf nodes
            accum_k_list = []
            accum_v_list = []
            for k, v in raw_kv_tuples:
                k_leaf = k.detach().clone().requires_grad_(True)
                v_leaf = v.detach().clone().requires_grad_(True)
                accum_k_list.append(k_leaf)
                accum_v_list.append(v_leaf)
                
            # 3. Find the target attention dtype and extract rotary embeddings
            # Find the target attention dtype
            if hasattr(self.pipeline.model, "model") and hasattr(self.pipeline.model.model, "layers") and len(self.pipeline.model.model.layers) > 0:
                attn_dtype = self.pipeline.model.model.layers[0].self_attn.q_proj.weight.dtype
            else:
                attn_dtype = next(self.pipeline.model.parameters()).dtype
                
            memory_length = x_all_tensor.shape[1]
            has_rotary = hasattr(self.pipeline.model.model, "rotary_emb")
                
            # 4. Accumulate gradients over all samples
            epoch_losses = {"total": [], "gen": [], "distill": []}
            
            # Shuffle training samples for stochastic flavor
            order = list(range(len(tokenized_samples)))
            np.random.seed(epoch)
            np.random.shuffle(order)
            
            # Enforce memory safety
            self.mem_guard.enforce_safety()
            
            with GPULockManager():
                for sample_idx in order:
                    prompt_ids, target_ids, triple_text, triple_idx, teacher_kv, target_triple = tokenized_samples[sample_idx]

                    # Build per-sample cache: retrieval-aware subset with target triple at end
                    if retrieval_aware and triple_idx >= 0 and len(unique_triples) > 1:
                        other_indices = [i for i in range(len(unique_triples)) if i != triple_idx]
                        np.random.shuffle(other_indices)
                        distractor_indices = other_indices[:num_distractors]
                        cache_indices = distractor_indices + [triple_idx]
                        if len(cache_indices) > max_cache_triples:
                            cache_indices = cache_indices[-max_cache_triples:]
                        sample_memory_length = len(cache_indices)
                    else:
                        cache_indices = list(range(memory_length))
                        sample_memory_length = memory_length

                    sample_cache = DynamicCache()
                    for idx in range(len(accum_k_list)):
                        k_subset = accum_k_list[idx][:, :, cache_indices, :]
                        v_subset = accum_v_list[idx][:, :, cache_indices, :]
                        k_cast = k_subset.to(dtype=attn_dtype)
                        v_cast = v_subset.to(dtype=attn_dtype)
                        if has_rotary:
                            pos_ids = torch.arange(sample_memory_length, dtype=torch.long, device=self.pipeline.device).unsqueeze(0)
                            dummy_x = torch.zeros(
                                1,
                                1,
                                sample_memory_length,
                                self.pipeline.model.config.hidden_size // self.pipeline.model.config.num_attention_heads,
                            ).to(self.pipeline.device)
                            cos, sin = self.pipeline.model.model.rotary_emb(dummy_x, pos_ids)
                            sub_cos = cos.unsqueeze(1).to(dtype=attn_dtype)
                            sub_sin = sin.unsqueeze(1).to(dtype=attn_dtype)
                            k1 = k_cast[..., : k_cast.shape[-1] // 2]
                            k2 = k_cast[..., k_cast.shape[-1] // 2 :]
                            rot_k = torch.cat((-k2, k1), dim=-1)
                            k_cast = (k_cast * sub_cos) + (rot_k * sub_sin)
                        sample_cache.update(k_cast, v_cast, idx)

                    # Prepare inputs
                    inputs = torch.cat([prompt_ids, target_ids], dim=-1).to(self.pipeline.device)
                    memory_mask = torch.ones(inputs.shape[0], sample_memory_length, dtype=torch.long, device=self.pipeline.device)
                    inputs_mask = torch.ones_like(inputs)
                    attention_mask = torch.cat([memory_mask, inputs_mask], dim=-1)
                    
                    position_ids = torch.arange(
                        sample_memory_length, 
                        sample_memory_length + inputs.shape[-1], 
                        dtype=torch.long, 
                        device=self.pipeline.device
                    ).unsqueeze(0).expand(inputs.shape[0], -1)
                    
                    # Forward frozen LLM
                    outputs = self.pipeline.model(
                        inputs,
                        past_key_values=sample_cache,
                        attention_mask=attention_mask,
                        position_ids=position_ids
                    )
                    
                    # Compute loss
                    logits = outputs.logits
                    prompt_len = prompt_ids.shape[-1]
                    target_len = target_ids.shape[-1]
                    shift_logits = logits[..., prompt_len - 1 : prompt_len + target_len - 1, :].contiguous()
                    shift_labels = target_ids.contiguous()
                    
                    loss_gen = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)
                    )
                    
                    loss_distill = torch.tensor(0.0, device=self.pipeline.device)
                    if self.distill_lambda > 0 and teacher_kv is not None and triple_idx >= 0:
                        # Distill the target triple at its position in the sample cache
                        target_pos = cache_indices.index(triple_idx) if triple_idx in cache_indices else -1
                        if target_pos >= 0:
                            sliced_men_kv = []
                            for k_leaf, v_leaf in zip(accum_k_list, accum_v_list):
                                sliced_men_kv.append((
                                    k_leaf[:, :, triple_idx:triple_idx+1, :],
                                    v_leaf[:, :, triple_idx:triple_idx+1, :]
                                ))
                            loss_distill = self._compute_distillation_loss(sliced_men_kv, teacher_kv)
                        
                    loss_total = loss_gen + self.distill_lambda * loss_distill
                    loss_sample = loss_total / len(tokenized_samples)
                    
                    # Backward pass for the sample to accumulate gradients on accum_k_list and accum_v_list
                    loss_sample.backward()
                    
                    epoch_losses["total"].append(loss_total.item())
                    epoch_losses["gen"].append(loss_gen.item())
                    epoch_losses["distill"].append(loss_distill.item())
                    
                # Compute routing regularization loss for all triples
                loss_routing_prior = torch.tensor(0.0, device=self.pipeline.device)
                if self.pipeline.men.use_routing:
                    gates = self.pipeline.men.routing_gate(x_all_tensor)
                    gates = gates.view(-1, self.pipeline.men.num_layers, self.pipeline.men.num_heads)
                    gates_mean_per_layer = gates.mean(dim=[0, 2])
                    
                    mu = (self.pipeline.men.num_layers - 1) / 2.0
                    sigma = 4.0
                    prior_target = []
                    for l in range(self.pipeline.men.num_layers):
                        prior_target.append(np.exp(-((l - mu) ** 2) / (2.0 * (sigma ** 2))))
                    prior_target = torch.tensor(prior_target, dtype=torch.float32, device=self.pipeline.device)
                    loss_routing_prior = F.mse_loss(gates_mean_per_layer, prior_target)
                    
                # 5. Backward pass for MEN using the accumulated gradients
                loss_men = 0.0
                for idx, (k, v) in enumerate(raw_kv_tuples):
                    if accum_k_list[idx].grad is not None:
                        loss_men = loss_men + torch.sum(k * accum_k_list[idx].grad)
                    if accum_v_list[idx].grad is not None:
                        loss_men = loss_men + torch.sum(v * accum_v_list[idx].grad)
                        
                if isinstance(loss_men, torch.Tensor):
                    # Add routing regularization loss
                    loss_men = loss_men + 0.1 * loss_routing_prior
                    loss_men.backward()
                    
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.pipeline.men.parameters(), max_norm=1.0)
                optimizer.step()
                
            avg_total = np.mean(epoch_losses["total"])
            avg_gen = np.mean(epoch_losses["gen"])
            avg_distill = np.mean(epoch_losses["distill"])
            
            history["train_loss"].append(avg_total)
            history["train_loss_gen"].append(avg_gen)
            history["train_loss_distill"].append(avg_distill)
            
            # Eval every 10 epochs or on first/last
            run_eval = (epoch % 10 == 0 or epoch == 1 or epoch == epochs)
            eval_recall_pct = -1.0
            
            if run_eval and eval_samples:
                eval_result = self.evaluate_recall(eval_samples, x_all_tensor)
                eval_recall_pct = eval_result["recall_rate"]
                history["eval_recall"].append(eval_recall_pct)
                
                # Check if this epoch is better (higher recall, or same recall with lower loss)
                is_better = False
                if eval_recall_pct > best_recall:
                    is_better = True
                elif eval_recall_pct == best_recall and avg_total < best_loss:
                    is_better = True
                    
                if is_better:
                    best_recall = eval_recall_pct
                    best_loss = avg_total
                    best_epoch = epoch
                    epochs_without_improvement = 0
                    import copy
                    best_state_dict = copy.deepcopy(self.pipeline.men.state_dict())
                else:
                    epochs_without_improvement += 10  # We eval every 10 epochs
            
            duration = time.time() - start_time
            
            # Print progress
            if epoch % 10 == 0 or epoch == 1 or epoch <= 5 or run_eval:
                eval_str = f" | Eval Recall: {eval_recall_pct:.1f}%" if eval_recall_pct >= 0 else ""
                print(f"  Epoch {epoch:3d}/{epochs} | Loss: {avg_total:.4f} (gen={avg_gen:.4f}, dist={avg_distill:.4f}){eval_str} | {duration:.2f}s")
            
            # Thermal safety
            self.thermal_guard.cycle()
            
            # Early stopping on eval recall
            if epochs_without_improvement >= patience and best_recall >= 0:
                print(f"\n  [Early Stop] No recall improvement for {patience} epochs. Best: {best_recall:.1f}% at epoch {best_epoch}.")
                break
            
            # Perfect recall — stop early only if distillation loss is also minimized (or not used)
            if best_recall >= 100.0 and (self.distill_lambda == 0.0 or avg_distill < 0.15):
                print(f"\n  [Perfect Recall] 100% eval recall and minimized distillation loss at epoch {epoch}. Stopping.")
                break
        
        # Restore the best weights
        if best_state_dict is not None:
            self.pipeline.men.load_state_dict(best_state_dict)
            self._save_men_checkpoint()
            
        self.pipeline.men.eval()
        print(f"\n=== TRAINING COMPLETE === Best eval recall: {best_recall:.1f}% (epoch {best_epoch})")
        
        return history

    def fit_prototype(self, epochs: int = 10):
        """
        Executes a prototype training run using synthetic triples to prove the MEN adapter optimization works.
        """
        print("\n=== STARTING MEN ADAPTER TRAINING PROTOTYPE ===")
        self.pipeline.initialize()
        
        # Ensure LLM parameters are frozen
        for param in self.pipeline.model.parameters():
            param.requires_grad = False
            
        # Ensure MEN parameters are trainable
        for param in self.pipeline.men.parameters():
            param.requires_grad = True

        optimizer = optim.AdamW(self.pipeline.men.parameters(), lr=self.lr)
        
        # Define synthetic training sample
        dummy_triples = [
            ["Project", "uses", "FastAPI"],
            ["Database", "connects_to", "PostgreSQL"],
            ["Database", "runs_on", "Port 8080"]
        ]
        
        # Encode triples into MEN input shape (1, 3, 768)
        texts_to_embed = []
        for trip in dummy_triples:
            texts_to_embed.append(f"{trip[0]} {trip[1]}")
            texts_to_embed.append(trip[2])
            
        embeddings = self.pipeline.retriever.embed_texts(texts_to_embed) # shape: (6, 384)
        
        x_list = []
        for idx in range(len(dummy_triples)):
            sp_vector = embeddings[2 * idx]
            o_vector = embeddings[2 * idx + 1]
            x_i = np.concatenate([sp_vector, o_vector])
            x_list.append(x_i)
            
        x_arr = np.stack(x_list)
        x_tensor = torch.tensor(x_arr, dtype=torch.float32).unsqueeze(0).to(self.pipeline.device)
        
        # Dummy Prompt and Target responses
        prompt_text = "User: Generate code for postgres db config\nAssistant:"
        target_text = "Here is your FastAPI PostgreSQL configuration running on port 8080 using asyncpg."
        
        prompt_ids = self.pipeline.tokenizer(prompt_text, return_tensors="pt").input_ids.to(self.pipeline.device)
        target_ids = self.pipeline.tokenizer(target_text, return_tensors="pt").input_ids.to(self.pipeline.device)
        
        print(f"Training parameters:")
        print(f"  Device           : {self.pipeline.device}")
        print(f"  Input Shape (x)  : {x_tensor.shape} (1 batch, 3 triples, 768 dimensions)")
        print(f"  Prompt tokens    : {prompt_ids.shape[-1]}")
        print(f"  Target tokens    : {target_ids.shape[-1]}")
        print(f"  Trainable Params : {sum(p.numel() for p in self.pipeline.men.parameters() if p.requires_grad)} parameters")
        print("Starting training optimization loop...")

        for epoch in range(1, epochs + 1):
            start_time = time.time()
            
            # Execute step (legacy mode — generation loss only, no distillation)
            losses = self.train_step(x_tensor, prompt_ids, target_ids, optimizer, triple_text=None)
            
            duration = time.time() - start_time
            print(f"  Epoch {epoch:2d}/{epochs:2d} | Loss: {losses['loss_total']:.6f} | Time: {duration:.3f}s")
            
            # Enforce thermal safety rules to avoid overheating GPU RTX A2000
            self.thermal_guard.cycle()
            
        print("=== PROTOTYPE TRAINING COMPLETED SUCCESSFULLY ===\n")
