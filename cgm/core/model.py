import torch
import torch.nn as nn
from typing import Tuple, List, Dict, Any
import math

class LowRankLinear(nn.Module):
    """
    Low-Rank Linear projection layer (LoRA-style).
    Decomposes a large projection matrix into a low-rank bottleneck with non-linearity.
    """
    def __init__(self, in_features: int, out_features: int, rank: int = 64):
        super().__init__()
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.relu = nn.ReLU()
        self.lora_B = nn.Linear(rank, out_features, bias=True)
        
        # Standard initialization
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_B.weight, a=math.sqrt(5))
        if self.lora_B.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.lora_B.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.lora_B.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora_B(self.relu(self.lora_A(x)))

class MemoryEncoderNetwork(nn.Module):
    """
    Memory Encoder Network (MEN).
    Projects dense representations of semantic triples to the key and value space
    of a target Frozen LLM for rectangular attention injection.
    """
    def __init__(self, input_dim: int = 768, num_layers: int = 12, 
                 num_heads: int = 12, head_dim: int = 64, use_routing: bool = True, rank: int = 64):
        """
        Args:
            input_dim: Dimension of the triple representation [enc(subj || pred); enc(obj)] (e.g., 2 * 384 = 768)
            num_layers: Number of transformer layers in the target LLM (e.g., 12 for GPT-2)
            num_heads: Number of key-value heads in the target LLM (e.g., 12 for GPT-2)
            head_dim: Dimension per key-value head in the target LLM (e.g., 64 for GPT-2)
            use_routing: Whether to use learned semantics-aware KV head/layer routing
            rank: Rank of low-rank projections (LoRA style)
        """
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.output_dim = num_heads * head_dim
        self.use_routing = use_routing

        # Define key (K) and value (V) low-rank projection layers for each LLM layer
        self.k_projections = nn.ModuleList([
            LowRankLinear(input_dim, self.output_dim, rank=rank) for _ in range(num_layers)
        ])
        
        self.v_projections = nn.ModuleList([
            LowRankLinear(input_dim, self.output_dim, rank=rank) for _ in range(num_layers)
        ])

        # Per-layer LayerNorm for magnitude matching against the LLM's internal KV distribution.
        # Without this, raw linear projections produce activations on an arbitrary scale that
        # the frozen attention heads were never trained to attend over.
        self.k_norms = nn.ModuleList([
            nn.LayerNorm(self.output_dim) for _ in range(num_layers)
        ])
        self.v_norms = nn.ModuleList([
            nn.LayerNorm(self.output_dim) for _ in range(num_layers)
        ])

        if self.use_routing:
            # Semantics-Aware KV Cache Routing Network
            # Maps input embeddings (batch, M, input_dim) -> routing weights (batch, M, num_layers * num_heads)
            self.routing_gate = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, num_layers * num_heads),
                nn.Sigmoid()
            )
            
            # Apply Layer-Depth Bias Initialization (prior favoring middle layers)
            import math
            with torch.no_grad():
                # Center Gaussian at mid-layer
                mu = (num_layers - 1) / 2.0
                sigma = 4.0
                for l in range(num_layers):
                    # Calculate prior scaling factor
                    prior_val = math.exp(-((l - mu) ** 2) / (2 * (sigma ** 2))) # range [0, 1]
                    # Shift and scale prior_val to map to bias values (e.g. from -1.0 to 1.5)
                    # This biases the gates to start higher in the middle layers (~0.73 sigmoid)
                    # and lower in early/late layers (~0.27 sigmoid)
                    bias_val = -1.0 + 2.5 * prior_val
                    for h in range(num_heads):
                        idx = l * num_heads + h
                        self.routing_gate[2].bias[idx] = bias_val

    def forward(self, x: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: Input tensor representing M triples. Shape: (batch_size, M, input_dim)
            
        Returns:
            A list of length `num_layers` containing tuples of (key_states, value_states),
            where each state has shape: (batch_size, num_heads, M, head_dim).
        """
        batch_size, M, _ = x.shape
        past_key_values = []

        # Compute dynamic routing gates if enabled
        if self.use_routing:
            # Shape: (batch_size, M, num_layers * num_heads)
            gates = self.routing_gate(x)
            # Reshape to (batch_size, M, num_layers, num_heads, 1) for broadcasting
            gates = gates.view(batch_size, M, self.num_layers, self.num_heads, 1)
        else:
            gates = None

        for layer_idx in range(self.num_layers):
            # Project key states
            # (batch_size, M, input_dim) -> (batch_size, M, num_heads * head_dim)
            k_proj = self.k_projections[layer_idx](x)
            k_proj = self.k_norms[layer_idx](k_proj)  # Magnitude matching
            
            # Reshape to (batch_size, M, num_heads, head_dim)
            k_states = k_proj.view(batch_size, M, self.num_heads, self.head_dim)

            # Project value states
            # (batch_size, M, input_dim) -> (batch_size, M, num_heads * head_dim)
            v_proj = self.v_projections[layer_idx](x)
            v_proj = self.v_norms[layer_idx](v_proj)  # Magnitude matching
            
            # Reshape to (batch_size, M, num_heads, head_dim)
            v_states = v_proj.view(batch_size, M, self.num_heads, self.head_dim)

            # Apply semantics-aware routing weights if active
            if gates is not None:
                # Extract layer gate: shape (batch_size, M, num_heads, 1)
                layer_gate = gates[:, :, layer_idx, :, :]
                k_states = k_states * layer_gate
                v_states = v_states * layer_gate

            # Permute to (batch_size, num_heads, M, head_dim) for HF cache compatibility
            k_states = k_states.permute(0, 2, 1, 3)
            v_states = v_states.permute(0, 2, 1, 3)

            past_key_values.append((k_states, v_states))

        return past_key_values

    def get_gate_stats(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Introspects the routing gate values for a given input tensor.
        Used for ablation analysis to determine if routing is meaningful or a no-op.
        
        Args:
            x: Input tensor representing M triples. Shape: (batch_size, M, input_dim)
            
        Returns:
            Dict with per-layer/head gate statistics: mean, std, min, max, 
            fraction near 1.0 (>0.9), fraction near 0.0 (<0.1).
        """
        import numpy as np
        
        if not self.use_routing:
            return {"routing_enabled": False, "message": "Routing is disabled."}
        
        self.eval()
        with torch.no_grad():
            gates = self.routing_gate(x)  # (batch, M, num_layers * num_heads)
        
        # Reshape to (batch, M, num_layers, num_heads)
        batch_size, M, _ = x.shape
        gates_reshaped = gates.view(batch_size, M, self.num_layers, self.num_heads)
        
        # Global statistics
        gate_vals = gates_reshaped.cpu().numpy().flatten()
        global_stats = {
            "routing_enabled": True,
            "total_gates": len(gate_vals),
            "global_mean": float(np.mean(gate_vals)),
            "global_std": float(np.std(gate_vals)),
            "global_min": float(np.min(gate_vals)),
            "global_max": float(np.max(gate_vals)),
            "frac_near_one": float(np.mean(gate_vals > 0.9)),
            "frac_near_zero": float(np.mean(gate_vals < 0.1)),
            "frac_mid_range": float(np.mean((gate_vals >= 0.1) & (gate_vals <= 0.9))),
        }
        
        # Per-layer statistics
        per_layer = []
        for layer_idx in range(self.num_layers):
            layer_gates = gates_reshaped[:, :, layer_idx, :].cpu().numpy().flatten()
            per_layer.append({
                "layer": layer_idx,
                "mean": float(np.mean(layer_gates)),
                "std": float(np.std(layer_gates)),
                "min": float(np.min(layer_gates)),
                "max": float(np.max(layer_gates)),
                "frac_near_one": float(np.mean(layer_gates > 0.9)),
                "frac_near_zero": float(np.mean(layer_gates < 0.1)),
            })
        global_stats["per_layer"] = per_layer
        
        return global_stats

