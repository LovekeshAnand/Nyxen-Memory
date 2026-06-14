import torch
import torch.nn as nn
from typing import Tuple, List

class MemoryEncoderNetwork(nn.Module):
    """
    Memory Encoder Network (MEN).
    Projects dense representations of semantic triples to the key and value space
    of a target Frozen LLM for rectangular attention injection.
    """
    def __init__(self, input_dim: int = 768, num_layers: int = 12, 
                 num_heads: int = 12, head_dim: int = 64, use_routing: bool = True):
        """
        Args:
            input_dim: Dimension of the triple representation [enc(subj || pred); enc(obj)] (e.g., 2 * 384 = 768)
            num_layers: Number of transformer layers in the target LLM (e.g., 12 for GPT-2)
            num_heads: Number of key-value heads in the target LLM (e.g., 12 for GPT-2)
            head_dim: Dimension per key-value head in the target LLM (e.g., 64 for GPT-2)
            use_routing: Whether to use learned semantics-aware KV head/layer routing
        """
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.output_dim = num_heads * head_dim
        self.use_routing = use_routing

        # Define key (K) and value (V) projection layers for each LLM layer
        # Maps shape (batch, M, input_dim) -> (batch, M, num_heads * head_dim)
        self.k_projections = nn.ModuleList([
            nn.Linear(input_dim, self.output_dim) for _ in range(num_layers)
        ])
        
        self.v_projections = nn.ModuleList([
            nn.Linear(input_dim, self.output_dim) for _ in range(num_layers)
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
            
            # Reshape to (batch_size, M, num_heads, head_dim)
            k_states = k_proj.view(batch_size, M, self.num_heads, self.head_dim)

            # Project value states
            # (batch_size, M, input_dim) -> (batch_size, M, num_heads * head_dim)
            v_proj = self.v_projections[layer_idx](x)
            
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

