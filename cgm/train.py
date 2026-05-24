import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, List
from transformers.cache_utils import DynamicCache
from cgm.schema import SQLiteGraphStore
from cgm.retriever import SubgraphRetriever
from cgm.model import MemoryEncoderNetwork
from cgm.pipeline import CGMPipeline
from cgm.safety import GPULockManager, GPUMemoryGuard, ThermalGuard

class MEGATrainer:
    """
    Trains the Memory Encoder Network (MEN) adapters to map
    retrieved graph triples to the attention space of the target LLM.
    Keeps the target LLM frozen.
    """
    def __init__(self, pipeline: CGMPipeline, lr: float = 1e-4):
        self.pipeline = pipeline
        self.lr = lr
        self.thermal_guard = ThermalGuard(batch_limit_before_cooldown=5, cooldown_seconds=2.0)
        self.mem_guard = GPUMemoryGuard()

    def train_step(self, x_triples: torch.Tensor, prompt_ids: torch.Tensor, 
                   target_ids: torch.Tensor, optimizer: optim.Optimizer) -> float:
        """
        Runs a single backpropagation training step on the MEN adapters.
        Ensures thread locks and memory limits are respected.
        """
        # Enforce memory safety
        self.mem_guard.enforce_safety()

        with GPULockManager():
            # Zero gradients
            optimizer.zero_grad()

            raw_kv_tuples = self.pipeline.men(x_triples)
            # Convert raw tuples to DynamicCache (compatible with all transformers versions)
            past_key_values = DynamicCache()
            for idx, (k, v) in enumerate(raw_kv_tuples):
                past_key_values.update(k, v, idx)

            # 2. Prepare inputs for LLM forward pass
            # We concatenate the prompt_ids and target_ids
            # inputs shape: (batch_size, prompt_len + target_len)
            inputs = torch.cat([prompt_ids, target_ids], dim=-1).to(self.pipeline.device)
            
            # Build attention mask that covers both injected memory positions and input tokens
            memory_length = x_triples.shape[1]
            memory_mask = torch.ones(inputs.shape[0], memory_length, dtype=torch.long, device=self.pipeline.device)
            inputs_mask = torch.ones_like(inputs)
            attention_mask = torch.cat([memory_mask, inputs_mask], dim=-1)
            
            # Start absolute positions after the memory keys/values to avoid position overlap
            position_ids = torch.arange(
                memory_length, 
                memory_length + inputs.shape[-1], 
                dtype=torch.long, 
                device=self.pipeline.device
            ).unsqueeze(0).expand(inputs.shape[0], -1)
            
            # Forward pass through frozen LLM
            # We pass the synthetic KV cache, combined attention mask, and shifted position IDs
            outputs = self.pipeline.model(
                inputs, 
                past_key_values=past_key_values, 
                attention_mask=attention_mask,
                position_ids=position_ids
            )
            
            # 3. Compute loss
            # We calculate cross-entropy loss strictly on the target_ids tokens
            logits = outputs.logits
            
            # Align logits and targets for loss calculation
            # Logits shape: (batch, seq_len, vocab_size)
            prompt_len = prompt_ids.shape[-1]
            target_len = target_ids.shape[-1]
            
            # We slice logits corresponding to the target tokens
            # We want to predict target_ids[t] given input prefix + target_ids[:t]
            # So we take logits from index (prompt_len - 1) to (prompt_len + target_len - 2)
            shift_logits = logits[..., prompt_len - 1 : prompt_len + target_len - 1, :].contiguous()
            shift_labels = target_ids.contiguous()
            
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            # 4. Backpropagation
            loss.backward()
            optimizer.step()

        return loss.item()

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
        # 3 triples representing database configurations
        dummy_triples = [
            ["Project", "uses", "FastAPI"],
            ["Database", "connects_to", "PostgreSQL"],
            ["Database", "runs_on", "Port 8080"]
        ]
        
        # Encode triples into MEN input shape (1, 3, 768)
        x_list = []
        for trip in dummy_triples:
            sp_vector = self.pipeline.retriever.embed_text(f"{trip[0]} {trip[1]}")
            o_vector = self.pipeline.retriever.embed_text(trip[2])
            x_i = np.concatenate([sp_vector, o_vector])
            x_list.append(x_i)
            
        x_arr = np.stack(x_list)
        x_tensor = torch.tensor(x_arr, dtype=torch.float32).unsqueeze(0).to(self.pipeline.device)
        
        # Dummy Prompt and Target responses
        # Turn 6 Query: "Generate code for postgres db config"
        # Turn 6 Target: "Here is your FastAPI PostgreSQL configuration running on port 8080 using asyncpg."
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
            
            # Execute step
            loss = self.train_step(x_tensor, prompt_ids, target_ids, optimizer)
            
            duration = time.time() - start_time
            print(f"  Epoch {epoch:2d}/{epochs:2d} | Loss: {loss:.6f} | Time: {duration:.3f}s")
            
            # Enforce thermal safety rules to avoid overheating GPU RTX A2000
            self.thermal_guard.cycle()
            
        print("=== PROTOTYPE TRAINING COMPLETED SUCCESSFULLY ===\n")
