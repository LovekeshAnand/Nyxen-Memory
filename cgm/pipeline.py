import torch
import numpy as np
from typing import Dict, Any, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from cgm.schema import SQLiteGraphStore
from cgm.retriever import SubgraphRetriever
from cgm.model import MemoryEncoderNetwork
from cgm.safety import GPULockManager, GPUMemoryGuard, InputBufferGuard

class CGMPipeline:
    """
    End-to-End inference pipeline for Conversational Graph Memory.
    Loads the target LLM and MEN under strict safety locks, retrieves relevant subgraph memory,
    projects it to KV space, and runs inference.
    """
    def __init__(self, model_name: str = "gpt2", db_path: str = "data/cgm_memory.db", device: str = None):
        self.db_path = db_path
        self.model_name = model_name
        
        # Initialize GPU guards
        self.mem_guard = GPUMemoryGuard()
        self.device = device
        
        self._tokenizer = None
        self._model = None
        self._men = None
        self._store = None
        self._retriever = None

    def initialize(self):
        """
        Loads the LLM, tokenizer, database, and retriever.
        Ensures thread safety and checks available VRAM to prevent OOM.
        """
        # Proactively check available VRAM before model load
        self.mem_guard.enforce_safety(model_size_est_mb=500.0)
        
        # Detect device
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Thread-safe load
        with GPULockManager():
            print(f"[Pipeline] Loading tokenizer for '{self.model_name}'...")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Set pad token if not defined (common for GPT-2)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
                
            print(f"[Pipeline] Loading LLM '{self.model_name}' on device: {self.device}...")
            self._model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self._model.to(self.device)
            self._model.eval() # Set to evaluation mode
            
            # Get LLM architecture parameters for MEN mapping
            config = self._model.config
            num_layers = getattr(config, "n_layer", getattr(config, "num_hidden_layers", 12))
            num_heads = getattr(config, "n_head", getattr(config, "num_key_value_heads", getattr(config, "num_attention_heads", 12)))
            hidden_size = getattr(config, "n_embd", getattr(config, "hidden_size", 768))
            head_dim = hidden_size // getattr(config, "n_head", getattr(config, "num_attention_heads", 12))
            
            print(f"[Pipeline] LLM Config: {num_layers} layers, {num_heads} KV heads, {head_dim} head_dim.")
            
            # Initialize Graph Store and Subgraph Retriever
            self._store = SQLiteGraphStore(self.db_path)
            self._retriever = SubgraphRetriever(self._store, device=self.device)
            
            # Initialize MEN model
            print("[Pipeline] Initializing Memory Encoder Network (MEN)...")
            # input_dim matches 2 * embedding dimension of all-MiniLM-L6-v2 (2 * 384 = 768)
            self._men = MemoryEncoderNetwork(input_dim=768, num_layers=num_layers, num_heads=num_heads, head_dim=head_dim)
            self._men.to(self.device)
            self._men.eval()

    # Getters to lazy-initialize if not loaded
    @property
    def tokenizer(self):
        if self._tokenizer is None: self.initialize()
        return self._tokenizer

    @property
    def model(self):
        if self._model is None: self.initialize()
        return self._model

    @property
    def men(self):
        if self._men is None: self.initialize()
        return self._men

    @property
    def store(self):
        if self._store is None: self.initialize()
        return self._store

    @property
    def retriever(self):
        if self._retriever is None: self.initialize()
        return self._retriever

    def generate(self, conversation_id: str, query_text: str, k: int = 15, max_new_tokens: int = 100) -> str:
        """
        Runs pipeline: retrieves subgraph -> projects memory -> injects KVs -> generates response.
        Fully protected by VRAM guards, thread locks, and input bounds checking.
        """
        # 1. Sanitize input text and check buffer limits
        query_clean = InputBufferGuard.sanitize_string(query_text)
        
        # 2. Check GPU Memory status
        self.mem_guard.enforce_safety()
        
        # 3. Retrieve relevant subgraph and summaries
        print(f"[Pipeline] Querying memory store for conversation '{conversation_id}'...")
        subgraph = self.retriever.retrieve(conversation_id, query_clean, k=k)
        
        # 4. If memory exists, project it to KV cache using MEN
        past_key_values = None
        memory_length = 0  # Track how many memory tokens are injected
        if subgraph["triples"]:
            print(f"[Pipeline] Extracted {len(subgraph['triples'])} relevant triples. Encoding memory...")
            
            # Encode triples as: [enc(subj || pred); enc(obj)]
            x_list = []
            for trip in subgraph["triples"]:
                subj, pred, obj = trip
                # Subject + predicate vector representation
                sp_vector = self.retriever.embed_text(f"{subj} {pred}")
                # Object vector representation
                o_vector = self.retriever.embed_text(obj)
                
                # Concatenate to form a 768 dimensional input vector
                x_i = np.concatenate([sp_vector, o_vector])
                x_list.append(x_i)
                
            x_arr = np.stack(x_list) # shape: (M, 768)
            # Add batch dimension: (1, M, 768)
            x_tensor = torch.tensor(x_arr, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Validate input bounds
            InputBufferGuard.validate_tensor_bounds(x_tensor.shape[1])
            memory_length = x_tensor.shape[1]  # Number of memory positions
            
            # Pass through MEN to generate past_key_values as raw tuples
            with torch.no_grad():
                with GPULockManager():
                    raw_kv_tuples = self.men(x_tensor)
                    
                    # Convert raw tuples to DynamicCache (compatible with transformers v5.9.0+)
                    past_key_values = DynamicCache()
                    for k, v in raw_kv_tuples:
                        past_key_values.key_cache.append(k)
                        past_key_values.value_cache.append(v)
                    print(f"[Pipeline] Memory KV cache created: {memory_length} memory positions across {len(raw_kv_tuples)} layers.")
                    
        # 5. Build prompt
        # We append the summarized text context to the prompt
        summary = subgraph.get("summary", "")
        if summary:
            prompt = f"Distilled Past Context: {summary}\n\nUser: {query_clean}\nAssistant:"
        else:
            prompt = f"User: {query_clean}\nAssistant:"
            
        # 6. Run generation under GPU lock
        print("[Pipeline] Executing LLM generation...")
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Check input bounds for token length
        InputBufferGuard.validate_tensor_bounds(inputs.input_ids.shape[1])
        
        # Build attention mask that covers both injected memory positions and prompt tokens
        if past_key_values is not None and memory_length > 0:
            # The model needs to attend to: [memory_positions] + [prompt_tokens]
            # Memory positions get attention=1 (the model should attend to them)
            memory_mask = torch.ones(1, memory_length, dtype=torch.long, device=self.device)
            full_attention_mask = torch.cat([memory_mask, inputs.attention_mask], dim=-1)
        else:
            full_attention_mask = inputs.attention_mask
        
        with torch.no_grad():
            with GPULockManager():
                outputs = self.model.generate(
                    inputs.input_ids,
                    past_key_values=past_key_values,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    attention_mask=full_attention_mask
                )
                
        # 7. Decode and return response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Strip out prompt from output
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        return response
