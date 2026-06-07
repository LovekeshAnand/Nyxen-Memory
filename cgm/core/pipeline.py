import torch
import numpy as np
import logging
import requests
import json
from typing import Dict, Any, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from cgm.database.schema import SQLiteGraphStore
from cgm.retrieval.rag_retriever import RAGRetriever, TurnMemory
from cgm.core.model import MemoryEncoderNetwork
from cgm.safety.safety import GPULockManager, GPUMemoryGuard, InputBufferGuard

# Silence Hugging Face and HTTPX network request logger spam
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

class CGMPipeline:
    """
    End-to-End inference pipeline for Conversational Graph Memory.
    Loads the target LLM (locally via PyTorch or calls local Ollama endpoints),
    retrieves relevant subgraph memory, projects it to KV space, and runs inference.
    """
    def __init__(self, model_name: str = "gpt2", db_path: str = "data/cgm_memory.db", device: str = None):
        self.db_path = db_path
        self.model_name = model_name
        self.device = device
        
        # Initialize GPU guards
        self.mem_guard = GPUMemoryGuard()
        
        # Check if Ollama model is requested
        if model_name.startswith("ollama/") or model_name == "ollama":
            self.use_ollama = True
            self.ollama_model = model_name.split("/", 1)[1] if "/" in model_name else "llama3"
            print(f"[Pipeline] Ollama integration enabled. Target model: '{self.ollama_model}'")
        else:
            self.use_ollama = False
            self.ollama_model = None

        self._tokenizer = None
        self._model = None
        self._men = None
        self._store = None
        self._retriever = None
        self.active_cache = None

    def initialize(self, verbose: bool = True):
        """
        Loads the LLM, tokenizer, database, and retriever.
        Ensures thread safety and checks available VRAM to prevent OOM.
        """
        # Always initialize store and retriever (which loads sentence transformers)
        self._store = SQLiteGraphStore(self.db_path)
        
        # Detect device
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self._retriever = RAGRetriever(self._store, device=self.device)
        
        # Warm up retriever to cache SentenceTransformer in GPU VRAM during boot
        if verbose:
            print("[Pipeline] Warming up SentenceTransformer embedder on GPU...")
        _ = self._retriever.embed_text("warmup query")

        if self.use_ollama:
            # For Ollama, we do not load a local PyTorch generative model.
            # Check if Ollama service is responsive
            url = "http://localhost:11434/api/tags"
            try:
                res = requests.get(url, timeout=3)
                if res.status_code == 200:
                    models_list = [m["name"] for m in res.json().get("models", [])]
                    print(f"[Pipeline] Connected to Ollama server. Available models: {models_list}")
                    # Match model names loosely
                    found = False
                    for m in models_list:
                        if self.ollama_model in m or m in self.ollama_model:
                            self.ollama_model = m  # normalize
                            found = True
                            break
                    if not found:
                        print(f"[Pipeline] Warning: Requested model '{self.ollama_model}' was not found in Ollama cache.")
                        print(f"[Pipeline] Please run: 'ollama pull {self.ollama_model}' in your terminal.")
                else:
                    print(f"[Pipeline] Warning: Ollama returned status {res.status_code}")
            except Exception as e:
                print(f"[Pipeline] Warning: Could not connect to local Ollama server. Is it running?")
                print(f"[Pipeline] Please start Ollama before executing queries.")
        else:
            # Proactively check available VRAM before local model load
            self.mem_guard.enforce_safety(model_size_est_mb=500.0)
            
            # Thread-safe load of local PyTorch model
            with GPULockManager():
                if verbose:
                    print(f"[Pipeline] Loading tokenizer for '{self.model_name}'...")
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                
                # Set pad token if not defined
                if self._tokenizer.pad_token is None:
                    self._tokenizer.pad_token = self._tokenizer.eos_token
                    
                if verbose:
                    print(f"[Pipeline] Loading LLM '{self.model_name}' on device: {self.device}...")
                self._model = AutoModelForCausalLM.from_pretrained(self.model_name)
                self._model.to(self.device)
                self._model.eval()
                
                # Get LLM architecture parameters for MEN mapping
                config = self._model.config
                num_layers = getattr(config, "n_layer", getattr(config, "num_hidden_layers", 12))
                num_heads = getattr(config, "n_head", getattr(config, "num_key_value_heads", getattr(config, "num_attention_heads", 12)))
                hidden_size = getattr(config, "n_embd", getattr(config, "hidden_size", 768))
                head_dim = hidden_size // getattr(config, "n_head", getattr(config, "num_attention_heads", 12))
                
                if verbose:
                    print(f"[Pipeline] LLM Config: {num_layers} layers, {num_heads} KV heads, {head_dim} head_dim.")
                
                # Initialize MEN model
                if verbose:
                    print("[Pipeline] Initializing Memory Encoder Network (MEN)...")
                # input_dim matches 2 * embedding dimension of all-MiniLM-L6-v2 (2 * 384 = 768)
                self._men = MemoryEncoderNetwork(input_dim=768, num_layers=num_layers, num_heads=num_heads, head_dim=head_dim)
                self._men.to(self.device)
                self._men.eval()

        # Log pipeline initialize event
        try:
            from cgm.visualization.visualize import log_pipeline_event
            target_desc = f"Ollama model '{self.ollama_model}'" if self.use_ollama else f"Local model '{self.model_name}'"
            log_pipeline_event("info", {
                "message": f"Pipeline successfully initialized for {target_desc} on device: {self.device}."
            })
        except Exception:
            pass

    # Getters to lazy-initialize if not loaded
    @property
    def tokenizer(self):
        if not self.use_ollama and self._tokenizer is None: self.initialize()
        return self._tokenizer

    @property
    def model(self):
        if not self.use_ollama and self._model is None: self.initialize()
        return self._model

    @property
    def men(self):
        if not self.use_ollama and self._men is None: self.initialize()
        return self._men

    @property
    def store(self):
        if self._store is None: self.initialize()
        return self._store

    @property
    def retriever(self):
        if self._retriever is None: self.initialize()
        return self._retriever

    def generate(self, conversation_id: str, query_text: str, k: int = 15, max_new_tokens: int = 100, mode: str = "inject") -> str:
        """
        Runs pipeline: retrieves memories -> projects memory -> injects KVs -> generates response.
        Fully protected by VRAM guards, thread locks, and input bounds checking.
        """
        # 1. Sanitize input text and check buffer limits
        query_clean = InputBufferGuard.sanitize_string(query_text)
        
        # 2. Check GPU Memory status (only if running local PyTorch models)
        if not self.use_ollama:
            self.mem_guard.enforce_safety()
        
        # 3. Retrieve relevant memories
        print(f"[Pipeline] Querying memory store for conversation '{conversation_id}'...")
        memories = self.retriever.retrieve(conversation_id, query_clean, k=k)
        
        # 4. If memory exists and mode is 'inject', project it to KV cache using MEN (only for PyTorch)
        past_key_values = None
        memory_length = 0  # Track how many memory tokens are injected
        self.active_cache = None
        
        if self.use_ollama:
            # Ollama does not support KV-cache injection, fallback to standard text RAG
            mode = "text"
            
        if mode == "inject" and memories:
            print(f"[Pipeline] Extracted {len(memories)} relevant turns. Encoding memory...")
            try:
                from cgm.visualization.visualize import log_pipeline_event
                log_pipeline_event("info", {
                    "message": f"Projecting {len(memories)} retrieved turns into synthetic KV cache matrices."
                })
            except Exception:
                pass
            
            x_list = []
            for mem in memories:
                # Compute embedding on the fly if not present
                if mem.embedding is None:
                    turn_text = f"{mem.user_text} {mem.assistant_text}"
                    mem.embedding = self.retriever.embed_text(turn_text)
                # Concatenate the embedding with itself to form a 768-dim vector
                x_i = np.concatenate([mem.embedding, mem.embedding])
                x_list.append(x_i)
                
            x_arr = np.stack(x_list) # shape: (M, 768)
            x_tensor = torch.tensor(x_arr, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Validate input bounds
            InputBufferGuard.validate_tensor_bounds(x_tensor.shape[1])
            memory_length = x_tensor.shape[1]  # Number of memory positions
            
            # Pass through MEN to generate past_key_values
            with torch.no_grad():
                with GPULockManager():
                    raw_kv_tuples = self.men(x_tensor)
                    past_key_values = DynamicCache()
                    for idx, (k_val, v_val) in enumerate(raw_kv_tuples):
                        past_key_values.update(k_val, v_val, idx)
            
            # Store in pipeline so the background compressor can monitor it
            self.active_cache = past_key_values
            print(f"[Pipeline] Memory KV cache injected: {memory_length} positions across {len(raw_kv_tuples)} layers.")

        # 5. Build prompt
        if mode == "text" and memories:
            # Sort retrieved memories chronologically to maintain temporal history context
            sorted_memories = sorted(memories, key=lambda m: m.turn_id)
            summary_parts = []
            for mem in sorted_memories:
                user_msg = mem.user_text if mem.user_text else mem.summary
                assistant_msg = mem.assistant_text if mem.assistant_text else ""
                if assistant_msg:
                    summary_parts.append(f"Turn {mem.turn_id} - User: {user_msg} | Assistant: {assistant_msg}")
                else:
                    summary_parts.append(f"Turn {mem.turn_id} - User: {user_msg}")
            summary_context = "\n".join(summary_parts)
            prompt = f"Distilled Past Conversation History:\n{summary_context}\n\nUser: {query_clean}\nAssistant:"
        else:
            prompt = f"User: {query_clean}\nAssistant:"
            
        # 6. Run generation
        if self.use_ollama:
            print(f"[Pipeline] Sending request to Ollama model '{self.ollama_model}'...")
            
            # Log generate activity to the visualizer
            try:
                from cgm.visualization.visualize import log_pipeline_event
                log_pipeline_event("generate", {
                    "conversation_id": conversation_id,
                    "prompt": prompt,
                    "memory_length": 0,
                    "message": f"Invoking Ollama inference. Model: '{self.ollama_model}'. Retained turns count: {len(memories)}."
                })
            except Exception:
                pass
                
            url = "http://localhost:11434/api/generate"
            payload = {
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False
            }
            try:
                res = requests.post(url, json=payload, timeout=30)
                if res.status_code == 200:
                    result = res.json()
                    response = result.get("response", "").strip()
                else:
                    response = f"Ollama Error: HTTP {res.status_code} - {res.text}"
                    print(f"[Pipeline] {response}")
            except Exception as e:
                response = f"Ollama Connection Error: {e}. Ensure Ollama is running."
                print(f"[Pipeline] {response}")
        else:
            print("[Pipeline] Executing LLM generation locally...")
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            InputBufferGuard.validate_tensor_bounds(inputs.input_ids.shape[1])
            
            position_ids = None
            if past_key_values is not None and memory_length > 0:
                memory_mask = torch.ones(1, memory_length, dtype=torch.long, device=self.device)
                full_attention_mask = torch.cat([memory_mask, inputs.attention_mask], dim=-1)
                
                prompt_length = inputs.input_ids.shape[1]
                position_ids = torch.arange(
                    memory_length, 
                    memory_length + prompt_length, 
                    dtype=torch.long, 
                    device=self.device
                ).unsqueeze(0)
            else:
                full_attention_mask = inputs.attention_mask

            # Log generate activity to the visualizer
            try:
                from cgm.visualization.visualize import log_pipeline_event
                log_pipeline_event("generate", {
                    "conversation_id": conversation_id,
                    "prompt": prompt,
                    "memory_length": memory_length,
                    "message": f"Invoking autoregressive generation. Prompt size: {inputs.input_ids.shape[1]} tokens. KV injection size: {memory_length} tokens."
                })
            except Exception:
                pass
                
            with torch.no_grad():
                with GPULockManager():
                    outputs = self.model.generate(
                        inputs.input_ids,
                        past_key_values=past_key_values,
                        max_new_tokens=max_new_tokens,
                        pad_token_id=self.tokenizer.pad_token_id,
                        attention_mask=full_attention_mask,
                        position_ids=position_ids,
                        do_sample=True,
                        temperature=0.8,
                        top_k=50,
                        top_p=0.95,
                        repetition_penalty=1.2
                    )
                    
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
                
            for stop_word in ["\nUser:", "\nAssistant:", "\n👤", "\n🤖", "User:", "Assistant:"]:
                if stop_word in response:
                    response = response.split(stop_word)[0].strip()
                
        # 8. Store the newly generated turn (user query + assistant response) to retriever memory
        try:
            with self.store._lock:
                conn = self.store._get_connection()
                cursor = conn.cursor()
                cursor.execute("SELECT MAX(turn_id) FROM turns WHERE conversation_id = ?", (conversation_id,))
                row = cursor.fetchone()
                max_tid = row[0] if (row is not None and row[0] is not None) else 0
                conn.close()
            new_turn_id = max_tid + 1
            
            turn_text = f"{query_clean} {response}"
            self.retriever.store_turn(
                conversation_id=conversation_id,
                turn_id=new_turn_id,
                text=turn_text,
                summary=f"User: '{query_clean[:100]}...' | Assistant: '{response[:100]}...'",
                user_text=query_clean,
                assistant_text=response
            )
            print(f"[Pipeline] Successfully stored new turn {new_turn_id} to memory.")
        except Exception as e:
            print(f"[Pipeline] Error saving turn to store: {e}")
            
        # 9. Trigger background compressor if VRAM is low (only for PyTorch local model)
        if not self.use_ollama:
            free_bytes, total_bytes = self.mem_guard.check_vram()
            if total_bytes > 0:
                free_mb = free_bytes / (1024 * 1024)
                if free_mb < 800.0:
                    if not hasattr(self, "compressor") or self.compressor is None:
                        from cgm.core.compressor import KVCompressor
                        self.compressor = KVCompressor()
                        self.compressor.start_background_monitor(self)
                    # Proactively run a compression step
                    if self.active_cache is not None:
                        self.compressor.compress(self, self.active_cache)

        return response
