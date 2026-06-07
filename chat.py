import os
import sys
import time
import re
import logging
import requests
from typing import List

# Configure stdout and stderr to handle UTF-8 encoding on Windows to prevent UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Silence Hugging Face and network logging spams during chat boot
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

def extract_semantic_triples(text: str) -> List[List[str]]:
    """
    Advanced pattern-based semantic triple extractor.
    Handles conversational variations like 'working on a project', 'likes', 'prefers', 'uses', etc.
    """
    text_clean = text.strip().lower()
    triples = []
    
    # 1. Project / Work
    work_patterns = [
        r"i(?:'m| am)?\s+(?:working\s+on|building|developing|making|coding|creating|writing)\s+(?:a\s+)?(?:project|app|system|software|tool)?\s*(?:named|called)?\s*([a-zA-Z0-9_\-\s]{2,})",
        r"my\s+project\s+(?:is|named|called)\s*([a-zA-Z0-9_\-\s]{2,})"
    ]
    for pat in work_patterns:
        m = re.search(pat, text_clean)
        if m:
            item = m.group(1).strip().title()
            item = re.sub(r'[.?!,]+$', '', item).strip()
            if item == "Nyxen Memory":
                item = "Nyxen-Memory"
            if item:
                triples.append(["User", "works_on", item])
                break
                
    # 2. Preferences (Like/Prefer/Love/Uses)
    pref_patterns = [
        r"i\s+(?:like|prefer|love|enjoy)\s+([a-zA-Z0-9_\-\s\#\+]{2,})",
        r"i\s+(?:use|utilize|employ)\s+([a-zA-Z0-9_\-\s\#\+]{2,})"
    ]
    for pat in pref_patterns:
        m = re.search(pat, text_clean)
        if m:
            verb = "likes" if "like" in pat or "prefer" in pat or "love" in pat else "uses"
            item = m.group(1).strip().title()
            item = re.sub(r'[.?!,]+$', '', item).strip()
            norm_map = {"Rust": "Rust", "Python": "Python", "Fastapi": "FastAPI", "Postgresql": "PostgreSQL"}
            item = norm_map.get(item, item)
            if item:
                triples.append(["User", verb, item])
                break
                
    # 3. Roles / Job
    role_patterns = [
        r"i(?:'m| am)?\s+a\s+([a-zA-Z0-9_\-\s]{3,})",
        r"my\s+role\s+is\s+([a-zA-Z0-9_\-\s]{3,})",
        r"i\s+work\s+as\s+a\s+([a-zA-Z0-9_\-\s]{3,})"
    ]
    for pat in role_patterns:
        m = re.search(pat, text_clean)
        if m:
            item = m.group(1).strip().title()
            item = re.sub(r'[.?!,]+$', '', item).strip()
            if item:
                triples.append(["User", "role", item])
                break
                
    # 4. Name
    name_patterns = [
        r"my\s+name\s+is\s+([a-zA-Z0-9_\-\s]{2,})",
        r"i(?:'m| am)?\s+called\s+([a-zA-Z0-9_\-\s]{2,})"
    ]
    for pat in name_patterns:
        m = re.search(pat, text_clean)
        if m:
            item = m.group(1).strip().title()
            item = re.sub(r'[.?!,]+$', '', item).strip()
            if item:
                triples.append(["User", "name", item])
                break
                
    return triples

# Ensure root directory is in sys.path
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from cgm.core.pipeline import CGMPipeline
from cgm.database.schema import HybridMemoryObject
from cgm.safety.safety import GPULockManager
from cgm.visualization.visualize import start_live_visualizer_server

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def interactive_chat():
    clear_screen()
    
    db_path = "data/cgm_memory.db"
    conversation_id = "interactive_user_session"
    
    print("=" * 70)
    print("      CONVERSATIONAL GRAPH MEMORY (CGM) - AUTOMATED SHELL")
    print("=" * 70)
    
    ollama_url = "http://localhost:11434"
    ollama_exe = "C:\\Users\\ZBook\\AppData\\Local\\Programs\\Ollama\\ollama.exe"
    target_model = "qwen2.5:1.5b"
    model_name = f"ollama/{target_model}"
    
    # Determine the target models path, checking D: drive availability
    models_path = os.environ.get("OLLAMA_MODELS")
    if not models_path:
        models_path = "D:\\OllamaModels" if os.path.exists("D:\\") else os.path.expanduser("~/.ollama/models")
        
    # 1. Verify / Launch Ollama Server
    print("[Ollama] Checking connection to local Ollama server...")
    server_running = False
    try:
        res = requests.get(f"{ollama_url}/api/tags", timeout=3)
        if res.status_code == 200:
            server_running = True
    except Exception:
        pass
        
    if not server_running:
        print("[Ollama] Ollama server is not running. Launching it in the background...")
        os.makedirs(models_path, exist_ok=True)
        env = os.environ.copy()
        env["OLLAMA_MODELS"] = models_path
        
        try:
            import subprocess
            subprocess.Popen(
                [ollama_exe, "serve"],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )
            # Wait for server to boot up
            print("[Ollama] Waiting for Ollama server to boot...", end="", flush=True)
            for i in range(10):
                print(".", end="", flush=True)
                time.sleep(1)
                try:
                    res = requests.get(f"{ollama_url}/api/tags", timeout=1)
                    if res.status_code == 200:
                        server_running = True
                        print(" Success!")
                        break
                except Exception:
                    pass
            if not server_running:
                print(" Timeout.")
        except Exception as e:
            print(f"\n[Ollama] Error launching Ollama server: {e}")
            
    if not server_running:
        print("\n[Error] Could not connect or start local Ollama server. Please verify Ollama is installed at:")
        print(f"   {ollama_exe} or run it manually.")
        sys.exit(1)
        
    # 2. Check if model is pulled
    print(f"[Ollama] Verifying if model '{target_model}' is downloaded...")
    model_cached = False
    try:
        res = requests.get(f"{ollama_url}/api/tags", timeout=3)
        if res.status_code == 200:
            models = [m["name"] for m in res.json().get("models", [])]
            for m in models:
                if target_model in m or m in target_model:
                    model_name = f"ollama/{m}"
                    model_cached = True
                    break
    except Exception:
        pass
        
    if not model_cached:
        print(f"[Ollama] Model '{target_model}' not found. Downloading it automatically to local models folder (please wait)...")
        env = os.environ.copy()
        env["OLLAMA_MODELS"] = models_path
        try:
            import subprocess
            # Pull model
            process = subprocess.Popen(
                [ollama_exe, "pull", target_model],
                env=env
            )
            process.wait()
            model_name = f"ollama/{target_model}"
            print(f"[Ollama] Model '{target_model}' successfully downloaded!")
        except Exception as e:
            print(f"[Ollama] Error pulling model: {e}")
            sys.exit(1)
    else:
        print(f"[Ollama] Model '{target_model}' is verified and ready.")

    # 3. Initialize pipeline (sentence transformers + SQLite RAG setup)
    print("\n[Pipeline] Initializing pipeline components and warming up embedder...")
    pipeline = CGMPipeline(model_name=model_name, db_path=db_path)
    pipeline.initialize(verbose=False)
    print("[Pipeline] Warmup completed.")
    
    # 4. Seed initial profile nodes for context (if database is empty)
    graph_data = pipeline.store.get_conversation_graph(conversation_id)
    if not graph_data.get("turns"):
        print("\n[Database] Seeding initial profile nodes for context...")
        initial_triples = [
            ["User", "name", "Lovekesh"],
            ["User", "role", "Lead Researcher"],
            ["Project", "name", "Nyxen-Memory"],
            ["Project", "runs_on", "RTX A2000 GPU"],
            ["Project", "uses", "Conversational Graph Memory"]
        ]
        initial_entities = [
            {"name": "Lovekesh", "type": "Person", "description": "Lead researcher of CGM"},
            {"name": "Nyxen-Memory", "type": "Project", "description": "Conversational Graph Memory system"},
            {"name": "RTX A2000 GPU", "type": "Hardware", "description": "Workstation GPU with CUDA support"}
        ]
        
        seed_text = "Lovekesh Lead Researcher Nyxen-Memory RTX A2000 GPU Conversational Graph Memory"
        emb = pipeline.retriever.embed_text(seed_text)
        
        # Pre-seed in TurboVec using unique encoded index ID
        import numpy as np
        index_id = pipeline.retriever.encode_id(conversation_id, 1)
        pipeline.retriever.index.add_with_ids(emb.reshape(1, -1), np.array([index_id], dtype=np.uint64))
        pipeline.retriever.persist()
        
        hmo = HybridMemoryObject(
            turn_id=1,
            timestamp=time.time(),
            summary="Initial session profile for Lovekesh.",
            triples=initial_triples,
            entities=initial_entities,
            sentence_embeddings=None,
            conversation_embedding=None,
            episodic_events=[],
            uncertainty_scores={},
            user_text="Initialize session profile for Lovekesh Anand, Lead Researcher on Nyxen-Memory project with RTX A2000 GPU.",
            assistant_text="",
            raw_embedding=emb
        )
        pipeline.store.save_hmo(conversation_id, hmo)
        print("[Database] Successfully initialized.")
    else:
        print("\n[Database] Pre-existing session found. Seeding skipped.")
    
    # 5. Start live visualizer background HTTP server & browser launch
    start_live_visualizer_server(conversation_id=conversation_id, db_path=db_path, port=8050)
    
    print("\nReady! Chat session initialized. Type 'exit' to quit.")
    print("Type 'visualize' to update your graph visualization during the chat!")
    print("-" * 70)
    
    chat_mode = "text"
    while True:
        try:
            user_input = input("\n[User] Lovekesh: ").strip()
            if not user_input:
                continue
                
            if user_input.lower() == 'exit':
                print("\n[Chat] Ending interactive session. Goodbye!")
                break
                
            if user_input.lower() == 'visualize':
                print("[Visualizer] Updating memory_graph.png with active session...")
                from cgm.visualization.visualize import generate_graph_visualization
                generate_graph_visualization(conversation_id=conversation_id, db_path=db_path, output_image="data/memory_graph.png")
                print("[Visualizer] Graph diagram refreshed: data/memory_graph.png")
                continue
            
            # 1. Run Pipeline generation
            print("[CGM] Querying graph store, extracting matching turns, and generating response...")
            start_time = time.perf_counter()
            response = pipeline.generate(conversation_id, user_input, k=10, max_new_tokens=150, mode=chat_mode)
            latency = time.perf_counter() - start_time
            
            print(f"\n[Model] CGM-Model: {response}")
            print(f"[Latency] Inference Latency: {latency:.4f}s")
            
            # Get the active turn ID that was just saved by generate()
            with pipeline.store._lock:
                conn = pipeline.store._get_connection()
                cursor = conn.cursor()
                cursor.execute("SELECT MAX(turn_id) FROM turns WHERE conversation_id = ?", (conversation_id,))
                row = cursor.fetchone()
                turn_counter = row[0] if (row is not None and row[0] is not None) else 2
                conn.close()
                
            # 2. Dynamic Memory Extraction (Simulation of Turn Boundary Processing)
            new_triples = extract_semantic_triples(user_input)
            if new_triples:
                for new_triple in new_triples:
                    print(f"\n[Closed Loop] Extracted new semantic triple from turn: {new_triple}")
                    print(f"[Closed Loop] Saving triple atomically in SQLite store...")
                    
                    # Store triples and entities in SQLite
                    with pipeline.store._lock:
                        conn = pipeline.store._get_connection()
                        cursor = conn.cursor()
                        cursor.execute("""
                            INSERT INTO entities (conversation_id, name, type, description, frequency)
                            VALUES (?, ?, ?, ?, 1)
                            ON CONFLICT(conversation_id, name) DO UPDATE SET
                                frequency = frequency + 1,
                                description = COALESCE(excluded.description, description)
                        """, (conversation_id, new_triple[2], "Concept" if new_triple[1] in ["likes", "prefers", "uses", "works_on"] else "Value", f"User relation: {new_triple[2]}"))
                        
                        cursor.execute("""
                            INSERT INTO triples (conversation_id, turn_id, subject, predicate, object)
                            VALUES (?, ?, ?, ?, ?)
                        """, (conversation_id, turn_counter, new_triple[0], new_triple[1], new_triple[2]))
                        conn.commit()
                        conn.close()
                print("[Closed Loop] Graph expanded live! Your live concept dashboard has refreshed.")
            
        except KeyboardInterrupt:
            print("\n[Chat] Session interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n[Error] Error: {e}")

if __name__ == "__main__":
    interactive_chat()
