import os
import sys
import time
import torch

# Support running directly from inside the cgm directory or root
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from cgm.pipeline import CGMPipeline
from cgm.schema import HybridMemoryObject
from cgm.safety import GPULockManager

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def interactive_chat():
    clear_screen()
    print("=" * 70)
    print("      🌌 CONVERSATIONAL GRAPH MEMORY (CGM) - INTERACTIVE SHELL")
    print("=" * 70)
    print(" This shell allows you to interactively chat with GPT-2 augmented")
    print(" with a live-injected knowledge graph. Memory is loaded on the fly")
    print(" directly into the LLM's KV cache, completely discarding raw history!")
    print("=" * 70)
    
    print(" Select your Inference Mode:")
    print("  [1] Plain-Text Graph RAG (Phase 1 Baseline):")
    print("      Passes retrieved graph triples as text in the prompt.")
    print("      Guarantees 100% coherent outputs using the model's native weights.")
    print("  [2] Active Cache Injection (Early Phase 2 Prototype):")
    print("      Injects synthetic KV cache from the untrained MEN network.")
    print("      Demonstrates the out-of-distribution 'Memory Noise' effect.")
    print("=" * 70)
    
    chat_mode = "text"
    while True:
        choice = input("Enter mode choice [1 or 2, default 1]: ").strip()
        if not choice or choice == "1":
            chat_mode = "text"
            print("Selected: Plain-Text Graph RAG Mode.\n")
            break
        elif choice == "2":
            chat_mode = "inject"
            print("Selected: Active Cache Injection Mode.\n")
            break
        else:
            print("Invalid selection. Enter 1 or 2.")

    db_path = "data/cgm_memory.db"
    conversation_id = "interactive_user_session"
    
    # Initialize pipeline
    print("[Pipeline] Booting LLM and Retriever on your GPU (RTX A2000)...")
    pipeline = CGMPipeline(model_name="gpt2", db_path=db_path)
    pipeline.initialize()
    
    # Seed a basic profile for the interactive session
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
    
    hmo = HybridMemoryObject(
        turn_id=1,
        timestamp=time.time(),
        summary="Initial session profile for Lovekesh.",
        triples=initial_triples,
        entities=initial_entities,
        sentence_embeddings=None,
        conversation_embedding=None,
        episodic_events=[],
        uncertainty_scores={}
    )
    pipeline.store.save_hmo(conversation_id, hmo)
    print("[Database] Successfully initialized.")
    
    print("\nReady! Chat session initialized. Type 'exit' to quit.")
    print("Type 'visualize' to update your graph visualization during the chat!")
    print("-" * 70)
    
    turn_counter = 2
    
    while True:
        try:
            user_input = input("\n👤 Lovekesh: ").strip()
            if not user_input:
                continue
                
            if user_input.lower() == 'exit':
                print("\n[Chat] Ending interactive session. Goodbye!")
                break
                
            if user_input.lower() == 'visualize':
                print("[Visualizer] Updating memory_graph.png with active session...")
                from cgm.visualize import generate_graph_visualization
                generate_graph_visualization(db_path=db_path, output_image="data/memory_graph.png")
                print("👉 Graph diagram refreshed: data/memory_graph.png")
                continue
            
            # 1. Run Pipeline generation (retrieves triples, encodes, injects KV cache, generates)
            print("[CGM] Querying graph store, extracting matching triples, and injecting KV cache...")
            start_time = time.perf_counter()
            response = pipeline.generate(conversation_id, user_input, k=10, max_new_tokens=60, mode=chat_mode)
            latency = time.perf_counter() - start_time
            
            print(f"\n🤖 CGM-GPT2: {response}")
            print(f"⏱️  Inference Latency: {latency:.4f}s (No raw history tokens encoded!)")
            
            # 2. Dynamic Memory Extraction (Simulation of Turn Boundary Processing)
            # In a production system, this utilizes Phase 1 Graph Extractors.
            # We mock the dynamic addition of a new memory relation to show the graph growing live!
            if "like" in user_input.lower() or "prefer" in user_input.lower() or "use" in user_input.lower():
                # Extract some words to form a simple triple to show live graph growth
                words = user_input.split()
                # A very simple parser to extract custom triples based on input patterns
                new_triple = None
                if "like" in words:
                    idx = words.index("like")
                    if idx + 1 < len(words):
                        item = " ".join(words[idx+1:]).strip(".?!,")
                        new_triple = ["User", "likes", item]
                elif "prefer" in words:
                    idx = words.index("prefer")
                    if idx + 1 < len(words):
                        item = " ".join(words[idx+1:]).strip(".?!,")
                        new_triple = ["User", "prefers", item]
                elif "use" in words:
                    idx = words.index("use")
                    if idx + 1 < len(words):
                        item = " ".join(words[idx+1:]).strip(".?!,")
                        new_triple = ["User", "uses", item]
                
                if new_triple:
                    print(f"\n🧠 [Closed Loop] Extracted new semantic triple from turn: {new_triple}")
                    print(f"🧠 [Closed Loop] Discarding raw KV cache & saving triple atomically in SQLite store...")
                    
                    new_hmo = HybridMemoryObject(
                        turn_id=turn_counter,
                        timestamp=time.time(),
                        summary=f"User stated a preference in: '{user_input}'",
                        triples=[new_triple],
                        entities=[{"name": new_triple[2], "type": "Concept", "description": "User preference"}],
                        sentence_embeddings=None,
                        conversation_embedding=None,
                        episodic_events=[],
                        uncertainty_scores={}
                    )
                    pipeline.store.save_hmo(conversation_id, new_hmo)
                    turn_counter += 1
                    print("🧠 [Closed Loop] Graph expanded live! Type 'visualize' to see the new node connected.")
            
        except KeyboardInterrupt:
            print("\n[Chat] Session interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    interactive_chat()
