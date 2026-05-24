import os
import sys

# Support running directly from inside the cgm directory or root
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import time
from cgm.schema import SQLiteGraphStore, HybridMemoryObject
from cgm.pipeline import CGMPipeline
from cgm.train import MEGATrainer

def seed_database(store: SQLiteGraphStore, conversation_id: str):
    """
    Seeds the SQLite database store with the dialogue history from Scenario 1
    (FastAPI / PostgreSQL setup preferences).
    """
    print(f"[Demo] Seeding database store for conversation '{conversation_id}'...")
    
    # Define past turns (representing turns 1 to 5)
    summary = (
        "The user is setting up a backend API using FastAPI. "
        "They have configured the database as PostgreSQL on port 8080. "
        "They specified using asyncpg for connectivity, typing_extensions for type hints, "
        "and Pydantic v2 for data schema validations. "
        "They also prefer Ruff for formatting with a max line length of 100 characters."
    )
    
    # Extract semantic triples manually to represent the compiled graph
    triples = [
        ["Project", "uses", "FastAPI"],
        ["Database", "connects_to", "PostgreSQL"],
        ["Database", "runs_on", "Port 8080"],
        ["Database", "uses", "asyncpg"],
        ["Project", "uses", "typing_extensions"],
        ["Project", "uses", "Pydantic v2"],
        ["Database", "has_table", "user_profiles"],
        ["Project", "formatted_by", "Ruff"],
        ["Ruff", "has_setting", "Line length 100"]
    ]
    
    entities = [
        {"name": "FastAPI", "type": "Technology", "description": "Backend web framework"},
        {"name": "PostgreSQL", "type": "Technology", "description": "Relational database system"},
        {"name": "Port 8080", "type": "Requirement", "description": "Port configuration for web service"},
        {"name": "asyncpg", "type": "Technology", "description": "Asynchronous Postgres connector"},
        {"name": "typing_extensions", "type": "Technology", "description": "Type hints python utility"},
        {"name": "Pydantic v2", "type": "Technology", "description": "Data schema validation library"},
        {"name": "user_profiles", "type": "CodeContext", "description": "Table for storing users info"},
        {"name": "Ruff", "type": "Technology", "description": "Formatter and linter tool"}
    ]
    
    # Create the Hybrid Memory Object (HMO)
    hmo = HybridMemoryObject(
        turn_id=1,
        timestamp=time.time(),
        summary=summary,
        triples=triples,
        entities=entities,
        # Create small dummy embeddings since retriever will embed on the fly
        sentence_embeddings=None,
        conversation_embedding=None,
        episodic_events=[{"event": "db_config_locked", "status": "completed"}],
        uncertainty_scores={"asyncpg_choice": 1.0}
    )
    
    # Save hmo to DB store
    store.save_hmo(conversation_id, hmo)
    print("[Demo] Database successfully seeded.")


def main():
    print("=" * 60)
    print("      CONVERSATIONAL GRAPH MEMORY (CGM) - RUN DEMO")
    print("=" * 60)
    
    conversation_id = "test_conversation_99"
    db_path = "data/cgm_memory.db"
    
    # Clean up old database if exists to ensure clean run
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
            print("[Demo] Removed old cgm_memory.db database for a clean run.")
        except Exception:
            pass
            
    # 1. Seed database FIRST (before loading heavy models)
    # We create the store directly to avoid triggering the pipeline's lazy model load
    store = SQLiteGraphStore(db_path)
    seed_database(store, conversation_id)
    
    # 2. Initialize Pipeline (targeting GPT-2 as a lightweight model for CPU/GPU)
    # Automatically runs on NVIDIA RTX A2000 (CUDA) if available
    pipeline = CGMPipeline(model_name="gpt2", db_path=db_path)
    
    # 3. Initialize Model and Tokenizer (loads everything once)
    print("\n[Demo] Initializing models and allocating GPU memory...")
    pipeline.initialize()
    
    # 4. Run Proof-of-Concept Adapter Training
    # Shows backprop updating the linear KV adapters under active thermal guards
    trainer = MEGATrainer(pipeline)
    trainer.fit_prototype(epochs=5)
    
    # 5. Run inference with memory KV injection
    query = "Write the python database connection string based on the configuration and port we established earlier."
    print(f"\n[Demo] Executing query: '{query}'")
    print("[Demo] Running CGMPipeline generation with KV Injection...")
    
    response = pipeline.generate(conversation_id, query, k=10, max_new_tokens=60)
    
    print("\n=== GENERATED RESPONSE (With Injected KV Cache) ===")
    print(response)
    print("===================================================\n")
    print("[Demo] Full process completed successfully under active hardware locks and memory guards.")
    print("=" * 60)

if __name__ == "__main__":
    main()
