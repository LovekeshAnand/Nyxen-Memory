import os
import sys
import time

# Ensure root directory is in sys.path
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from cgm.database.schema import SQLiteGraphStore, HybridMemoryObject
from cgm.core.pipeline import CGMPipeline
from cgm.training.train import MEGATrainer

def seed_database(pipeline: CGMPipeline, conversation_id: str):
    """
    Seeds the SQLite database store with dense turn representations and semantic triples
    representing the dialogue history.
    """
    print(f"[Demo] Seeding database store for conversation '{conversation_id}'...")
    
    summary = (
        "The user is setting up a backend API using FastAPI. "
        "They have configured the database as PostgreSQL on port 8080. "
        "They specified using asyncpg for connectivity, typing_extensions for type hints, "
        "and Pydantic v2 for data schema validations. "
        "They also prefer Ruff for formatting with a max line length of 100 characters."
    )
    
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
    
    # Store turn using retriever (embeds text and updates SQLite + TurboVec)
    turn_text = (
        "We are setting up a backend FastAPI API project. The database connects to PostgreSQL "
        "on Port 8080 using asyncpg for asynchronous connectivity. The project uses typing_extensions "
        "and Pydantic v2 for validation, with user_profiles as a database table. Code is formatted by "
        "Ruff with a max line length setting of 100."
    )
    
    pipeline.retriever.store_turn(
        conversation_id=conversation_id,
        turn_id=1,
        text=turn_text,
        summary=summary,
        user_text="What backend stack setup did we decide on?",
        assistant_text=f"FastAPI with PostgreSQL on port 8080 using asyncpg. Validation is via Pydantic v2 and code is formatted with Ruff."
    )
    
    # Save triples and entities to SQLite for concept map visualization
    with pipeline.store._lock:
        conn = pipeline.store._get_connection()
        cursor = conn.cursor()
        for ent in entities:
            cursor.execute("""
                INSERT OR REPLACE INTO entities (conversation_id, name, type, description)
                VALUES (?, ?, ?, ?)
            """, (conversation_id, ent["name"], ent["type"], ent["description"]))
        for trip in triples:
            cursor.execute("""
                INSERT OR REPLACE INTO triples (conversation_id, turn_id, subject, predicate, object)
                VALUES (?, 1, ?, ?, ?)
            """, (conversation_id, trip[0], trip[1], trip[2]))
        conn.commit()
        conn.close()
        
    print("[Demo] Database successfully seeded with RAG embeddings and graph structures.")


def main():
    print("=" * 60)
    print("      CONVERSATIONAL GRAPH MEMORY (CGM) - RUN DEMO")
    print("=" * 60)
    
    conversation_id = "test_conversation_99"
    db_path = "data/cgm_memory.db"
    index_path = "data/cgm_rag.tvim"
    
    # Clean up old database and index to ensure a clean run
    for filepath in [db_path, index_path]:
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                print(f"[Demo] Removed old {filepath} for a clean run.")
            except Exception:
                pass
                
    # 1. Initialize Pipeline (targeting GPT-2 as a lightweight model for CPU/GPU)
    # Automatically runs on NVIDIA RTX A2000 (CUDA) if available
    pipeline = CGMPipeline(model_name="gpt2", db_path=db_path)
    
    # 2. Initialize Model and Tokenizer (loads everything once)
    print("\n[Demo] Initializing models and allocating GPU memory...")
    pipeline.initialize()
    
    # 3. Seed database after initialization using the loaded pipeline
    seed_database(pipeline, conversation_id)
    
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
