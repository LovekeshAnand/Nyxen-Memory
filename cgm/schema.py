import os
import json
import sqlite3
import numpy as np
import threading
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

@dataclass
class HybridMemoryObject:
    """
    Structured per-turn memory container (HMO).
    Combines symbolic elements (triples, entities), latent representations (embeddings),
    narrative summaries, and episodic temporal metadata.
    """
    turn_id: int
    timestamp: float
    summary: str
    triples: List[List[str]] = field(default_factory=list)  # Each triple: [subject, predicate, object]
    entities: List[Dict[str, Any]] = field(default_factory=list)  # Dict: {name, type, description}
    sentence_embeddings: Optional[np.ndarray] = None  # shape: (n_sentences, d_embed)
    conversation_embedding: Optional[np.ndarray] = None  # shape: (d_embed,)
    episodic_events: List[Dict[str, Any]] = field(default_factory=list)
    uncertainty_scores: Dict[str, float] = field(default_factory=dict)


class SQLiteGraphStore:
    """
    Thread-safe SQLite persistent store for Conversational Graph Memory.
    Ensures fully atomic updates using transactions and a thread lock.
    """
    def __init__(self, db_path: str = "cgm_memory.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        # Return row-factory to make results easy to parse
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """
        Initializes schema tables if they do not exist.
        """
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # 1. Conversation turns metadata table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS turns (
                    conversation_id TEXT,
                    turn_id INTEGER,
                    timestamp REAL,
                    summary TEXT,
                    episodic_events TEXT,
                    uncertainty_scores TEXT,
                    PRIMARY KEY (conversation_id, turn_id)
                )
            """)
            
            # 2. Entity nodes table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS entities (
                    conversation_id TEXT,
                    name TEXT,
                    type TEXT,
                    description TEXT,
                    frequency INTEGER DEFAULT 1,
                    PRIMARY KEY (conversation_id, name)
                )
            """)
            
            # 3. Semantic relation edges table (Triples)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS triples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT,
                    turn_id INTEGER,
                    subject TEXT,
                    predicate TEXT,
                    object TEXT,
                    FOREIGN KEY (conversation_id, turn_id) REFERENCES turns (conversation_id, turn_id)
                )
            """)
            
            # 4. Dense embeddings storage
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    conversation_id TEXT,
                    turn_id INTEGER,
                    embedding_type TEXT, -- 'sentence' or 'conversation'
                    dimensions TEXT,     -- JSON array representing dimensions
                    data BLOB,           -- Raw bytes of numpy array
                    PRIMARY KEY (conversation_id, turn_id, embedding_type)
                )
            """)
            
            conn.commit()
            conn.close()

    def _serialize_numpy(self, array: Optional[np.ndarray]) -> Optional[Tuple[bytes, str]]:
        if array is None:
            return None
        # Convert to float32 for storage savings
        arr_f32 = array.astype(np.float32)
        dimensions = json.dumps(list(arr_f32.shape))
        data_bytes = arr_f32.tobytes()
        return data_bytes, dimensions

    def _deserialize_numpy(self, data_bytes: bytes, dimensions_str: str) -> np.ndarray:
        dims = json.loads(dimensions_str)
        arr = np.frombuffer(data_bytes, dtype=np.float32)
        return arr.reshape(dims)

    def save_hmo(self, conversation_id: str, hmo: HybridMemoryObject):
        """
        Saves a Hybrid Memory Object atomically inside a thread lock and database transaction.
        """
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            try:
                # Insert Turn metadata
                cursor.execute("""
                    INSERT OR REPLACE INTO turns 
                    (conversation_id, turn_id, timestamp, summary, episodic_events, uncertainty_scores)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    conversation_id,
                    hmo.turn_id,
                    hmo.timestamp,
                    hmo.summary,
                    json.dumps(hmo.episodic_events),
                    json.dumps(hmo.uncertainty_scores)
                ))
                
                # Insert Entities
                for entity in hmo.entities:
                    cursor.execute("""
                        INSERT INTO entities (conversation_id, name, type, description, frequency)
                        VALUES (?, ?, ?, ?, 1)
                        ON CONFLICT(conversation_id, name) DO UPDATE SET
                            frequency = frequency + 1,
                            description = COALESCE(excluded.description, description)
                    """, (conversation_id, entity["name"], entity["type"], entity["description"]))
                    
                # Insert Triples
                for triple in hmo.triples:
                    cursor.execute("""
                        INSERT INTO triples (conversation_id, turn_id, subject, predicate, object)
                        VALUES (?, ?, ?, ?, ?)
                    """, (conversation_id, hmo.turn_id, triple[0], triple[1], triple[2]))
                    
                # Insert Embeddings
                for embed_type, arr in [("sentence", hmo.sentence_embeddings), ("conversation", hmo.conversation_embedding)]:
                    serialized = self._serialize_numpy(arr)
                    if serialized:
                        data_bytes, dims_str = serialized
                        cursor.execute("""
                            INSERT OR REPLACE INTO embeddings (conversation_id, turn_id, embedding_type, dimensions, data)
                            VALUES (?, ?, ?, ?, ?)
                        """, (conversation_id, hmo.turn_id, embed_type, dims_str, data_bytes))
                
                conn.commit()
            except Exception as e:
                conn.rollback()
                raise e
            finally:
                conn.close()

    def get_conversation_graph(self, conversation_id: str) -> Dict[str, Any]:
        """
        Retrieves the aggregated semantic graph (all triples and entities) for a conversation.
        """
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Fetch triples
            cursor.execute("SELECT turn_id, subject, predicate, object FROM triples WHERE conversation_id = ?", (conversation_id,))
            triples = [dict(row) for row in cursor.fetchall()]
            
            # Fetch entities
            cursor.execute("SELECT name, type, description, frequency FROM entities WHERE conversation_id = ?", (conversation_id,))
            entities = [dict(row) for row in cursor.fetchall()]
            
            # Fetch summaries per turn
            cursor.execute("SELECT turn_id, summary, timestamp FROM turns WHERE conversation_id = ? ORDER BY turn_id ASC", (conversation_id,))
            turns = [dict(row) for row in cursor.fetchall()]
            
            conn.close()
            
        return {
            "conversation_id": conversation_id,
            "turns": turns,
            "entities": entities,
            "triples": triples
        }
