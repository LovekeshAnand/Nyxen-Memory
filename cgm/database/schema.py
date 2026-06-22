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
    user_text: Optional[str] = None
    assistant_text: Optional[str] = None
    raw_embedding: Optional[np.ndarray] = None
    dia_id: Optional[str] = None


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
                    user_text TEXT,
                    assistant_text TEXT,
                    episodic_events TEXT,
                    uncertainty_scores TEXT,
                    tenant_id TEXT DEFAULT 'default_tenant',
                    user_id TEXT DEFAULT 'default_user',
                    dia_id TEXT,
                    PRIMARY KEY (conversation_id, turn_id, tenant_id, user_id)
                )
            """)
            
            # Migration helper for turns
            for col in ["user_text TEXT", "assistant_text TEXT", "tenant_id TEXT DEFAULT 'default_tenant'", "user_id TEXT DEFAULT 'default_user'", "dia_id TEXT"]:
                try:
                    cursor.execute(f"ALTER TABLE turns ADD COLUMN {col}")
                except sqlite3.OperationalError:
                    pass

            # 2. Entity nodes table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS entities (
                    conversation_id TEXT,
                    name TEXT,
                    type TEXT,
                    description TEXT,
                    frequency INTEGER DEFAULT 1,
                    tenant_id TEXT DEFAULT 'default_tenant',
                    user_id TEXT DEFAULT 'default_user',
                    PRIMARY KEY (conversation_id, name, tenant_id, user_id)
                )
            """)
            
            # Migration helper for entities
            for col in ["tenant_id TEXT DEFAULT 'default_tenant'", "user_id TEXT DEFAULT 'default_user'"]:
                try:
                    cursor.execute(f"ALTER TABLE entities ADD COLUMN {col}")
                except sqlite3.OperationalError:
                    pass
            
            # 3. Semantic relation edges table (Triples)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS triples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT,
                    turn_id INTEGER,
                    subject TEXT,
                    predicate TEXT,
                    object TEXT,
                    tenant_id TEXT DEFAULT 'default_tenant',
                    user_id TEXT DEFAULT 'default_user',
                    valid_from REAL,
                    valid_until REAL,
                    superseded_by INTEGER,
                    is_negated INTEGER DEFAULT 0
                )
            """)
            
            # Migration helper for triples
            for col in [
                "tenant_id TEXT DEFAULT 'default_tenant'",
                "user_id TEXT DEFAULT 'default_user'",
                "valid_from REAL",
                "valid_until REAL",
                "superseded_by INTEGER",
                "is_negated INTEGER DEFAULT 0"
            ]:
                try:
                    cursor.execute(f"ALTER TABLE triples ADD COLUMN {col}")
                except sqlite3.OperationalError:
                    pass
            
            # 4. Dense embeddings storage
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    conversation_id TEXT,
                    turn_id INTEGER,
                    embedding_type TEXT, -- 'sentence' or 'conversation'
                    dimensions TEXT,     -- JSON array representing dimensions
                    data BLOB,           -- Raw bytes of numpy array
                    tenant_id TEXT DEFAULT 'default_tenant',
                    user_id TEXT DEFAULT 'default_user',
                    PRIMARY KEY (conversation_id, turn_id, embedding_type, tenant_id, user_id)
                )
            """)

            # Migration helper for embeddings
            for col in ["tenant_id TEXT DEFAULT 'default_tenant'", "user_id TEXT DEFAULT 'default_user'"]:
                try:
                    cursor.execute(f"ALTER TABLE embeddings ADD COLUMN {col}")
                except sqlite3.OperationalError:
                    pass

            # 5. Turn-level dense embeddings storage for RAG pipeline
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS turn_embeddings (
                    conversation_id TEXT,
                    turn_id INTEGER,
                    data BLOB,
                    dim INTEGER,
                    tenant_id TEXT DEFAULT 'default_tenant',
                    user_id TEXT DEFAULT 'default_user',
                    PRIMARY KEY (conversation_id, turn_id, tenant_id, user_id)
                )
            """)

            # Migration helper for turn_embeddings
            for col in ["tenant_id TEXT DEFAULT 'default_tenant'", "user_id TEXT DEFAULT 'default_user'"]:
                try:
                    cursor.execute(f"ALTER TABLE turn_embeddings ADD COLUMN {col}")
                except sqlite3.OperationalError:
                    pass
            
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

    def save_hmo(self, conversation_id: str, hmo: HybridMemoryObject, tenant_id: str = "default_tenant", user_id: str = "default_user"):
        """
        Saves a Hybrid Memory Object atomically inside a thread lock and database transaction.
        Supports temporal conflict resolution and negation flagging.
        """
        import time
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            try:
                # Insert Turn metadata
                cursor.execute("""
                    INSERT OR REPLACE INTO turns 
                    (conversation_id, turn_id, timestamp, summary, user_text, assistant_text, episodic_events, uncertainty_scores, tenant_id, user_id, dia_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    conversation_id,
                    hmo.turn_id,
                    hmo.timestamp,
                    hmo.summary,
                    hmo.user_text,
                    hmo.assistant_text,
                    json.dumps(hmo.episodic_events),
                    json.dumps(hmo.uncertainty_scores),
                    tenant_id,
                    user_id,
                    hmo.dia_id
                ))
                
                # Insert Entities
                for entity in hmo.entities:
                    cursor.execute("""
                        INSERT INTO entities (conversation_id, name, type, description, frequency, tenant_id, user_id)
                        VALUES (?, ?, ?, ?, 1, ?, ?)
                        ON CONFLICT(conversation_id, name, tenant_id, user_id) DO UPDATE SET
                            frequency = frequency + 1,
                            description = COALESCE(excluded.description, description)
                    """, (conversation_id, entity["name"], entity["type"], entity["description"], tenant_id, user_id))
                    
                # Insert Triples with Conflict/Negation Resolution
                current_time = hmo.timestamp or time.time()
                for triple in hmo.triples:
                    subj, pred, obj = triple[0], triple[1], triple[2]
                    
                    # Detect negation triples
                    is_neg = 0
                    clean_pred = pred
                    if pred.startswith("NOT_") or "not_" in pred.lower() or pred.lower().startswith("not "):
                        is_neg = 1
                        if pred.startswith("NOT_"):
                            clean_pred = pred[4:]
                    
                    # Find conflicting active triples (same subject, same predicate/clean_pred, valid_until is null)
                    cursor.execute("""
                        SELECT id, object FROM triples
                        WHERE conversation_id = ? AND subject = ? AND (predicate = ? OR predicate = ?)
                          AND tenant_id = ? AND user_id = ? AND valid_until IS NULL
                    """, (conversation_id, subj, pred, clean_pred, tenant_id, user_id))
                    existing = cursor.fetchall()
                    
                    # If this new triple is negated, or if we have an active conflicting object
                    superseded_ids = []
                    for row in existing:
                        old_id = row["id"]
                        old_obj = row["object"]
                        
                        if is_neg == 1 or old_obj.lower().strip() != obj.lower().strip():
                            cursor.execute("""
                                UPDATE triples
                                SET valid_until = ?
                                WHERE id = ?
                            """, (current_time, old_id))
                            superseded_ids.append(old_id)
                    
                    # Insert the new triple
                    cursor.execute("""
                        INSERT INTO triples (conversation_id, turn_id, subject, predicate, object, tenant_id, user_id, valid_from, is_negated)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (conversation_id, hmo.turn_id, subj, pred, obj, tenant_id, user_id, current_time, is_neg))
                    new_id = cursor.lastrowid
                    
                    # Link superseded_by field
                    for old_id in superseded_ids:
                        cursor.execute("""
                            UPDATE triples
                            SET superseded_by = ?
                            WHERE id = ?
                        """, (new_id, old_id))
                    
                # Insert Embeddings
                for embed_type, arr in [("sentence", hmo.sentence_embeddings), ("conversation", hmo.conversation_embedding)]:
                    serialized = self._serialize_numpy(arr)
                    if serialized:
                        data_bytes, dims_str = serialized
                        cursor.execute("""
                            INSERT OR REPLACE INTO embeddings (conversation_id, turn_id, embedding_type, dimensions, data, tenant_id, user_id)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, (conversation_id, hmo.turn_id, embed_type, dims_str, data_bytes, tenant_id, user_id))
                
                # Insert turn-level raw embedding if present
                if hmo.raw_embedding is not None:
                    arr_f32 = hmo.raw_embedding.astype(np.float32)
                    data_bytes = arr_f32.tobytes()
                    dim = arr_f32.shape[-1]
                    cursor.execute("""
                        INSERT OR REPLACE INTO turn_embeddings (conversation_id, turn_id, data, dim, tenant_id, user_id)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (conversation_id, hmo.turn_id, data_bytes, dim, tenant_id, user_id))

                conn.commit()
            except Exception as e:
                conn.rollback()
                raise e
            finally:
                conn.close()

    def save_turn_embedding(self, conversation_id: str, turn_id: int, embedding: np.ndarray, tenant_id: str = "default_tenant", user_id: str = "default_user"):
        """
        Saves a turn-level embedding (384-dim) for a conversation turn.
        """
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            try:
                arr_f32 = embedding.astype(np.float32)
                data_bytes = arr_f32.tobytes()
                dim = arr_f32.shape[-1]
                cursor.execute("""
                    INSERT OR REPLACE INTO turn_embeddings (conversation_id, turn_id, data, dim, tenant_id, user_id)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (conversation_id, turn_id, data_bytes, dim, tenant_id, user_id))
                conn.commit()
            except Exception as e:
                conn.rollback()
                raise e
            finally:
                conn.close()

    def get_turn_embedding(self, conversation_id: str, turn_id: int, tenant_id: str = "default_tenant", user_id: str = "default_user") -> np.ndarray:
        """
        Retrieves the turn-level embedding for a conversation turn.
        """
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            try:
                cursor.execute("""
                    SELECT data, dim FROM turn_embeddings
                    WHERE conversation_id = ? AND turn_id = ? AND tenant_id = ? AND user_id = ?
                """, (conversation_id, turn_id, tenant_id, user_id))
                row = cursor.fetchone()
                if row is None:
                    raise KeyError(f"No turn embedding found for conversation '{conversation_id}', turn {turn_id}")
                arr = np.frombuffer(row["data"], dtype=np.float32)
                return arr.reshape((row["dim"],))
            finally:
                conn.close()

    def get_conversation_graph(self, conversation_id: str, tenant_id: str = "default_tenant", user_id: str = "default_user") -> Dict[str, Any]:
        """
        Retrieves the active aggregated semantic graph for a conversation.
        """
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Fetch triples
            cursor.execute("""
                SELECT turn_id, subject, predicate, object, is_negated FROM triples 
                WHERE conversation_id = ? AND tenant_id = ? AND user_id = ? AND valid_until IS NULL
            """, (conversation_id, tenant_id, user_id))
            triples = [dict(row) for row in cursor.fetchall()]
            
            # Fetch entities
            cursor.execute("""
                SELECT name, type, description, frequency FROM entities 
                WHERE conversation_id = ? AND tenant_id = ? AND user_id = ?
            """, (conversation_id, tenant_id, user_id))
            entities = [dict(row) for row in cursor.fetchall()]
            
            # Fetch summaries per turn
            cursor.execute("""
                SELECT turn_id, summary, timestamp FROM turns 
                WHERE conversation_id = ? AND tenant_id = ? AND user_id = ? ORDER BY turn_id ASC
            """, (conversation_id, tenant_id, user_id))
            turns = [dict(row) for row in cursor.fetchall()]
            
            conn.close()
            
        return {
            "conversation_id": conversation_id,
            "turns": turns,
            "entities": entities,
            "triples": triples
        }
