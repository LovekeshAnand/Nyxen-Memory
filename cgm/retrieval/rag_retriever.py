import os
import time
import numpy as np
from typing import List, Optional
from dataclasses import dataclass
from cgm.database.schema import SQLiteGraphStore, HybridMemoryObject
from cgm.safety.safety import GPULockManager
from cgm.visualization.visualize import log_pipeline_event
from turbovec import IdMapIndex

@dataclass
class TurnMemory:
    turn_id: int
    summary: str
    user_text: str
    assistant_text: str
    score: float
    embedding: Optional[np.ndarray] = None

class RAGRetriever:
    """
    Dense vector retrieval system backed by TurboVec (IdMapIndex) and SQLite.
    Converts queries and conversation turns to 384-dimensional embeddings
    and searches them with time-window filtering.
    """
    def __init__(self, store: SQLiteGraphStore, device: Optional[str] = None, 
                 embedding_model_name: str = "all-MiniLM-L6-v2", 
                 index_path: str = "data/cgm_rag.tvim"):
        self.store = store
        self.device = device
        self.embedding_model_name = embedding_model_name
        self.index_path = index_path
        self._embedder = None
        self.index = None
        
        # Load or initialize the index
        self._init_index()

    def _init_index(self):
        """
        Loads the TurboVec IdMapIndex from disk if it exists, or initializes a new one.
        """
        # Ensure parent directory exists
        if self.index_path:
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            if os.path.exists(self.index_path):
                print(f"[RAGRetriever] Loading existing TurboVec index from: {self.index_path}")
                try:
                    with GPULockManager():
                        self.index = IdMapIndex.load(self.index_path)
                except Exception as e:
                    print(f"[RAGRetriever] Error loading index: {e}. Creating a new one.")
                
        if self.index is None:
            print("[RAGRetriever] Initializing a new TurboVec IdMapIndex (384-dim, 4-bit width)...")
            with GPULockManager():
                self.index = IdMapIndex(dim=384, bit_width=4)

    @property
    def embedder(self):
        """
        Lazy loader for sentence-transformers to avoid boot overhead.
        """
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            import torch
            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                
            print(f"[RAGRetriever] Loading SentenceTransformer '{self.embedding_model_name}' on device: {self.device}...")
            with GPULockManager():
                self._embedder = SentenceTransformer(self.embedding_model_name, device=self.device)
        return self._embedder

    def embed_text(self, text: str) -> np.ndarray:
        """
        Computes dense vector representation of a given string.
        """
        with GPULockManager():
            embedding = self.embedder.encode(text, convert_to_numpy=True)
        return embedding

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Computes dense vector representations of a list of strings in a single batch.
        """
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        with GPULockManager():
            embeddings = self.embedder.encode(texts, convert_to_numpy=True)
        return embeddings

    def persist(self):
        """
        Saves the TurboVec index to disk.
        """
        if self.index is not None and self.index_path:
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            with GPULockManager():
                self.index.write(self.index_path)
            print(f"[RAGRetriever] TurboVec index successfully persisted to: {self.index_path}")

    def encode_id(self, conversation_id: str, turn_id: int) -> int:
        """
        Encodes a (conversation_id, turn_id) tuple into a stable unique 64-bit unsigned integer
        to avoid conflicts in the shared TurboVec index.
        """
        import hashlib
        h = hashlib.sha256(conversation_id.encode('utf-8')).digest()
        # Extract 48 bits (6 bytes) for the conversation hash
        conv_hash = int.from_bytes(h[:6], byteorder='big')
        # Combine 48-bit hash and 16-bit turn_id (clipped to 0xFFFF)
        return (conv_hash << 16) | (turn_id & 0xFFFF)

    def store_turn(self, conversation_id: str, turn_id: int, text: str, 
                   summary: Optional[str] = None, user_text: Optional[str] = None, 
                   assistant_text: Optional[str] = None):
        """
        Embeds the turn, adds it to the TurboVec index, and stores it in the SQLite database.
        """
        # 1. Compute embedding
        embedding = self.embed_text(text)
        
        # 2. Add to TurboVec index using unique encoded ID
        index_id = self.encode_id(conversation_id, turn_id)
        with GPULockManager():
            self.index.add_with_ids(embedding.reshape(1, -1), np.array([index_id], dtype=np.uint64))
            
        # 3. Store embedding and metadata in SQLite Graph Store
        self.store.save_turn_embedding(conversation_id, turn_id, embedding)
        
        # Determine summary and text values
        if summary is None:
            summary = text[:250] + "..." if len(text) > 250 else text
            
        hmo = HybridMemoryObject(
            turn_id=turn_id,
            timestamp=time.time(),
            summary=summary,
            user_text=user_text or text,
            assistant_text=assistant_text or "",
            raw_embedding=embedding
        )
        self.store.save_hmo(conversation_id, hmo)
        
        # 4. Auto-persist TurboVec index
        self.persist()

        # Log store event to live visualizer
        try:
            log_pipeline_event("store", {
                "conversation_id": conversation_id,
                "turn_id": turn_id,
                "summary": summary,
                "user_text": user_text or text,
                "assistant_text": assistant_text or "",
                "message": f"Stored Turn {turn_id} and generated 384-dimensional dense representation."
            })
        except Exception:
            pass

    def retrieve(self, conversation_id: str, query: str, k: int = 10, 
                 time_window: Optional[int] = None) -> List[TurnMemory]:
        """
        Queries the TurboVec index for the top-k most semantically relevant turns,
        optionally filtered by a temporal window.
        """
        if self.index is None:
            return []
            
        # 1. Embed the search query
        query_emb = self.embed_text(query)
        
        # 2. Build allowlist of all turns in this conversation (and apply time window if requested)
        with self.store._lock:
            conn = self.store._get_connection()
            cursor = conn.cursor()
            if time_window is not None:
                cursor.execute("""
                    SELECT turn_id FROM turns
                    WHERE conversation_id = ?
                    ORDER BY turn_id DESC LIMIT ?
                """, (conversation_id, time_window))
            else:
                cursor.execute("""
                    SELECT turn_id FROM turns
                    WHERE conversation_id = ?
                """, (conversation_id,))
            rows = cursor.fetchall()
            conn.close()
            
        if not rows:
            return []
            
        recent_ids = [self.encode_id(conversation_id, row["turn_id"]) for row in rows]
        # Filter to only keep IDs that are present in the TurboVec index to avoid KeyError crashes
        recent_ids = [rid for rid in recent_ids if rid in self.index]
        if not recent_ids:
            return []
        allowlist = np.array(recent_ids, dtype=np.uint64)
                
        # 3. Perform TurboVec search using allowlist to restrict search to this conversation only
        with GPULockManager():
            scores, ids = self.index.search(query_emb.reshape(1, -1), k=k, allowlist=allowlist)
                
        # 4. Process search results and fetch records from SQLite
        scores = scores[0]
        ids = ids[0]
        
        valid_indices = [i for i, tid in enumerate(ids) if tid != 18446744073709551615 and tid != 0]
        
        retrieved_memories = []
        for idx in valid_indices:
            tid = int(ids[idx])
            score = float(scores[idx])
            
            # Decode turn_id from index_id
            turn_id = tid & 0xFFFF
            
            # Fetch summary, texts, and embedding from SQLite
            with self.store._lock:
                conn = self.store._get_connection()
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT summary, user_text, assistant_text FROM turns
                    WHERE conversation_id = ? AND turn_id = ?
                """, (conversation_id, turn_id))
                turn_row = cursor.fetchone()
                
                cursor.execute("""
                    SELECT data, dim FROM turn_embeddings
                    WHERE conversation_id = ? AND turn_id = ?
                """, (conversation_id, turn_id))
                emb_row = cursor.fetchone()
                conn.close()
                
            if turn_row is not None:
                summary = turn_row["summary"] or ""
                user_text = turn_row["user_text"] or ""
                assistant_text = turn_row["assistant_text"] or ""
                
                embedding = None
                if emb_row is not None:
                    embedding = np.frombuffer(emb_row["data"], dtype=np.float32).reshape((emb_row["dim"],))
                
                retrieved_memories.append(TurnMemory(
                    turn_id=turn_id,
                    summary=summary,
                    user_text=user_text,
                    assistant_text=assistant_text,
                    score=score,
                    embedding=embedding
                ))

        # Log retrieve event to live visualizer
        try:
            log_pipeline_event("retrieve", {
                "conversation_id": conversation_id,
                "query": query,
                "k": k,
                "results": [{"turn_id": m.turn_id, "score": m.score, "summary": m.summary} for m in retrieved_memories],
                "message": f"Retrieved {len(retrieved_memories)} memories for query: '{query}'"
            })
        except Exception:
            pass
                
        return retrieved_memories
