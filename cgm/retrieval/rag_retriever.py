import os
import time
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from cgm.database.schema import SQLiteGraphStore, HybridMemoryObject
from cgm.safety.safety import GPULockManager
from cgm.visualization.visualize import log_pipeline_event
from turbovec import IdMapIndex

class BM25Retriever:
    """
    Lightweight, zero-dependency BM25 sparse keyword retriever.
    """
    def __init__(self, b: float = 0.75, k1: float = 1.5):
        self.b = b
        self.k1 = k1
        self.corpus_size = 0
        self.avg_doc_len = 0.0
        self.docs = []  # list of dict: {'turn_id', 'tokens', 'counts', 'len'}
        self.idf = {}

    def tokenize(self, text: str) -> List[str]:
        import re
        text = text.lower()
        return re.findall(r'\b\w+\b', text)

    def fit(self, doc_list: List[Dict[str, Any]]):
        self.docs = []
        self.corpus_size = len(doc_list)
        if self.corpus_size == 0:
            return
        
        total_len = 0
        df = {}
        for item in doc_list:
            tokens = self.tokenize(item['text'])
            token_counts = {}
            for t in tokens:
                token_counts[t] = token_counts.get(t, 0) + 1
            self.docs.append({
                'turn_id': item['turn_id'],
                'tokens': tokens,
                'counts': token_counts,
                'len': len(tokens)
            })
            total_len += len(tokens)
            
            for t in set(tokens):
                df[t] = df.get(t, 0) + 1
                
        self.avg_doc_len = total_len / self.corpus_size
        
        import math
        for token, freq in df.items():
            self.idf[token] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)

    def score(self, query: str) -> List[Tuple[int, float]]:
        query_tokens = self.tokenize(query)
        scores = []
        for doc in self.docs:
            score = 0.0
            for token in query_tokens:
                if token in doc['counts']:
                    tf = doc['counts'][token]
                    idf = self.idf.get(token, 0.0)
                    num = tf * (self.k1 + 1.0)
                    den = tf + self.k1 * (1.0 - self.b + self.b * (doc['len'] / (self.avg_doc_len + 1e-8)))
                    score += idf * (num / den)
            scores.append((doc['turn_id'], score))
        return sorted(scores, key=lambda x: x[1], reverse=True)

@dataclass
class TurnMemory:
    turn_id: int
    summary: str
    user_text: str
    assistant_text: str
    score: float
    conversation_id: str = "default_conv"
    embedding: Optional[np.ndarray] = None
    dia_id: Optional[str] = None

class RAGRetriever:
    """
    Hybrid dense-sparse retrieval system backed by TurboVec, BM25, and SQLite.
    Performs reciprocal rank fusion (RRF) and Cross-Encoder reranking.
    """
    def __init__(self, store: SQLiteGraphStore, device: Optional[str] = None, 
                 embedding_model_name: str = "all-MiniLM-L6-v2", 
                 index_path: str = "data/cgm_rag.tvim"):
        self.store = store
        self.device = device
        self.embedding_model_name = embedding_model_name
        self.index_path = index_path
        self._embedder = None
        self._cross_encoder = None
        self.index = None
        
        self._embedding_cache: Dict[str, np.ndarray] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        self._init_index()

    def _init_index(self):
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
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            import torch
            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[RAGRetriever] Loading SentenceTransformer '{self.embedding_model_name}' on device: {self.device}...")
            with GPULockManager():
                self._embedder = SentenceTransformer(self.embedding_model_name, device=self.device)
        return self._embedder

    @property
    def cross_encoder(self):
        if self._cross_encoder is None:
            from sentence_transformers import CrossEncoder
            import torch
            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[RAGRetriever] Loading Cross-Encoder 'cross-encoder/ms-marco-MiniLM-L-6-v2' on device: {self.device}...")
            with GPULockManager():
                self._cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=self.device)
        return self._cross_encoder

    def embed_text(self, text: str) -> np.ndarray:
        if text in self._embedding_cache:
            self._cache_hits += 1
            return self._embedding_cache[text]
        self._cache_misses += 1
        with GPULockManager():
            embedding = self.embedder.encode(text, convert_to_numpy=True)
        self._embedding_cache[text] = embedding
        return embedding

    def get_cache_stats(self) -> Dict[str, int]:
        total = self._cache_hits + self._cache_misses
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "total": total,
            "hit_rate": (self._cache_hits / total * 100) if total > 0 else 0.0,
            "cache_size": len(self._embedding_cache),
        }

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        with GPULockManager():
            embeddings = self.embedder.encode(texts, convert_to_numpy=True)
        return embeddings

    def persist(self):
        if self.index is not None and self.index_path:
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            with GPULockManager():
                self.index.write(self.index_path)
            print(f"[RAGRetriever] TurboVec index successfully persisted to: {self.index_path}")

    def reset_index(self):
        """Reset the persistent TurboVec index so a conversation can be re-seeded cleanly."""
        if self.index_path and os.path.exists(self.index_path):
            os.remove(self.index_path)
        with GPULockManager():
            self.index = IdMapIndex(dim=384, bit_width=4)
        self._embedding_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    def encode_id(self, conversation_id: str, turn_id: int) -> int:
        import hashlib
        h = hashlib.sha256(conversation_id.encode('utf-8')).digest()
        conv_hash = int.from_bytes(h[:6], byteorder='big')
        return (conv_hash << 16) | (turn_id & 0xFFFF)

    def store_turn(self, conversation_id: str, turn_id: int, text: str, 
                   summary: Optional[str] = None, user_text: Optional[str] = None, 
                   assistant_text: Optional[str] = None, tenant_id: str = "default_tenant", user_id: str = "default_user",
                   dia_id: Optional[str] = None):
        embedding = self.embed_text(text)
        index_id = self.encode_id(conversation_id, turn_id)
        with GPULockManager():
            self.index.add_with_ids(embedding.reshape(1, -1), np.array([index_id], dtype=np.uint64))
            
        self.store.save_turn_embedding(conversation_id, turn_id, embedding, tenant_id=tenant_id, user_id=user_id)
        
        if summary is None:
            summary = text[:250] + "..." if len(text) > 250 else text
            
        hmo = HybridMemoryObject(
            turn_id=turn_id,
            timestamp=time.time(),
            summary=summary,
            user_text=user_text or text,
            assistant_text=assistant_text or "",
            raw_embedding=embedding,
            dia_id=dia_id
        )
        self.store.save_hmo(conversation_id, hmo, tenant_id=tenant_id, user_id=user_id)
        self.persist()

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
                 time_window: Optional[int] = None, tenant_id: str = "default_tenant", user_id: str = "default_user") -> List[TurnMemory]:
        if self.index is None:
            return []
            
        # 1. Fetch all turns in SQLite for this conversation and tenant/user
        with self.store._lock:
            conn = self.store._get_connection()
            cursor = conn.cursor()
            if time_window is not None:
                cursor.execute("""
                    SELECT turn_id, user_text, assistant_text, summary, dia_id FROM turns
                    WHERE conversation_id = ? AND tenant_id = ? AND user_id = ?
                    ORDER BY turn_id DESC LIMIT ?
                """, (conversation_id, tenant_id, user_id, time_window))
            else:
                cursor.execute("""
                    SELECT turn_id, user_text, assistant_text, summary, dia_id FROM turns
                    WHERE conversation_id = ? AND tenant_id = ? AND user_id = ?
                """, (conversation_id, tenant_id, user_id))
            all_turns = cursor.fetchall()
            conn.close()
            
        if not all_turns:
            return []

        # 2. Sparse Retrieval (BM25)
        docs_bm25 = []
        turn_dict = {}
        for row in all_turns:
            t_id = row["turn_id"]
            u_txt = row["user_text"] or ""
            a_txt = row["assistant_text"] or ""
            summary = row["summary"] or ""
            text_full = f"{u_txt} {a_txt} {summary}"
            docs_bm25.append({'turn_id': t_id, 'text': text_full})
            turn_dict[t_id] = row
            
        bm25 = BM25Retriever()
        bm25.fit(docs_bm25)
        sparse_scores = bm25.score(query)
        
        # 3. Dense Retrieval (TurboVec)
        query_emb = self.embed_text(query)
        recent_ids = [self.encode_id(conversation_id, row["turn_id"]) for row in all_turns]
        recent_ids = [rid for rid in recent_ids if rid in self.index]
        
        dense_scores_dict = {}
        if recent_ids:
            allowlist = np.array(recent_ids, dtype=np.uint64)
            with GPULockManager():
                scores, ids = self.index.search(query_emb.reshape(1, -1), k=len(recent_ids), allowlist=allowlist)
            scores = scores[0]
            ids = ids[0]
            for idx, tid in enumerate(ids):
                if tid != 18446744073709551615 and tid != 0:
                    turn_id = int(tid) & 0xFFFF
                    dense_scores_dict[turn_id] = float(scores[idx])

        # Entity-Graph keyword overlap retrieval
        import re
        stop_words = {"what", "is", "the", "of", "in", "on", "to", "for", "a", "an", "and", "or", "but", "with", "at", "by", "from", "when", "where", "who", "how", "why", "did", "does", "do", "was", "were", "go", "went", "has", "have", "had", "been", "about"}
        query_words = set(re.findall(r'\b\w+\b', query.lower())) - stop_words
        
        graph_turns = set()
        if query_words:
            with self.store._lock:
                conn = self.store._get_connection()
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT turn_id, subject, object FROM triples
                    WHERE conversation_id = ? AND tenant_id = ? AND user_id = ? AND valid_until IS NULL AND is_negated = 0
                """, (conversation_id, tenant_id, user_id))
                triples_rows = cursor.fetchall()
                conn.close()
                
            for row in triples_rows:
                t_id = row["turn_id"]
                subj_words = set(re.findall(r'\b\w+\b', (row["subject"] or "").lower()))
                obj_words = set(re.findall(r'\b\w+\b', (row["object"] or "").lower()))
                if (subj_words & query_words) or (obj_words & query_words):
                    graph_turns.add(t_id)

        # 4. Reciprocal Rank Fusion (RRF) with Entity-Graph Boost
        sparse_rank = {turn_id: rank for rank, (turn_id, _) in enumerate(sparse_scores)}
        sorted_dense = sorted(dense_scores_dict.items(), key=lambda x: x[1], reverse=True)
        dense_rank = {turn_id: rank for rank, (turn_id, _) in enumerate(sorted_dense)}
        
        rrf_scores = {}
        all_turn_ids = (set(sparse_rank.keys()) | set(dense_rank.keys()) | graph_turns) & set(turn_dict.keys())
        for turn_id in all_turn_ids:
            rank_sparse = sparse_rank.get(turn_id, 9999)
            rank_dense = dense_rank.get(turn_id, 9999)
            base_score = (1.0 / (60.0 + rank_sparse)) + (1.0 / (60.0 + rank_dense))
            if turn_id in graph_turns:
                base_score += 0.5  # Substantial boost for direct entity-graph matches
            rrf_scores[turn_id] = base_score
            
        sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        candidate_ids = [turn_id for turn_id, _ in sorted_rrf[:max(20, k * 2)]]

        # 5. Build Candidates and Run Cross-Encoder Reranking
        candidates = []
        for turn_id in candidate_ids:
            row = turn_dict.get(turn_id)
            if row is not None:
                u_txt = row["user_text"] or ""
                a_txt = row["assistant_text"] or ""
                summary = row["summary"] or ""
                
                try:
                    embedding = self.store.get_turn_embedding(conversation_id, turn_id, tenant_id=tenant_id, user_id=user_id)
                except Exception:
                    embedding = None
                    
                initial_score = rrf_scores.get(turn_id, 0.0)
                
                candidates.append(TurnMemory(
                    turn_id=turn_id,
                    summary=summary,
                    user_text=u_txt,
                    assistant_text=a_txt,
                    score=initial_score,
                    conversation_id=conversation_id,
                    embedding=embedding,
                    dia_id=row["dia_id"]
                ))
                
        if not candidates:
            return []
            
        # Rerank with Cross-Encoder
        pairs = [(query, f"{cand.user_text} {cand.assistant_text} {cand.summary}") for cand in candidates]
        try:
            with GPULockManager():
                ce_scores = self.cross_encoder.predict(pairs)
            # Normalize raw CE logits to [0,1] using min-max over this batch.
            ce_arr = np.array(ce_scores, dtype=np.float32)
            ce_min, ce_max = ce_arr.min(), ce_arr.max()
            ce_range = ce_max - ce_min
            if ce_range > 1e-6:
                normalized = (ce_arr - ce_min) / ce_range
            else:
                normalized = np.ones_like(ce_arr) * 0.5
            for idx, cand in enumerate(candidates):
                cand.score = float(normalized[idx])
            candidates = sorted(candidates, key=lambda x: x.score, reverse=True)
        except Exception as e:
            print(f"[RAGRetriever] Warning: Cross-Encoder reranking failed: {e}. Falling back to RRF rankings.")
            pass
            
        retrieved_memories = candidates[:k]

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
