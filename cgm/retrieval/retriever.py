import numpy as np
from typing import Dict, Any, List
from cgm.database.schema import SQLiteGraphStore
from cgm.safety.safety import GPULockManager

class SubgraphRetriever:
    """
    Retrieves the most semantically relevant and temporally recent portions of the
    Conversational Graph Memory to construct the injection prompt.
    """
    def __init__(self, store: SQLiteGraphStore, embedding_model_name: str = "all-MiniLM-L6-v2", device: str = None):
        self.store = store
        self.device = device
        self._embedder = None
        self.embedding_model_name = embedding_model_name

    @property
    def embedder(self):
        """
        Lazy loader for sentence-transformers to avoid overhead during setup.
        Loads model securely using GPULockManager.
        """
        if self._embedder is None:
            # We defer import until needed
            from sentence_transformers import SentenceTransformer
            import torch
            
            # Autodetect CUDA RTX A2000 if not specified
            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                
            print(f"[Retriever] Loading SentenceTransformer '{self.embedding_model_name}' on device: {self.device}...")
            
            # Load the model directly (the caller embed_text already holds the GPULockManager context)
            self._embedder = SentenceTransformer(self.embedding_model_name, device=self.device)
                
        return self._embedder

    def embed_text(self, text: str) -> np.ndarray:
        """
        Computes the dense vector representation of a given string.
        """
        with GPULockManager():
            embedding = self.embedder.encode(text, convert_to_numpy=True)
        return embedding

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Computes the dense vector representations of a list of strings in a single batch.
        """
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        with GPULockManager():
            embeddings = self.embedder.encode(texts, convert_to_numpy=True)
        return embeddings

    def retrieve(self, conversation_id: str, query_text: str, k: int = 20, 
                 alpha: float = 0.5, beta: float = 0.3, gamma: float = 0.2, 
                 decay_rate: float = 0.1) -> Dict[str, Any]:
        """
        Performs a multi-signal scoring procedure to retrieve the top-k semantic elements.
        Scoring formula:
          score(triple) = alpha * similarity(query, triple_text) 
                        + beta * similarity(query, turn_summary)
                        + gamma * exp(-decay_rate * (current_turn - turn_id))
        """
        graph_data = self.store.get_conversation_graph(conversation_id)
        triples = graph_data["triples"]
        turns = graph_data["turns"]
        entities = graph_data["entities"]
        
        if not triples:
            return {"triples": [], "summary": "", "entities": []}

        # 1. Embed query
        query_embed = self.embed_text(query_text)
        
        # 2. Get summaries and their turn IDs
        current_turn = max(t["turn_id"] for t in turns) if turns else 1
        
        # Build turn summaries lookup & batch embed summaries
        turn_summaries = {t["turn_id"]: t["summary"] for t in turns}
        turn_ids_list = list(turn_summaries.keys())
        summaries_list = [turn_summaries[tid] for tid in turn_ids_list]
        
        summary_embeddings = self.embed_texts(summaries_list) # shape: (num_turns, 384)
        turn_embeddings = {tid: summary_embeddings[idx] for idx, tid in enumerate(turn_ids_list)}

        # 3. Score each triple
        # Collect all triple texts for batch embedding
        trip_texts = [f"{trip['subject']} {trip['predicate']} {trip['object']}" for trip in triples]
        trip_embeddings = self.embed_texts(trip_texts) # shape: (num_triples, 384)
        
        scored_triples = []
        for idx, trip in enumerate(triples):
            turn_id = trip["turn_id"]
            trip_embed = trip_embeddings[idx]
            
            # Calculate cosine similarities
            # similarity = (A . B) / (||A|| * ||B||)
            sim_query_trip = np.dot(query_embed, trip_embed) / (np.linalg.norm(query_embed) * np.linalg.norm(trip_embed) + 1e-9)
            
            summary_embed = turn_embeddings.get(turn_id)
            if summary_embed is not None:
                sim_query_summary = np.dot(query_embed, summary_embed) / (np.linalg.norm(query_embed) * np.linalg.norm(summary_embed) + 1e-9)
            else:
                sim_query_summary = 0.0
                
            # Recency decay
            recency = np.exp(-decay_rate * (current_turn - turn_id))
            
            # Combined score
            score = (alpha * sim_query_trip) + (beta * sim_query_summary) + (gamma * recency)
            
            scored_triples.append((score, trip))

        # Sort by score descending and take top k
        scored_triples.sort(key=lambda x: x[0], reverse=True)
        top_k_triples = [item[1] for item in scored_triples[:k]]
        
        # 4. Filter entities belonging to the retrieved triples
        retrieved_entity_names = set()
        for trip in top_k_triples:
            retrieved_entity_names.add(trip["subject"])
            retrieved_entity_names.add(trip["object"])
            
        top_entities = [ent for ent in entities if ent["name"] in retrieved_entity_names]
        
        # 5. Extract summaries of the turns associated with retrieved triples
        retrieved_turn_ids = {t["turn_id"] for t in top_k_triples}
        associated_summaries = [turn_summaries[tid] for tid in sorted(retrieved_turn_ids) if tid in turn_summaries]
        combined_summary = " ".join(associated_summaries)
        
        return {
            "triples": [[t["subject"], t["predicate"], t["object"]] for t in top_k_triples],
            "entities": top_entities,
            "summary": combined_summary
        }
