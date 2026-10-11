"""Offline regression checks for the core store/retriever/model. Run: python tests/test_core.py"""
import os
import sys
import tempfile
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cgm.core.compressor import rotary_cos_sin
from cgm.core.model import MemoryEncoderNetwork
from cgm.database.schema import HybridMemoryObject, SQLiteGraphStore
from cgm.retrieval.rag_retriever import RAGRetriever


def fake_embed(text):
    rng = np.random.default_rng(abs(hash(text)) % 2**32)
    v = rng.standard_normal(384).astype(np.float32)
    return v / np.linalg.norm(v)


def new_store():
    return SQLiteGraphStore(os.path.join(tempfile.mkdtemp(), "m.db"))


def active(store):
    return {(t["predicate"], t["object"]) for t in store.get_conversation_graph("c")["triples"]}


def add(store, turn, *triples):
    store.save_hmo("c", HybridMemoryObject(turn_id=turn, timestamp=time.time() + turn, summary="", triples=list(triples)))


def test_triple_supersession():
    s = new_store()
    add(s, 1, ["db", "is", "MySQL"])
    add(s, 2, ["db", "is", "Postgres"])
    assert active(s) == {("is", "Postgres")}, active(s)
    add(s, 3, ["db", "NOT_is", "Postgres"])
    assert active(s) == {("NOT_is", "Postgres")}, active(s)
    add(s, 4, ["db", "is", "Postgres"])  # re-affirm after negation
    assert active(s) == {("is", "Postgres")}, active(s)
    add(s, 5, ["db", "NOT_is", "Redis"])  # negating another value keeps the current one
    assert active(s) == {("is", "Postgres"), ("NOT_is", "Redis")}, active(s)


def make_retriever(store):
    r = RAGRetriever(store, device="cpu")
    r.embed_text = fake_embed
    return r


def test_index_isolated_and_rebuilt():
    s1, s2 = new_store(), new_store()
    r1 = make_retriever(s1)
    r1.store_turn("c", 1, "we use postgres")
    r1.store_turn("c", 1, "we use postgres, really")  # re-storing a turn must not raise
    make_retriever(s2).store_turn("c", 1, "other db")  # same ids, different store: no collision
    r1b = make_retriever(s1)  # fresh process: index rebuilt from SQLite
    assert r1b.index.contains(r1b.encode_id("c", 1))
    assert not r1b.index.contains(r1b.encode_id("c", 1, user_id="someone_else"))


def test_rotary_absent_for_gpt2_like():
    class NoModel(torch.nn.Module):
        pass
    assert rotary_cos_sin(NoModel(), 4, 8, "cpu") == (None, None)


def test_men_shapes():
    men = MemoryEncoderNetwork(input_dim=768, num_layers=2, num_heads=2, head_dim=8, rank=4, tokens_per_triple=3)
    kv = men(torch.randn(1, 5, 768))
    assert len(kv) == 2 and kv[0][0].shape == (1, 2, 15, 8) and kv[0][1].shape == (1, 2, 15, 8)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
