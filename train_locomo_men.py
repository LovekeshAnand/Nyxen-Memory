#!/usr/bin/env python3
"""Train a reusable MEN checkpoint on LoCoMo-derived QA samples."""

import argparse
import json
import os
import numpy as np
import random
import sys

root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from cgm.core.pipeline import CGMPipeline
from cgm.eval.locomo_ingest import seed_conversation
from cgm.eval.locomo_load import filter_qa, load_all_conversations
from cgm.eval.prompts import SYSTEM_PREFIX, wrap_training_prompt
from cgm.training.train import MEGATrainer
from cgm.training.train_data import TripleTrainingSample, encode_triple
from cgm_locomo_eval import seed_locomo_database, map_qa_to_samples


def _evidence_session(evidence):
    for item in evidence or []:
        if ":" not in item:
            continue
        session = item.split(":", 1)[0]
        if session.startswith("D"):
            try:
                return int(session[1:])
            except ValueError:
                return None
    return None


def _temporal_query(question: str) -> bool:
    query = question.lower()
    return any(token in query for token in ("when", "date", "day", "month", "year"))


def _score_triple(question_embedding, triple, embed_fn, temporal: bool) -> float:
    triple_text = f"{triple[0]} {triple[1]} {triple[2]}"
    triple_embedding = embed_fn(triple_text)
    score = float(
        np.dot(question_embedding, triple_embedding)
        / (np.linalg.norm(question_embedding) * np.linalg.norm(triple_embedding) + 1e-8)
    )
    if temporal:
        triple_lower = triple_text.lower()
        if any(token in triple_lower for token in ("date", "year", "month", "day", "time", "yesterday", "today", "last", "before", "after")):
            score += 0.25
        if any(ch.isdigit() for ch in triple_lower):
            score += 0.15
    return score


def map_parsed_qa_to_samples(pipeline, conv, conversation_id: str, categories):
    with pipeline.store._lock:
        conn = pipeline.store._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT turn_id, subject, predicate, object FROM triples WHERE conversation_id = ?",
            (conversation_id,),
        )
        rows = cursor.fetchall()
        conn.close()

    all_triples = [[row["subject"], row["predicate"], row["object"]] for row in rows]
    triples_by_session = {}
    for row in rows:
        session_idx = int(row["turn_id"]) // 3600 if int(row["turn_id"]) >= 3600 else None
        triples_by_session.setdefault(session_idx, []).append([row["subject"], row["predicate"], row["object"]])
        triples_by_session.setdefault(row["turn_id"], []).append([row["subject"], row["predicate"], row["object"]])

    turn_to_session = {turn["turn_id"]: turn["session_idx"] for turn in conv["turns"]}
    triples_by_session = {}
    for row in rows:
        turn_id = int(row["turn_id"])
        session_idx = turn_to_session.get(turn_id)
        if session_idx is not None:
            triples_by_session.setdefault(session_idx, []).append([row["subject"], row["predicate"], row["object"]])

    samples = []
    for qa in filter_qa(conv["qa"], categories):
        answer = qa.get("answer") or qa.get("adversarial_answer")
        if answer is None:
            continue
        evidence_session = _evidence_session(qa.get("evidence", []))
        candidates = []
        if evidence_session is not None:
            for session_candidate in (evidence_session, evidence_session + 1, evidence_session - 1):
                candidates.extend(triples_by_session.get(session_candidate, []))
        if not candidates:
            candidates = all_triples
        if not candidates:
            continue

        question_embedding = pipeline.retriever.embed_text(qa["question"])
        temporal = _temporal_query(qa["question"])
        best_triple = max(
            candidates,
            key=lambda triple: _score_triple(question_embedding, triple, pipeline.retriever.embed_text, temporal),
        )
        x_vec = encode_triple(best_triple, pipeline.retriever.embed_text, all_triples=all_triples)
        samples.append(TripleTrainingSample(
            triple=best_triple,
            x_vector=x_vec,
            prompt_text=wrap_training_prompt(f"User: {qa['question']}\nAssistant:", SYSTEM_PREFIX),
            target_text=str(answer).strip(),
            triple_text=f"{best_triple[0]} {best_triple[1]} {best_triple[2]}",
        ))
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MEN on LoCoMo graph/question pairs.")
    parser.add_argument("--dataset", default="data/locomo10.json")
    parser.add_argument("--graph-path", default="data/locomo_extracted_graph.json")
    parser.add_argument("--use-extracted-graph", action="store_true")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--distill-lambda", type=float, default=0.5)
    parser.add_argument("--checkpoint", default="data/checkpoints/men_state_dict.pt")
    parser.add_argument("--resume-checkpoint", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-conversations", type=int, default=1)
    parser.add_argument("--categories", default="1,2,3,4")
    args = parser.parse_args()

    load_checkpoint = args.checkpoint if args.resume_checkpoint else None
    pipeline = CGMPipeline(model_name=args.model, rank=args.rank, men_checkpoint=load_checkpoint)
    pipeline.initialize()
    pipeline.mem_guard.enforce_safety = lambda *a, **kw: None

    if args.use_extracted_graph:
        if not os.path.exists(args.graph_path):
            raise FileNotFoundError(f"Graph JSON not found: {args.graph_path}")
        with open(args.graph_path, "r", encoding="utf-8") as handle:
            graph_data = json.load(handle)
        conversation_id = "locomo_conv_0"
        seed_locomo_database(pipeline, graph_data, conversation_id)
        samples = map_qa_to_samples(pipeline, graph_data, conversation_id)
    else:
        categories = [int(category) for category in args.categories.split(",")]
        conversations = load_all_conversations(args.dataset)
        if args.limit_conversations:
            conversations = conversations[: args.limit_conversations]
        samples = []
        for conv in conversations:
            conversation_id = conv["sample_id"]
            seed_conversation(pipeline, conv, conversation_id=conversation_id)
            samples.extend(map_parsed_qa_to_samples(pipeline, conv, conversation_id, categories))

    random.seed(args.seed)
    random.shuffle(samples)
    split_index = max(1, int(len(samples) * 0.7))
    train_samples = samples[:split_index]
    eval_samples = samples[split_index:] or samples[:1]

    trainer = MEGATrainer(pipeline, lr=args.lr, distill_lambda=args.distill_lambda)
    trainer.default_checkpoint_path = args.checkpoint
    trainer.mem_guard.enforce_safety = lambda *a, **kw: None
    trainer.fit(
        train_samples=train_samples,
        eval_samples=eval_samples,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
    )

    print(f"\nSaved checkpoint: {args.checkpoint}")


if __name__ == "__main__":
    main()
