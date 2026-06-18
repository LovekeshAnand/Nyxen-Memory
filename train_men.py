#!/usr/bin/env python3
"""Offline MEN training entrypoint for reusable CGM checkpoints."""

import argparse
import os
import sys

root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from cgm.core.pipeline import CGMPipeline
from cgm.training.train import MEGATrainer
from cgm.training.train_data import TRIPLE_QA_MAP, build_training_samples, split_train_eval


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MEN once and save a reusable checkpoint.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--db-path", default="data/cgm_memory.db")
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--distill-lambda", type=float, default=2.0)
    parser.add_argument("--checkpoint", default="data/checkpoints/men_state_dict.pt")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    pipeline = CGMPipeline(model_name=args.model, db_path=args.db_path, rank=args.rank, men_checkpoint=args.checkpoint)
    pipeline.initialize()
    pipeline.mem_guard.enforce_safety = lambda *a, **kw: None

    triples = [key.split("|") for key in TRIPLE_QA_MAP.keys()]

    samples = build_training_samples(triples, pipeline.retriever.embed_text)
    train_samples, eval_samples = split_train_eval(samples)

    if args.max_samples:
        train_samples = train_samples[: args.max_samples]
        eval_samples = eval_samples[: max(1, min(len(eval_samples), args.max_samples // 4 or 1))]

    trainer = MEGATrainer(pipeline, lr=args.lr, distill_lambda=args.distill_lambda)
    trainer.fit(
        train_samples=train_samples,
        eval_samples=eval_samples,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
    )

    print(f"\nCheckpoint ready: {args.checkpoint}")


if __name__ == "__main__":
    main()
