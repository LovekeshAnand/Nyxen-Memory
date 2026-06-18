#!/usr/bin/env python3
"""
LongMemEval benchmark runner — generates hypothesis JSONL for official scoring.

Usage:
    python run_longmem_benchmark.py --dataset data/longmemeval_oracle.json --mode cgm --limit 10
    python run_longmem_benchmark.py --dataset data/longmemeval_oracle.json --mode rag --output data/longmem/hypotheses.jsonl
"""
import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cgm.core.pipeline import CGMPipeline
from cgm.eval.longmem_load import load_all_instances
from cgm.eval.longmem_ingest import seed_haystack
from cgm.eval.prompts import build_user_prompt, build_locomo_prompt


def generate_hypothesis(
    pipeline: CGMPipeline,
    instance: Dict[str, Any],
    mode: str,
    k: int = 10,
    max_new_tokens: int = 64,
) -> str:
    cid = instance["question_id"]
    question = instance["question"]

    if mode == "cgm":
        return pipeline.generate(
            cid, question, k=k, max_new_tokens=max_new_tokens,
            mode="inject", do_sample=False,
        )
    elif mode == "rag":
        memories = pipeline.retriever.retrieve(cid, question, k=k)
        context_parts = [m.assistant_text or m.user_text or "" for m in memories if m.assistant_text or m.user_text]
        context = "Relevant history:\n" + "\n".join(context_parts)
        prompt = build_locomo_prompt(question, context)
        inputs = pipeline.tokenizer(prompt, return_tensors="pt").to(pipeline.device)
        with torch.no_grad():
            outputs = pipeline.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return pipeline.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        ).strip()
    raise ValueError(f"Unknown mode: {mode}")


def run_benchmark(args) -> Dict[str, Any]:
    if not os.path.exists(args.dataset):
        print(f"Error: Dataset not found: {args.dataset}")
        print("Download oracle: curl -L -o data/longmemeval_oracle.json "
              "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json")
        sys.exit(1)

    instances = load_all_instances(args.dataset)
    if args.limit:
        instances = instances[: args.limit]

    pipeline = CGMPipeline(model_name=args.model, rank=args.rank)
    pipeline.initialize()
    pipeline.mem_guard.enforce_safety = lambda *a, **kw: None

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    results = []
    resume_ids = set()

    if args.resume and os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                resume_ids.add(row["question_id"])
                results.append(row)

    for idx, instance in enumerate(instances):
        qid = instance["question_id"]
        if qid in resume_ids:
            continue

        print(f"\n[LongMemEval] ({idx+1}/{len(instances)}) {qid} ({instance['question_type']})")
        seed_haystack(pipeline, instance)

        t0 = time.time()
        hypothesis = generate_hypothesis(pipeline, instance, args.mode, k=args.k, max_new_tokens=args.max_new_tokens)
        latency = time.time() - t0

        row = {
            "question_id": qid,
            "question_type": instance["question_type"],
            "question": instance["question"],
            "answer": instance["answer"],
            "hypothesis": hypothesis,
            "mode": args.mode,
            "latency": latency,
            "is_abstention": instance["is_abstention"],
        }
        results.append(row)

        with open(args.output, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        # Simple inline scoring for non-abstention
        if not instance["is_abstention"] and instance["answer"]:
            hit = instance["answer"].lower() in hypothesis.lower()
            print(f"  Answer: {hypothesis[:80]}... -> {'HIT' if hit else 'MISS'} ({latency:.2f}s)")

    # Summary
    non_abs = [r for r in results if not r.get("is_abstention")]
    hits = sum(1 for r in non_abs if r.get("answer", "").lower() in r.get("hypothesis", "").lower())
    acc = 100 * hits / len(non_abs) if non_abs else 0

    report = {
        "mode": args.mode,
        "num_questions": len(results),
        "substring_accuracy": round(acc, 2),
        "output": args.output,
    }
    report_path = args.output.replace(".jsonl", "_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n[LongMemEval] Substring accuracy: {acc:.1f}% ({hits}/{len(non_abs)})")
    print(f"Hypotheses: {args.output}")
    print(f"Score with official eval: clone LongMemEval repo and run evaluate_qa.py on {args.output}")
    return report


def main():
    parser = argparse.ArgumentParser(description="LongMemEval benchmark runner")
    parser.add_argument("--dataset", default="data/longmemeval_oracle.json")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--mode", default="cgm", choices=["cgm", "rag"])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default="data/longmem/hypotheses.jsonl")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
