#!/usr/bin/env python3
"""
Full LoCoMo benchmark runner — official F1 scoring, all 10 conversations, 1540 scored questions.

Usage:
    python run_locomo_benchmark.py --dataset data/locomo10.json --modes rag,cgm
    python run_locomo_benchmark.py --dataset data/locomo10.json --limit-conversations 1 --max-questions 20
"""
import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cgm.core.pipeline import CGMPipeline
from cgm.eval.locomo_load import load_all_conversations, filter_qa
from cgm.eval.locomo_ingest import seed_conversation, load_extracted_graph
from cgm.eval.locomo_score import score_qa_item, aggregate_results, CATEGORY_NAMES
from cgm.eval.prompts import build_locomo_prompt, build_user_prompt


def generate_answer(
    pipeline: CGMPipeline,
    conversation_id: str,
    question: str,
    mode: str,
    k: int = 10,
    max_new_tokens: int = 64,
    full_dialogue: str = "",
) -> str:
    """Generate an answer for one LoCoMo question."""
    if mode == "cgm":
        prompt = build_user_prompt(question)
        return pipeline.generate(
            conversation_id, question, k=k, max_new_tokens=max_new_tokens,
            mode="inject", do_sample=False, persist_response=False,
        )
    elif mode == "rag":
        memories = pipeline.retriever.retrieve(conversation_id, question, k=k)
        context_parts = []
        for mem in sorted(memories, key=lambda m: m.turn_id):
            text = mem.assistant_text or mem.user_text or ""
            if text:
                context_parts.append(text)
        context = "Relevant conversation history:\n" + "\n".join(context_parts)
        prompt = build_locomo_prompt(question, context)
        inputs = pipeline.tokenizer(prompt, return_tensors="pt").to(pipeline.device)
        with torch.no_grad():
            outputs = pipeline.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return pipeline.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        ).strip()
    elif mode == "text":
        context = f"Full conversation history:\n{full_dialogue}" if full_dialogue else ""
        prompt = build_locomo_prompt(question, context)
        inputs = pipeline.tokenizer(prompt, return_tensors="pt").to(pipeline.device)
        try:
            with torch.no_grad():
                outputs = pipeline.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            return pipeline.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
            ).strip()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                return ""
            raise
    else:
        raise ValueError(f"Unknown mode: {mode}")


def run_benchmark(args) -> Dict[str, Any]:
    pipeline = CGMPipeline(model_name=args.model, rank=args.rank, men_checkpoint=args.men_checkpoint)
    pipeline.initialize()
    pipeline.mem_guard.enforce_safety = lambda *a, **kw: None

    if args.extracted_graph:
        print(f"[LoCoMo] Loading pre-extracted graph: {args.extracted_graph}")
        conversations = None
        graph_mode = True
    else:
        if not os.path.exists(args.dataset):
            print(f"Error: Dataset not found: {args.dataset}")
            print("Download: curl -L -o data/locomo10.json "
                  "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json")
            sys.exit(1)
        conversations = load_all_conversations(args.dataset)
        if args.limit_conversations:
            conversations = conversations[: args.limit_conversations]
        graph_mode = False

    categories = [int(c) for c in args.categories.split(",")]
    modes = [m.strip() for m in args.modes.split(",")]

    all_results: Dict[str, List[Dict[str, Any]]] = {m: [] for m in modes}
    resume_ids = set()
    if args.resume and os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                resume_ids.add((row["mode"], row["sample_id"], row["question"]))
                all_results[row["mode"]].append(row)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    if graph_mode:
        conv_list = [{"sample_id": "conv-26", "graph_path": args.extracted_graph}]
    else:
        conv_list = conversations

    for conv_idx, conv in enumerate(conv_list):
        if graph_mode:
            cid = load_extracted_graph(pipeline, conv["graph_path"], "locomo_conv_0")
            sample_id = "conv-26"
            import json as _json
            with open(conv["graph_path"], "r", encoding="utf-8") as f:
                gd = _json.load(f)
            qa_list = filter_qa(gd["qa"], categories)
            full_dialogue = "\n".join(s["formatted_text"] for s in gd["sessions"])
        else:
            sample_id = conv["sample_id"]
            print(f"\n[LoCoMo] ({conv_idx+1}/{len(conv_list)}) Ingesting {sample_id}...")
            cid = seed_conversation(pipeline, conv, conversation_id=sample_id)
            qa_list = filter_qa(conv["qa"], categories)
            full_dialogue = "\n".join(t["formatted_line"] for t in conv["turns"])

        if args.max_questions:
            qa_list = qa_list[: args.max_questions]

        print(f"  {len(qa_list)} questions (categories {categories})")

        for qa in qa_list:
            question = qa["question"]
            for mode in modes:
                key = (mode, sample_id, question)
                if key in resume_ids:
                    continue

                t0 = time.time()
                prediction = generate_answer(
                    pipeline, cid, question, mode,
                    k=args.k, max_new_tokens=args.max_new_tokens,
                    full_dialogue=full_dialogue,
                )
                latency = time.time() - t0
                f1 = score_qa_item(qa, prediction)

                row = {
                    "mode": mode,
                    "sample_id": sample_id,
                    "question": question,
                    "prediction": prediction,
                    "answer": qa.get("answer") or qa.get("adversarial_answer"),
                    "category": int(qa["category"]),
                    "category_name": CATEGORY_NAMES.get(int(qa["category"]), "unknown"),
                    "f1": f1,
                    "latency": latency,
                    "evidence": qa.get("evidence", []),
                }
                all_results[mode].append(row)

                with open(args.output, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")

                if len(all_results[mode]) % 10 == 0:
                    agg = aggregate_results(all_results[mode])
                    print(f"  [{mode}] {len(all_results[mode])} done — running F1: {agg['overall_f1']:.1f}%")

    # Final report
    report = {"modes": {}, "config": vars(args)}
    for mode, results in all_results.items():
        report["modes"][mode] = aggregate_results(results)

    report_path = args.output.replace(".jsonl", "_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print("LoCoMo Benchmark Results (Official F1)")
    print("=" * 60)
    for mode, agg in report["modes"].items():
        print(f"\n  [{mode.upper()}] Overall F1: {agg['overall_f1']:.2f}% ({agg['num_questions']} questions)")
        for cat, score in agg.get("per_category", {}).items():
            print(f"    {cat}: {score:.2f}%")
    print(f"\nResults: {args.output}")
    print(f"Report:  {report_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Full LoCoMo benchmark with official F1 scoring")
    parser.add_argument("--dataset", default="data/locomo10.json")
    parser.add_argument("--extracted-graph", default=None, help="Use pre-extracted graph JSON instead of raw dataset")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--men-checkpoint", default="data/checkpoints/men_state_dict.pt")
    parser.add_argument("--modes", default="rag,cgm", help="Comma-separated: text,rag,cgm")
    parser.add_argument("--categories", default="1,2,3,4", help="LoCoMo categories to score")
    parser.add_argument("--k", type=int, default=10, help="Retrieval top-k")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--limit-conversations", type=int, default=None)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--output", default="data/locomo/results/run.jsonl")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
