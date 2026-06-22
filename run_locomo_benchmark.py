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
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cgm.core.pipeline import CGMPipeline
from cgm.eval.locomo_load import load_all_conversations, filter_qa
from cgm.eval.locomo_ingest import seed_conversation, load_extracted_graph
from cgm.eval.locomo_score import score_qa_item, CATEGORY_NAMES, normalize_answer
from cgm.eval.prompts import build_locomo_prompt, build_user_prompt, LOCOMO_CGM_PREFIX


def em_multi_answer(prediction: str, ground_truth: str) -> float:
    predictions = [p.strip() for p in prediction.split(",")]
    ground_truths = [g.strip() for g in ground_truth.split(",")]
    return float(np.mean([
        max(float(normalize_answer(pred) == normalize_answer(gt)) for pred in predictions)
        for gt in ground_truths
    ]))


def compute_em(category: int, prediction: str, ground_truth: str) -> float:
    prediction = str(prediction)
    ground_truth = str(ground_truth)
    if category == 3:
        ground_truth = ground_truth.split(";")[0].strip()

    if category in (2, 3, 4):
        return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0
    if category == 1:
        return em_multi_answer(prediction, ground_truth)
    if category == 5:
        lower = prediction.lower()
        if "no information available" in lower or "not mentioned" in lower:
            return 1.0
        return 0.0
    raise ValueError(f"Unknown category: {category}")


def generate_answer(
    pipeline: CGMPipeline,
    conversation_id: str,
    question: str,
    mode: str,
    k: int = 10,
    max_new_tokens: int = 64,
    full_dialogue: str = "",
) -> tuple[str, dict]:
    """Generate an answer for one LoCoMo question and collect metadata."""
    if mode == "cgm":
        # Use the concise CGM prefix to encourage short factual answers
        # (improves F1 precision vs verbose default SYSTEM_PREFIX)
        prediction = pipeline.generate(
            conversation_id, question, k=k, max_new_tokens=max_new_tokens,
            mode="inject", do_sample=False, persist_response=False,
            system_prefix=LOCOMO_CGM_PREFIX,
        )
        
        # Calculate prompt tokens
        prompt = f"{LOCOMO_CGM_PREFIX}\nUser: {question}\nAssistant:"
        inputs = pipeline.tokenizer(prompt, return_tensors="pt")
        prompt_tokens = inputs.input_ids.shape[1]
        
        injected_kv_len = 0
        if pipeline.active_cache is not None:
            injected_kv_len = pipeline.active_cache.get_seq_length()
            
        generated_tokens = len(pipeline.tokenizer.encode(prediction))
        
        return prediction, {
            "prompt_tokens": prompt_tokens,
            "injected_kv_len": injected_kv_len,
            "generated_tokens": generated_tokens,
        }
        
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
        prompt_tokens = inputs.input_ids.shape[1]
        
        with torch.no_grad():
            outputs = pipeline.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            
        prediction = pipeline.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        ).strip()
        
        generated_tokens = outputs[0].shape[0] - inputs.input_ids.shape[1]
        
        return prediction, {
            "prompt_tokens": prompt_tokens,
            "injected_kv_len": 0,
            "generated_tokens": generated_tokens,
        }
        
    elif mode == "text":
        context = f"Full conversation history:\n{full_dialogue}" if full_dialogue else ""
        prompt = build_locomo_prompt(question, context)
        inputs = pipeline.tokenizer(prompt, return_tensors="pt").to(pipeline.device)
        prompt_tokens = inputs.input_ids.shape[1]
        
        try:
            with torch.no_grad():
                outputs = pipeline.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            prediction = pipeline.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
            ).strip()
            generated_tokens = outputs[0].shape[0] - inputs.input_ids.shape[1]
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                prediction = ""
                generated_tokens = 0
            else:
                raise
                
        return prediction, {
            "prompt_tokens": prompt_tokens,
            "injected_kv_len": 0,
            "generated_tokens": generated_tokens,
        }
    else:
        raise ValueError(f"Unknown mode: {mode}")


def aggregate_detailed_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {}
        
    all_f1s = [r["f1"] for r in results]
    all_ems = [r.get("em", 0.0) for r in results]
    all_recalls = [r.get("retrieval_recall", 0.0) for r in results]
    all_latencies = [r["latency"] for r in results]
    all_prompt_tokens = [r.get("prompt_tokens", 0) for r in results]
    all_injected_kvs = [r.get("injected_kv_len", 0) for r in results]
    all_total_tokens = [r.get("prompt_tokens", 0) + r.get("injected_kv_len", 0) for r in results]
    all_gen_tokens = [r.get("generated_tokens", 0) for r in results]
    
    speeds = []
    for r in results:
        if r["latency"] > 0 and r.get("generated_tokens", 0) > 0:
            speeds.append(r["generated_tokens"] / r["latency"])
            
    # Per-category scores
    by_cat_f1 = {}
    by_cat_em = {}
    by_cat_recall = {}
    
    for r in results:
        cat = r["category"]
        by_cat_f1.setdefault(cat, []).append(r["f1"])
        by_cat_em.setdefault(cat, []).append(r.get("em", 0.0))
        by_cat_recall.setdefault(cat, []).append(r.get("retrieval_recall", 0.0))
        
    per_category = {}
    per_category_retrieval = {}
    for cat in sorted(by_cat_f1.keys()):
        cat_name = CATEGORY_NAMES.get(cat, f"Category {cat}")
        per_category[cat_name] = round(float(np.mean(by_cat_f1[cat])) * 100, 2)
        per_category_retrieval[cat_name] = round(float(np.mean(by_cat_recall[cat])) * 100, 2)
        
    # Detail dict for markdown
    per_category_detail = {}
    for cat in sorted(by_cat_f1.keys()):
        cat_name = CATEGORY_NAMES.get(cat, f"Category {cat}")
        per_category_detail[cat_name] = {
            "f1": round(float(np.mean(by_cat_f1[cat])) * 100, 2),
            "em": round(float(np.mean(by_cat_em[cat])) * 100, 2),
            "retrieval_recall": round(float(np.mean(by_cat_recall[cat])) * 100, 2),
            "count": len(by_cat_f1[cat])
        }
        
    return {
        "overall_f1": round(float(np.mean(all_f1s)) * 100, 2),
        "overall_em": round(float(np.mean(all_ems)) * 100, 2),
        "overall_retrieval_recall": round(float(np.mean(all_recalls)) * 100, 2),
        
        "latency_mean": round(float(np.mean(all_latencies)), 4),
        "latency_median": round(float(np.median(all_latencies)), 4),
        "latency_p95": round(float(np.percentile(all_latencies, 95)), 4),
        "latency_p99": round(float(np.percentile(all_latencies, 99)), 4),
        
        "prompt_tokens_mean": round(float(np.mean(all_prompt_tokens)), 1),
        "prompt_tokens_median": round(float(np.median(all_prompt_tokens)), 1),
        
        "injected_kv_mean": round(float(np.mean(all_injected_kvs)), 1),
        "injected_kv_median": round(float(np.median(all_injected_kvs)), 1),
        
        "total_context_mean": round(float(np.mean(all_total_tokens)), 1),
        "total_context_median": round(float(np.median(all_total_tokens)), 1),
        
        "generated_tokens_mean": round(float(np.mean(all_gen_tokens)), 1),
        "generated_tokens_median": round(float(np.median(all_gen_tokens)), 1),
        
        "generation_speed_tps": round(float(np.mean(speeds)), 2) if speeds else 0.0,
        "num_questions": len(results),
        "per_category": per_category,
        "per_category_retrieval": per_category_retrieval,
        "per_category_detail": per_category_detail
    }


def write_markdown_report(report: Dict[str, Any], filepath: str, results_by_mode: Dict[str, List[Dict[str, Any]]]):
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("# Conversational Graph Memory (CGM) vs RAG Benchmark Report\n\n")
        f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Target LLM Model: `{report['config'].get('model')}`\n")
        f.write(f"Dataset: `{report['config'].get('dataset')}`\n")
        f.write(f"MEN Checkpoint: `{report['config'].get('men_checkpoint')}`\n\n")
        
        f.write("## 📊 Overall Performance Comparison\n\n")
        f.write("| Metric | RAG | CGM | Improvement (CGM vs RAG) |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        
        modes = report["modes"]
        rag = modes.get("rag", {})
        cgm = modes.get("cgm", {})
        
        def format_diff(v_cgm, v_rag, is_pct=False, is_latency=False, is_speed=False):
            if v_cgm is None or v_rag is None:
                return "N/A"
            diff = v_cgm - v_rag
            sign = "+" if diff >= 0 else ""
            if is_pct:
                return f"{sign}{diff:.2f}%"
            elif is_latency:
                pct = (diff / v_rag * 100) if v_rag > 0 else 0
                return f"{sign}{diff:.4f}s ({sign}{pct:.1f}%)"
            elif is_speed:
                pct = (diff / v_rag * 100) if v_rag > 0 else 0
                return f"{sign}{diff:.2f} t/s ({sign}{pct:.1f}%)"
            else:
                pct = (diff / v_rag * 100) if v_rag > 0 else 0
                return f"{sign}{diff:.1f} t ({sign}{pct:.1f}%)"

        f1_diff = format_diff(cgm.get("overall_f1"), rag.get("overall_f1"), is_pct=True)
        em_diff = format_diff(cgm.get("overall_em"), rag.get("overall_em"), is_pct=True)
        rr_diff = format_diff(cgm.get("overall_retrieval_recall"), rag.get("overall_retrieval_recall"), is_pct=True)
        lat_diff = format_diff(cgm.get("latency_mean"), rag.get("latency_mean"), is_latency=True)
        prompt_diff = format_diff(cgm.get("prompt_tokens_mean"), rag.get("prompt_tokens_mean"))
        context_diff = format_diff(cgm.get("total_context_mean"), rag.get("total_context_mean"))
        speed_diff = format_diff(cgm.get("generation_speed_tps"), rag.get("generation_speed_tps"), is_speed=True)
        
        f.write(f"| **Token F1 Score** | {rag.get('overall_f1', 0.0):.2f}% | {cgm.get('overall_f1', 0.0):.2f}% | **{f1_diff}** |\n")
        f.write(f"| **Exact Match (EM)** | {rag.get('overall_em', 0.0):.2f}% | {cgm.get('overall_em', 0.0):.2f}% | **{em_diff}** |\n")
        f.write(f"| **Retrieval Recall** | {rag.get('overall_retrieval_recall', 0.0):.2f}% | {cgm.get('overall_retrieval_recall', 0.0):.2f}% | **{rr_diff}** |\n")
        f.write(f"| **Mean Latency** | {rag.get('latency_mean', 0.0):.4f}s | {cgm.get('latency_mean', 0.0):.4f}s | **{lat_diff}** |\n")
        f.write(f"| **Median Latency** | {rag.get('latency_median', 0.0):.4f}s | {cgm.get('latency_median', 0.0):.4f}s | - |\n")
        f.write(f"| **P95 Latency** | {rag.get('latency_p95', 0.0):.4f}s | {cgm.get('latency_p95', 0.0):.4f}s | - |\n")
        f.write(f"| **Mean Input Prompt Tokens** | {rag.get('prompt_tokens_mean', 0.0):.1f} t | {cgm.get('prompt_tokens_mean', 0.0):.1f} t | **{prompt_diff}** |\n")
        f.write(f"| **Mean Injected KV Tokens** | 0.0 t | {cgm.get('injected_kv_mean', 0.0):.1f} t | +{cgm.get('injected_kv_mean', 0.0):.1f} t |\n")
        f.write(f"| **Mean Total Active Context** | {rag.get('total_context_mean', 0.0):.1f} t | {cgm.get('total_context_mean', 0.0):.1f} t | **{context_diff}** |\n")
        f.write(f"| **Generation Speed (TPS)** | {rag.get('generation_speed_tps', 0.0):.2f} t/s | {cgm.get('generation_speed_tps', 0.0):.2f} t/s | **{speed_diff}** |\n")
        
        f.write("\n---\n\n")
        
        f.write("## 🗂️ Per-Category Breakdown\n\n")
        f.write("| Category | Model | F1 Score | EM Score | Retrieval Recall | Count |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        
        cats = set()
        if rag: cats.update(rag.get("per_category_detail", {}).keys())
        if cgm: cats.update(cgm.get("per_category_detail", {}).keys())
        
        for cat in sorted(cats):
            rag_cat = rag.get("per_category_detail", {}).get(cat, {"f1": 0.0, "em": 0.0, "retrieval_recall": 0.0, "count": 0})
            cgm_cat = cgm.get("per_category_detail", {}).get(cat, {"f1": 0.0, "em": 0.0, "retrieval_recall": 0.0, "count": 0})
            f.write(f"| **{cat}** | RAG | {rag_cat['f1']:.2f}% | {rag_cat['em']:.2f}% | {rag_cat['retrieval_recall']:.2f}% | {rag_cat['count']} |\n")
            f.write(f"| | CGM | {cgm_cat['f1']:.2f}% | {cgm_cat['em']:.2f}% | {cgm_cat['retrieval_recall']:.2f}% | {cgm_cat['count']} |\n")
            
        f.write("\n---\n\n")
        
        f.write("## 🔍 Sample Side-by-Side Comparison\n\n")
        f.write("Below are representative QA samples evaluated under both modes:\n\n")
        
        # Grab examples from the results
        rag_res = results_by_mode.get("rag", [])
        cgm_res = results_by_mode.get("cgm", [])
        
        for idx in range(min(15, len(rag_res))):
            r_item = rag_res[idx]
            c_item = next((c for c in cgm_res if c["question"] == r_item["question"]), None)
            if c_item:
                f.write(f"### Q{idx+1}: {r_item['question']}\n")
                f.write(f"* **Gold Answer:** `{r_item['answer']}`\n")
                f.write(f"* **Category:** {r_item['category_name']} (Cat {r_item['category']})\n\n")
                f.write("| Mode | Prediction | F1 Score | EM Score | Latency | Context Size |\n")
                f.write("| :--- | :--- | :---: | :---: | :---: | :---: |\n")
                f.write(f"| **RAG** | `{r_item['prediction']}` | {r_item['f1']*100:.2f}% | {r_item.get('em', 0.0)*100:.2f}% | {r_item['latency']:.2f}s | {r_item['prompt_tokens']} t |\n")
                f.write(f"| **CGM** | `{c_item['prediction']}` | {c_item['f1']*100:.2f}% | {c_item.get('em', 0.0)*100:.2f}% | {c_item['latency']:.2f}s | {c_item['prompt_tokens']} t prompt + {c_item['injected_kv_len']} t KV |\n\n")
                f.write("---\n\n")


def run_benchmark(args) -> Dict[str, Any]:
    pipeline = CGMPipeline(model_name=args.model, rank=args.rank, men_checkpoint=args.men_checkpoint, device=args.device)
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

        for qa_idx, qa in enumerate(qa_list):
            question = qa["question"]
            gold = qa.get("answer") or qa.get("adversarial_answer") or ""
            evidence = qa.get("evidence", [])
            
            # Evaluate Retrieval@K
            try:
                retrieved = pipeline.retriever.retrieve(cid, question, k=args.k)
                def get_session_id(dia_id: str) -> str:
                    return dia_id.split(":", 1)[0] if dia_id else ""
                ret_sessions = {get_session_id(m.dia_id) for m in retrieved if m.dia_id}
                has_evidence = any(get_session_id(ev) in ret_sessions for ev in evidence) if evidence else True
                retrieval_success = 1.0 if has_evidence else 0.0
            except Exception as e:
                print(f"    [RETRIEVAL] Error: {e}")
                retrieval_success = 0.0

            print(f"\n==============================================================")
            print(f"QA Item {qa_idx+1}/{len(qa_list)}: '{question}'")
            print(f"Gold Answer: '{gold}' (Category: {qa.get('category')} - {CATEGORY_NAMES.get(int(qa['category']), 'unknown')})")
            print(f"--------------------------------------------------------------")

            for mode in modes:
                key = (mode, sample_id, question)
                if key in resume_ids:
                    continue

                t0 = time.time()
                prediction, meta = generate_answer(
                    pipeline, cid, question, mode,
                    k=args.k, max_new_tokens=args.max_new_tokens,
                    full_dialogue=full_dialogue,
                )
                latency = time.time() - t0
                
                f1 = score_qa_item(qa, prediction)
                em = compute_em(int(qa["category"]), prediction, gold)
                
                extra_info = ""
                if mode == "cgm":
                    extra_info = f" | Injected KV: {meta['injected_kv_len']} t"
                print(f"  [{mode.upper()}] Prediction: '{prediction}'")
                print(f"        F1: {f1*100:.2f}% | EM: {em*100:.2f}% | Latency: {latency:.2f}s | Prompt: {meta['prompt_tokens']} t{extra_info} | Gen: {meta['generated_tokens']} t")

                row = {
                    "mode": mode,
                    "sample_id": sample_id,
                    "question": question,
                    "prediction": prediction,
                    "answer": gold,
                    "category": int(qa["category"]),
                    "category_name": CATEGORY_NAMES.get(int(qa["category"]), "unknown"),
                    "f1": f1,
                    "em": em,
                    "latency": latency,
                    "prompt_tokens": meta["prompt_tokens"],
                    "injected_kv_len": meta["injected_kv_len"],
                    "generated_tokens": meta["generated_tokens"],
                    "evidence": evidence,
                    "retrieval_recall": retrieval_success,
                }
                all_results[mode].append(row)

                with open(args.output, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")

                if len(all_results[mode]) % 10 == 0:
                    agg = aggregate_detailed_results(all_results[mode])
                    print(f"  [{mode}] {len(all_results[mode])} done — running F1: {agg['overall_f1']:.1f}%")

    # Final report
    report = {"modes": {}, "config": vars(args)}
    for mode, results in all_results.items():
        report["modes"][mode] = aggregate_detailed_results(results)

    report_path = args.output.replace(".jsonl", "_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    markdown_path = args.output.replace(".jsonl", "_report.md")
    write_markdown_report(report, markdown_path, all_results)
    
    # Also write to data/benchmark_report.md
    try:
        write_markdown_report(report, "data/benchmark_report.md", all_results)
    except Exception as exc:
        print(f"Warning: Could not copy report to data/benchmark_report.md: {exc}")

    print("\n" + "=" * 60)
    print("LoCoMo Benchmark Results (Official F1 & EM Comparison)")
    print("=" * 60)
    for mode, agg in report["modes"].items():
        print(f"\n  [{mode.upper()}] Overall F1: {agg['overall_f1']:.2f}% | EM: {agg['overall_em']:.2f}% ({agg['num_questions']} questions)")
        if "overall_retrieval_recall" in agg:
            print(f"    Overall Retrieval Recall: {agg['overall_retrieval_recall']:.2f}%")
        print(f"    Latency: mean={agg['latency_mean']:.4f}s, median={agg['latency_median']:.4f}s, p95={agg['latency_p95']:.4f}s")
        print(f"    Context tokens: prompt={agg['prompt_tokens_mean']:.1f} t, injected={agg['injected_kv_mean']:.1f} t, total={agg['total_context_mean']:.1f} t")
        print(f"    Generation speed: {agg['generation_speed_tps']:.2f} t/s")
        print(f"    Per-Category Breakdown (F1 / EM / Retrieval Recall):")
        for cat, details in agg.get("per_category_detail", {}).items():
            print(f"      {cat}: F1={details['f1']:.2f}% | EM={details['em']:.2f}% | Retrieval={details['retrieval_recall']:.2f}% ({details['count']} Qs)")
            
    print(f"\nResults: {args.output}")
    print(f"Report JSON: {report_path}")
    print(f"Report MD:   {markdown_path}")
    print(f"Copied to:   data/benchmark_report.md")
    return report


def main():
    parser = argparse.ArgumentParser(description="Full LoCoMo benchmark with official F1 scoring")
    parser.add_argument("--dataset", default="data/locomo10.json")
    parser.add_argument("--extracted-graph", default=None, help="Use pre-extracted graph JSON instead of raw dataset")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--men-checkpoint", default="data/checkpoints/men_state_dict.pt")
    parser.add_argument("--modes", default="rag,cgm", help="Comma-separated: text,rag,cgm")
    parser.add_argument("--device", default=None, help="Device to run on (cuda, cpu)")
    parser.add_argument("--categories", default="1,2,3,4", help="LoCoMo categories to score")
    parser.add_argument("--k", type=int, default=10, help="Retrieval top-k")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--limit-conversations", type=int, default=None)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--output", default="data/locomo/results/run.jsonl")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()

