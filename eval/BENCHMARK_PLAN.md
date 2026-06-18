# CGM Benchmark Plan: Beat mem0 & supermemory on LoCoMo + LongMemEval

**Targets:** LoCoMo ≥93% mean F1 (beat mem0 92.5%, supermemory 77.1%) · LongMemEval ≥95% (beat mem0 94.4%, supermemory 81.6–85.4%)

**Current baseline:** LoCoMo ~53% keyword recall on 30-Q subset · LongMemEval 4-question synthetic demo · Qwen2.5-0.5B on 4GB GPU

---

## Phase 0 — Measurement Foundation (Week 1)

**Goal:** Apples-to-apples comparison with competitors. No model changes until scoring is correct.

| Task | Status | Files |
|------|--------|-------|
| Port official LoCoMo F1 scorer (stemmed token F1, per-category) | Done | `cgm/eval/locomo_score.py` |
| Load `locomo10.json` (10 conversations, 1,986 QA) | Done | `cgm/eval/locomo_load.py` |
| Turn-level ingest (not session-level) | Done | `cgm/eval/locomo_ingest.py` |
| Full harness CLI with resume + JSONL output | Done | `run_locomo_benchmark.py` |
| LongMemEval loader + per-question haystack ingest | Done | `cgm/eval/longmem_*.py`, `run_longmem_benchmark.py` |
| Download scripts for official datasets | TODO | `scripts/download_benchmarks.py` |

**KPI gate:** Harness runs end-to-end on 1 conversation with official F1 before any recall optimization.

```bash
# Download LoCoMo
curl -L -o data/locomo10.json https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json

# Smoke test (1 conversation, RAG only)
python run_locomo_benchmark.py --dataset data/locomo10.json --limit-conversations 1 --modes rag --max-questions 20

# Full eval (all 10 conversations, categories 1-4 = 1540 questions)
python run_locomo_benchmark.py --dataset data/locomo10.json --modes rag,cgm,text --output data/locomo/results/full_run.jsonl
```

---

## Phase 1 — Close the Recall Gap (Weeks 2–4)

**Root cause:** Train/inference distribution mismatch, not abstention or scoring alone.

### P0 fixes (implemented in this sprint)

| Fix | Expected lift | Rationale |
|-----|---------------|-----------|
| Triple CE-rerank: best triple at **end** of KV cache (closest to prompt) | +10–20% | Was sorted descending (best at front) — opposite of memory turn ordering |
| Cap injected triples (`max_injected_triples=20`) | +5% | Neighbor expansion injected 69 slots vs 41 trained |
| Unified `SYSTEM_PREFIX` in training + inference | +10–15% | MEN trained without system prefix; pipeline adds it at inference |
| Retrieval-aware MEN training (target triple + distractors) | +15–25% | Training used full graph cache; inference uses retrieval subset |
| Raise `distill_lambda` to 0.5–2.0 on benchmark path | +5–10% | Was 0.05 in `benchmark.py` (20× weaker than intended) |
| Deterministic `encode_triple` (no order-dependent neighbor mutation) | +3–5% | Encoding varied by triple list order |

### P1 fixes (next sprint)

| Fix | Expected lift |
|-----|---------------|
| Triple-level retrieval (not turn-level → triple explosion) | +10–15% |
| Teacher KV from answer text, not triple_text last-token | +5–10% |
| Train targets = short answers (match LoCoMo F1), not full sentences | +10% |
| Hybrid mode: CGM inject + text RAG fallback when CE score < threshold | +15–20% |
| Per-category tuning (temporal + multi-hop are mem0's biggest wins) | +10–15% |

### P2 fixes (polish)

- Routing ablation (gate floor ≥0.5 or disable SA-KVR)
- Disable empty user/tenant scope retrieval in single-conversation eval
- LLM backend upgrade path (GPT-4o via API for fair competitor comparison)

**KPI gate:** CGM beats text-RAG on local stress suite (`benchmark.py`) at ≥70% recall before running full LoCoMo.

---

## Phase 2 — Extraction & Graph Quality (Weeks 3–5)

Competitors win on **what gets stored**, not just **how it's retrieved**.

| Task | Impact |
|------|--------|
| Rebuild `extract_locomo_graph.py` for all 10 conversations | Required for CGM (needs triples) |
| Session-aware triple extraction (Gemini or local LLM) | Temporal + multi-hop depend on this |
| Store observations + session summaries from LoCoMo JSON | RAG baseline parity with official scripts |
| Temporal validity: supersede old triples on knowledge updates | LongMemEval `knowledge-update` type |
| Negation triples with `is_negated=1` | LongMemEval abstention + negation |
| Evidence-linked triples (map `dia_id` → turn_id) | Better QA→triple alignment than embedding heuristic |

**KPI gate:** Triple coverage ≥80% of evidence turns per conversation.

---

## Phase 3 — Full LoCoMo Campaign (Week 5–6)

### Protocol (match competitor fairness)

1. **Ingest only** — no MEN training on benchmark QA labels (default `--no-train`)
2. **Optional MEN warm-start** — train on synthetic triples from held-out conversations only
3. **Three modes:** `text` (full dialogue RAG), `rag` (top-k turns), `cgm` (KV inject)
4. **Score:** Official F1 per category + overall mean
5. **Report:** Overall F1, per-category F1, tokens/query, latency, retrieval recall

### Category-specific strategy

| Cat | Type | Count | Strategy |
|-----|------|-------|----------|
| 1 | Multi-hop | 282 | 2-hop neighbor expansion + hybrid text fallback |
| 2 | Temporal | 321 | `valid_until` supersession + session timestamps |
| 3 | Open-domain | 96 | Broader retrieval k + abstention for unknown |
| 4 | Single-hop | 841 | CGM inject (highest win potential) |
| 5 | Adversarial | 446 | Abstention gate (separate metric) |

**KPI gate:** CGM ≥ RAG on overall F1, then CGM > RAG by ≥5 points.

---

## Phase 4 — Full LongMemEval Campaign (Week 6–8)

### Protocol

1. Start with **oracle** (`longmemeval_oracle.json`) — evidence sessions only
2. Scale to **S** (`longmemeval_s_cleaned.json`) — ~115K tokens/question
3. Score with official GPT-4o autoeval (`evaluate_qa.py`)

```bash
# Download
curl -L -o data/longmemeval_oracle.json https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json

# Run CGM
python run_longmem_benchmark.py --dataset data/longmemeval_oracle.json --mode cgm --output data/longmem/hypotheses.jsonl

# Score (requires OPENAI_API_KEY + LongMemEval eval scripts)
python run_longmem_benchmark.py --score data/longmem/hypotheses.jsonl --reference data/longmemeval_oracle.json
```

### Per-question isolation

Each LongMemEval instance has a **unique haystack** → one `conversation_id` per `question_id`, fresh DB per question.

### Question-type strategy

| Type | CGM approach |
|------|-------------|
| `single-session-user` | Triple inject from user turns |
| `single-session-assistant` | Assistant turn triples |
| `single-session-preference` | Preference triples with user scope |
| `temporal-reasoning` | Timestamp-ordered turns + `valid_until` |
| `knowledge-update` | Supersede old triples |
| `multi-session` | Cross-session graph neighbors |
| `*_abs` (30 abstention) | Abstention gate |

**KPI gate:** Oracle ≥80%, then S ≥85%, targeting ≥95% to beat mem0.

---

## Phase 5 — Model & Production Parity (Week 8–10)

To beat competitors **fairly** in published numbers:

| Upgrade | Why |
|---------|-----|
| GPT-4o / Qwen2.5-7B+ backend | Isolates memory architecture from base model weakness |
| Same judge model as LongMemEval official eval | Comparable scoring |
| MemoryBench / mem0 benchmarks repo harness | Third-party reproducibility |
| Publish token/latency Pareto curve | CGM's differentiation story |

---

## Success Metrics Dashboard

| Metric | Now | Phase 1 target | Final target | mem0 | supermemory |
|--------|-----|----------------|--------------|------|-------------|
| LoCoMo overall F1 | ~53%* | 65% | **≥93%** | 92.5% | 77.1% |
| LoCoMo temporal F1 | — | 60% | **≥93%** | 92.8% | — |
| LoCoMo multi-hop F1 | — | 55% | **≥93%** | 93.3% | — |
| LongMemEval overall | 4-Q demo | 70% oracle | **≥95%** | 94.4% | 81.6% |
| Tokens / query (CGM) | 13 | <100 | <500 | 6,956 | — |
| Local stress recall | 9.8% | 70% | 85% | — | — |

*Non-official keyword metric on 30-Q subset; replace with official F1 immediately.

---

## Execution Priority (this sprint)

```
[Done]  Phase 0 harness + official F1 scorer
[Done]  P0: triple sort fix, max cap, system prefix, retrieval-aware training
[Next]  Run RAG baseline on full LoCoMo → establish true baseline
[Next]  Rebuild triple extraction for all 10 conversations
[Next]  Hybrid CGM+text mode
[Next]  Full LoCoMo CGM run → compare to RAG
[Next]  LongMemEval oracle run
```

---

## Risk Register

| Risk | Mitigation |
|------|------------|
| 0.5B model ceiling | Upgrade backend for published numbers; keep 0.5B for edge story |
| Triple extraction quality | Hybrid text-RAG fallback; use LoCoMo observations field |
| OOM on context stuffing | Chunked eval or skip stuffing baseline on laptop |
| MEN not generalizing | Retrieval-aware training + hybrid mode |
| Competitors use GPT-4o | Match backend for fair shootout; differentiate on tokens/latency |
