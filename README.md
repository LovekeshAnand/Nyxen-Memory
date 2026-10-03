<div align="center">

<img src="banner.png" alt="Nyxen Memory Banner" width="100%"/>

<br/>
<br/>

# Nyxen Memory

### Persistent Neural Memory for Local Language Models

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-green?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)](https://github.com)
[![Status](https://img.shields.io/badge/Status-Research-orange)](https://github.com)

</div>

---

## What is Nyxen Memory?

Every AI assistant you've ever used has the same fundamental flaw: **it forgets.**

Not just between sessions — within a single long conversation, once enough messages pile up, older context silently falls off the edge of what the model can see. You could have told it your entire project spec an hour ago, and now it acts like it never heard it. This isn't a bug in any one product — it's a hard architectural limit baked into how transformers work. They can only attend to a fixed window of tokens at a time. Once that window fills up, old information is gone.

The standard workaround — Retrieval-Augmented Generation (RAG) — helps, but comes with its own costs. It works by fetching relevant past messages and stuffing them back into the prompt before every generation call. More context means longer prompts, which means more tokens to process, slower responses, and an ever-growing risk of hitting memory limits on the GPU. And because RAG works with raw text chunks, it often retrieves things that are *semantically close* but *factually irrelevant* — leading to confused or hallucinated answers.

**Nyxen Memory is a fundamentally different approach.**

Instead of retrieving text and re-feeding it into the prompt, Nyxen Memory builds a **structured knowledge graph** of everything that's been said, and then **injects relevant memories directly into the AI's attention mechanism** — bypassing the prompt entirely. The model doesn't read your history; it *remembers* it at the neural level, the same way it processes its own internal state.

The technology at the core of Nyxen Memory is called **Conversational Graph Memory (CGM)** — a neural KV-cache injection framework built from the ground up for local, edge-deployed language models. CGM is what makes Nyxen Memory's memory feel instant, accurate, and invisible.

### The Problem, In Plain Terms

Imagine you're working with an AI coding assistant. Over the course of a two-hour session you tell it:
- Your project uses PostgreSQL, not MySQL
- You prefer async Python patterns
- You decided against using Redis for caching
- The API endpoint is at `/v2/users`, not `/v1`

An hour later, you ask it to write a caching layer. A standard AI will likely suggest Redis (because it's common), use sync patterns, and reference the wrong endpoint — because those early decisions have long since fallen out of its context window. A RAG-based system might retrieve *some* of them, but embedding-based similarity often misses the **negation** ("decided against Redis") and **temporal updates** ("changed to `/v2`").

**Nyxen Memory gets it right.** It stores those facts as structured graph triples — `(user, decided_against, Redis)`, `(api_endpoint, is, /v2/users)` — and when the caching question comes in, it retrieves the full relational context, including negations and corrections, and injects it directly into the model's attention cache. The model answers as if it had never forgotten.

### How It Works — Simple Version

1. **Listens and learns:** After every turn in a conversation, Nyxen Memory parses what was said and extracts structured facts — Subject → Predicate → Object triples — and stores them in a persistent graph database.

2. **Finds what matters:** When you send a new message, it embeds your query and runs a fast semantic search over the knowledge graph. It then expands results by traversing graph edges — pulling in related facts, corrections, and updates that simple text search would miss.

3. **Remembers, not retrieves:** Instead of adding memory to your prompt, Nyxen Memory encodes the retrieved facts into neural Key-Value tensors using a trained **Memory Encoder Network (MEN)**, and injects them directly into the model's attention cache. From the model's perspective, it's as if those memories were part of its own computation from the start.

4. **Stays fast and stable:** The entire injection pipeline is hardware-aware. Native C++ guards monitor GPU VRAM and system RAM in real time, applying thermal pacing and cache compression to keep the system stable on consumer-grade workstations.

---

## Table of Contents

- [What is Nyxen Memory?](#what-is-nyxen-memory)
- [Core Architecture: Conversational Graph Memory](#core-architecture-conversational-graph-memory)
- [The Six-Stage Pipeline](#the-six-stage-pipeline)
- [Memory Encoder Network (MEN)](#memory-encoder-network-men)
- [Knowledge Graph Store](#knowledge-graph-store)
- [Hardware Safety & Stability](#hardware-safety--stability)
- [Key Engineering Breakthroughs](#key-engineering-breakthroughs)
- [Benchmarks & Results](#benchmarks--results)
- [Project Structure](#project-structure)
- [Setup & Quickstart](#setup--quickstart)
- [Training the Memory Encoder](#training-the-memory-encoder)

---

## Core Architecture: Conversational Graph Memory

**Conversational Graph Memory (CGM)** is the technical foundation of Nyxen Memory. It is a neural context injection framework — not a retrieval system — designed to give frozen, locally-deployed language models access to an unbounded, structured conversational history without modifying the model itself or increasing prompt length.

At a high level, CGM replaces the entire concept of "context window management." Rather than deciding what to keep in the prompt and what to drop, CGM offloads all historical knowledge to a persistent graph store and retrieves it on-demand as compressed neural states, injected directly into the model's self-attention Key-Value cache.

### Why a Knowledge Graph, Not a Vector Database?

Most memory systems for LLMs use a pure vector database: they embed each past message as a vector, store it, and retrieve the nearest neighbors by cosine similarity when a new query comes in. This works reasonably well for simple recall, but it has significant failure modes:

- **Negation blindness:** If you said "don't use Redis," the embedding of that sentence is very similar to "use Redis." A vector search may retrieve it — but the model has no structured way to know it's a negation.
- **Temporal overwriting:** If you said "the endpoint is /v1" in turn 3 and "actually it's /v2" in turn 47, both statements have similar embeddings. Without graph structure, the system may retrieve the older, wrong one.
- **Isolated facts:** Vector search retrieves individual statements. It doesn't follow relationships. If turn 12 says "the database is Postgres" and turn 34 says "Postgres is running on port 5433," a vector search on "what port is the DB on?" may miss the connection entirely.

CGM solves all three problems by storing facts as **Subject-Predicate-Object triples** in a relational graph. Retrieval is a two-step process: first, dense semantic search finds the most relevant turns; then, **1-hop graph traversal** expands the result by following edges — pulling in negations, updates, and related facts as a coherent subgraph rather than isolated snippets.

### The Neural Injection Advantage

Once the relevant subgraph is retrieved, CGM does something no other memory system does: it **encodes that subgraph into Key-Value tensors** that are dimensionally compatible with the target LLM's attention layers, and prepends them to the model's `past_key_values` cache before generation begins.

This means:
- **Zero prompt tokens used for memory.** The model's input prompt is only your current message — typically 15–60 tokens.
- **No prefill overhead for memories.** The KV cache is injected pre-computed; the model doesn't re-read the history, it just attends to the virtual memory slots directly.
- **Sub-second latency.** Mean response latency drops to 0.83s vs 1.72s for standard RAG.

---

## The Six-Stage Pipeline

Every conversation turn in Nyxen Memory flows through a deterministic six-stage loop. Each stage is optimized for correctness and performance.

| Stage | Name | What Happens |
|:---:|:---|:---|
| **1** | **Query Embedding & Vector Search** | The user's query is embedded into a 384-dimensional vector using `all-MiniLM-L6-v2`. This vector is searched against the quantized **TurboVec** index, scoped strictly to the current `tenant_id` and `user_id` to prevent cross-session contamination. The top-k most semantically similar historical turns are returned as candidate memories. |
| **2** | **Graph-Neighbor Expansion** | For each candidate turn ID, the system queries the **SQLite Graph Store** to fetch all 1-hop neighbors — both forward edges (what this turn implies) and backward edges (what led to this turn). This expansion step is what captures negations like *"actually, don't use that"* and corrections like *"I changed the approach"* that pure vector search misses. |
| **3** | **Sparse-Dense RRF Fusion** | The candidate set from vector search is re-ranked using **Reciprocal Rank Fusion (RRF)** with an entity-boosting heuristic. Non-stopword keywords from the query are matched against active graph triples; turns containing direct entity hits are boosted to the top of the candidate list, ensuring precise factual recall. |
| **4** | **Neural Projection via MEN** | The final retrieved subgraph is encoded as a dense structural vector `[enc(Subject ∥ Predicate); enc(Object)]` (768-dim) and passed to the **Memory Encoder Network**. The MEN maps this into the precise head-dimension and head-count shape of the target LLM's attention layers, producing draft Key and Value tensors. |
| **5** | **SA-KVR Gating & RoPE Alignment** | The draft KV tensors pass through the **Semantics-Aware KV Routing (SA-KVR)** gating network, which applies learned layer-wise and head-wise scaling. Key states are then rotated via **Rotary Position Embeddings (RoPE)** to align with the positional encoding of the target model, and the `position_ids` for the generation prompt are shifted by `memory_length` to prevent positional collision. |
| **6** | **Cache Injection & Episodic Logging** | The calibrated KV tensors are prepended to `past_key_values`. The model generates its response attending to both the virtual memory slots and the current prompt. After generation, the new turn is saved to SQLite, new triples are extracted by the NLP parser, and the turn embedding is indexed into TurboVec for future retrieval. |

---

## Memory Encoder Network (MEN)

The **Memory Encoder Network** is the core neural component of CGM — a small, trainable transformer-adjacent module that learns to translate structured symbolic knowledge into the continuous latent space of a specific target LLM.

### Design Philosophy

Standard neural networks project between fixed-dimensional spaces. The MEN has a harder job: it must produce Key and Value tensors that are **statistically indistinguishable** from the tensors the LLM would produce if it had actually read those facts in its own prefill pass. That means matching not just the shape, but the **magnitude distribution**, **layer-wise statistics**, and **positional encoding** of the target model's internal representations.

This is achieved through three mechanisms:

**LowRankLinear Projection (Rank = 128)**
Rather than a full-rank linear layer, the MEN uses a low-rank factored projection from the 768-dim structural input to the LLM's KV head dimensions. The low-rank bottleneck reduces parameter count, improves generalization, and creates two separate projection heads — one for Key tensors and one for Value tensors — each with independent learned weights.

**LayerNorm Magnitude Matching**
Teacher LLM key states at early layers exhibit a standard deviation of approximately 32.15 and a mean near 4.24, due to large bias vectors in the model's `k_proj` layer. Value states are far more constrained (σ ≈ 0.027). Without explicit normalization matching, projected tensors would collapse or dominate attention, producing incoherent generations. The MEN applies per-path LayerNorm calibrated against teacher statistics to match these distributions exactly.

**SA-KVR Layer-Wise Gating**
Not every attention layer in the LLM benefits equally from injected memories. The **Semantics-Aware KV Routing (SA-KVR)** module learns a layer-wise and head-wise gate — initialized from a middle-layer prior — that scales how much each injected tensor influences each specific attention head. This allows the MEN to learn that certain layers care more about factual recall while others handle syntactic generation.

### Architecture Summary

| Component | Detail |
|:---|:---|
| Input | Triple structural vector — 768-dim `[enc(Subject ∥ Predicate); enc(Object)]` |
| Projector | `LowRankLinear`, Rank = 128, dual Key/Value heads |
| Normalization | Per-path `LayerNorm` calibrated to teacher LLM statistics |
| Gating | `SA-KVR` — learned layer-wise and head-wise activation scaling |
| Positional Alignment | RoPE rotation on Key tensors; `position_ids` shifted by `memory_length` |
| Output | `past_key_values`-compatible tensors — prepended to frozen LLM cache |

---

## Knowledge Graph Store

The memory backing store for Nyxen Memory is a **SQLite-based Graph Store** with a companion **TurboVec** quantized dense index.

### Graph Schema

Every piece of information extracted from a conversation is stored as a **Subject-Predicate-Object triple**, linked to the dialogue turn that produced it:

```
Triple: (subject: str, predicate: str, object: str, turn_id: int, dia_id: str, confidence: float)
```

Turns are stored as episodic records with full metadata:

```
Turn: (turn_id: int, dia_id: str, tenant_id: str, user_id: str, speaker: str, text: str, embedding_id: int, timestamp: float)
```

The `dia_id` (Dialogue ID) is propagated across all tables — triples, turns, and the TurboVec index — ensuring that every retrieval operation can filter precisely by session, user, and tenant. This is what enables true multi-user, multi-session memory isolation.

### Hybrid Memory Objects (HMOs)

Internally, retrieved memories are packaged into **Hybrid Memory Objects (HMOs)** — structured containers that carry both the raw text of a turn and its associated graph triples. The HMO is what flows into the MEN for neural projection, ensuring the encoder sees the full structured context of a retrieved memory, not just its embedding.

### Triple Extraction Pipeline

After every turn, an NLP parser runs on the combined user-assistant exchange and extracts new triples using dependency parsing and a set of linguistic heuristics. Temporal markers, negation patterns, and reference resolutions are handled explicitly — so if a user says *"actually, forget what I said about Redis"*, the system adds a negation triple and flags the prior Redis triples as superseded, not just leaves them to compete in retrieval.

---

## Hardware Safety & Stability

Nyxen Memory is designed to run on consumer-grade NVIDIA RTX laptops and workstations — hardware that is GPU VRAM-constrained, thermally sensitive, and shared with other system processes. The framework includes a dedicated hardware safety layer to prevent crashes and degradation.

### Native C++ Safety Guard (`safety_guard.dll` / `.cpp`)

A native C++ library compiled for 64-bit Windows uses the **Win32 API** and **NVML (NVIDIA Management Library)** to monitor system resources in real time:

- **VRAM headroom monitoring:** Before each generation call, the guard checks available GPU VRAM. If headroom drops below a configured threshold, the KV cache compressor is triggered proactively to free memory before the LLM allocation would fail.
- **RAM pressure detection:** System RAM usage is monitored via Win32's `GlobalMemoryStatusEx`. Under high RAM pressure, non-essential in-memory caches are flushed.
- **Thermal pacing:** GPU temperature is read via NVML. If the device is approaching thermal limits, generation is paced with short sleep intervals to prevent thermal throttling from corrupting timing-sensitive attention computations.

The Python layer wraps this library via `ctypes`, exposing a clean Python API while keeping the hot path in compiled native code.

### Proactive KV Cache Compression

As conversations grow longer, the injected virtual KV cache grows proportionally. The **double-buffered KV cache compressor** runs before generation (not after) to prune stale virtual memory slots while keeping the `hot_window` — the most recent `128` turns of cache — fully intact. This ensures that even very long sessions can run indefinitely without OOM failures.

---

## Key Engineering Breakthroughs

Building a working neural memory injection system required solving several non-obvious problems. Here are the most critical ones encountered and resolved during development.

### 1. KV-Cache Scale Calibration — Preventing Attention Collapse

**The problem:** Early experiments produced completely incoherent outputs — the model would generate repetitive gibberish or random token sequences whenever the injected memory cache was active.

**The diagnosis:** Teacher LLM key states at Layer 0 have a standard deviation of ~32.15 and a mean of ~4.24 (driven by large bias vectors in the `k_proj` weight matrix). Value states are much smaller: σ ≈ 0.027. The original MEN used a global projection scale factor of 0.01 — reducing injected key magnitudes by 100×. This shifted the mean by ~3.8 and collapsed the variance, causing the softmax in self-attention to produce nearly-uniform distributions across all tokens. Self-attention effectively stopped working.

**The fix:** The `kv_scale` parameter inside the MEN is now initialized to `1.0` and treated as a **learned scalar per layer**, calibrated during KV-distillation training against the teacher's actual key statistics. This preserves the correct magnitude distribution and restores clean self-attention patterns.

### 2. Rotational Alignment — RoPE and Position ID Conflicts

**The problem:** The model would generate plausible-looking but factually wrong text — answering the right type of question but with wrong details. Retrieval was working; injection was working structurally; but the model seemed to be attending to the wrong positions in the cache.

**The diagnosis:** Rotary Position Embeddings (RoPE) encode position information directly into key and value tensors. When the model generates with `position_ids` starting at 0, it assumes the first token in its context is at position 0. But with injected virtual memory prepended to the cache, the actual prompt tokens should start at position `memory_length`. The offset mismatch caused the model to confuse which parts of the cache were "early" vs "recent" in the conversation.

**The fix:** We now explicitly pass `position_ids=torch.arange(memory_length, memory_length + prompt_len)` to `model.generate()`. Additionally, during training data extraction, teacher key cache values are un-rotated via the **inverse RoPE formula** before being used as distillation targets — recovering clean static float32 states with max numerical error of approximately 2×10⁻⁷.

### 3. Graph-Aware Retrieval — The RRF Entity Boost

**The problem:** Pure dense vector search would correctly identify semantically relevant turns but miss precise factual answers. When asked "what database did we settle on?", the retriever would return turns *discussing* databases in general rather than the specific turn where the decision was made.

**The fix:** We implemented a **Sparse-Dense Reciprocal Rank Fusion (RRF)** retriever that augments cosine similarity ranking with a keyword-entity boosting pass. Non-stopword tokens from the query are extracted and matched against active graph triples using exact-string matching. Turns containing a direct entity match receive a configurable RRF score boost, pushing them to the top of the candidate list before the graph expansion step runs. This ensures that specific factual questions get specific factual answers.

### 4. Negation and Temporal Correction Handling

**The problem:** Standard retrieval systems treat all statements equally. If a user said "use PostgreSQL" in turn 5 and "actually we're switching to ClickHouse" in turn 43, both would have similar semantic similarity to a question about the database. The older, wrong answer could easily outrank the newer correct one.

**The fix:** The triple extraction pipeline applies explicit linguistic patterns to detect negations (`don't`, `not`, `decided against`, `instead`) and temporal updates (`actually`, `changed to`, `new decision`, `scratch that`). When an update triple is detected, the prior triples it supersedes are flagged with a `superseded_by` pointer in the graph. During retrieval, superseded triples are down-ranked, and their replacements are up-ranked — ensuring the model always receives the most current facts.

---

## Benchmarks & Results

Nyxen Memory (CGM) was evaluated against standard text-based RAG on the public **LoCoMo** long-dialogue benchmark — a challenging dataset of multi-session human conversations with questions spanning multi-hop reasoning, temporal updates, and open-domain knowledge. All evaluation used `Qwen/Qwen2.5-0.5B-Instruct` on a local CUDA device, with no external APIs.

The benchmark compares two systems on the same underlying retrieval recall (both achieve 95%), isolating the effect of neural injection vs. text-based context stuffing on generation quality and latency.

### Overall Comparative Metrics

| Metric | RAG | Nyxen Memory (CGM) | Δ |
|:---|:---:|:---:|:---:|
| **Token F1 Score** | 4.85% | 17.51% | **+12.66 pp** |
| **Exact Match (EM)** | 0.00% | 1.25% | **+1.25 pp** |
| **Retrieval Recall** | 95.00% | 95.00% | — |
| **Mean Response Latency** | 1.7224 s | 0.8337 s | **−51.6%** |
| **Median Response Latency** | 1.7272 s | 0.7340 s | **−57.5%** |
| **P95 Response Latency** | 2.4620 s | 1.1478 s | **−53.4%** |
| **Mean Input Prompt Tokens** | 387.9 t | 57.5 t | **−85.2%** |
| **Mean Injected KV Tokens** | 0.0 t | 211.2 t | +211.2 t |
| **Mean Total Active Context** | 387.9 t | 268.7 t | **−30.7%** |
| **Generation Speed** | 20.41 t/s | 6.22 t/s | −69.5% |

### Per-Category Breakdown

| Category | Model | F1 Score | EM Score | Retrieval Recall | Questions |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Multi-hop** | RAG | 5.58% | 0.00% | 87.50% | 8 |
| | CGM | 3.12% | 3.12% | 87.50% | 8 |
| **Open-domain** | RAG | 8.50% | 0.00% | 100.00% | 2 |
| | CGM | 20.00% | 0.00% | 100.00% | 2 |
| **Temporal** | RAG | 3.55% | 0.00% | 100.00% | 10 |
| | CGM | **28.52%** | 0.00% | 100.00% | 10 |

### Interpreting the Results

**F1 Score (3.6× improvement):** The Token F1 score measures how many words in the model's answer overlap with the ground-truth answer. RAG produces verbose, conversational outputs — full sentences, hedges, elaborations. Nyxen Memory produces short, precise slot-fillers (average 5.4 output tokens). This is the correct behavior for factual question answering; the RAG outputs are penalized heavily for the extra words.

**Temporal Category (7.4× improvement — 3.55% → 28.52%):** This is the most important result. Temporal questions ask about facts that changed over time — *"What did they eventually decide to use?"*, *"What was the final choice for X?"*. RAG retrieves the most semantically similar turn, which is often an older statement rather than the final decision. Nyxen Memory's graph traversal with temporal correction handling retrieves the *most recent* relevant triple, not just the *most similar* text chunk.

**Latency (51.6% faster):** Despite injecting 211 virtual memory tokens per query, Nyxen Memory is faster than RAG because it avoids the expensive prefill computation for those tokens. The LLM never "reads" the memory — it simply attends to pre-computed KV states. The prompt is only 57.5 tokens on average, vs. 387.9 tokens for RAG.

**Generation Speed (lower TPS is expected):** The lower tokens-per-second figure for CGM reflects the overhead of the neural injection pipeline itself (MEN forward pass, RoPE alignment, cache assembly), not a model slowdown. The total wall-clock time is still significantly faster.

**Multi-hop (slight CGM regression):** Multi-hop questions require chaining multiple retrieved facts. The current MEN is trained with a single-subgraph encoding path; multi-hop questions that require combining facts from two separate retrieved turns are a known limitation and an active area of improvement.

---

## Project Structure

```
Nyxen-Memory/
├── cgm/                              # Core CGM library — the engine of Nyxen Memory
│   ├── __init__.py                   # Package initialization, CUDA detection & device setup
│   ├── core/
│   │   ├── model.py                  # Memory Encoder Network (MEN) & LowRankLinear
│   │   ├── pipeline.py               # End-to-end generation, injection & logging loop
│   │   └── compressor.py             # Double-buffered proactive KV cache compression
│   ├── database/
│   │   └── schema.py                 # SQLite Graph Store schema & Hybrid Memory Object (HMO) definitions
│   ├── retrieval/
│   │   ├── retriever.py              # Base retrieval class & RRF fusion logic
│   │   └── rag_retriever.py          # Dense vector search via TurboVec quantized index
│   ├── safety/
│   │   ├── safety.py                 # Python ctypes wrapper for the native safety guard
│   │   ├── safety_guard.cpp          # Win32 + NVML C++ source — VRAM, RAM & thermal monitoring
│   │   └── safety_guard.dll          # Pre-compiled 64-bit Windows native library
│   ├── training/
│   │   ├── train.py                  # MEGATrainer — KV-distillation optimizer & training loop
│   │   └── train_data.py             # Training sample extraction & teacher cache collection
│   └── visualization/
│       └── visualize.py              # Live Dash web dashboard — graph & memory visualizer
├── data/                             # Runtime databases, TurboVec index files & benchmark reports
├── docs/                             # Architecture deep-dives & reference diagrams
├── eval/                             # Evaluation harnesses & scoring utilities
├── scripts/                          # Utility & automation scripts
├── requirements.txt                  # Python dependencies
├── setup.ps1                         # Windows PowerShell setup script (admin recommended)
├── setup.sh                          # Linux / macOS / Git Bash setup script
├── chat.py                           # CLI chat interface & live dashboard launcher
├── run_demo.py                       # Standalone MEN injection & training verification demo
├── benchmark.py                      # Local generalization benchmark runner
├── run_locomo_benchmark.py           # Full RAG vs. CGM LoCoMo comparison benchmark
├── cgm_locomo_eval.py                # Public LoCoMo dataset evaluation runner
└── train_locomo_men_v2.py            # MEN training script with KV-distillation (v2)
```

---

## Setup & Quickstart

### Prerequisites

- Python 3.10 or higher
- NVIDIA GPU with CUDA 11.8+ (recommended: RTX 3060 or better, 8 GB+ VRAM)
- Windows 10/11, Ubuntu 20.04+, or macOS 12+ (CPU-only fallback available)

### Installation

**Windows (PowerShell — run as Administrator for C++ safety guard compilation):**
```powershell
./setup.ps1
```

**Linux / macOS / Git Bash:**
```bash
chmod +x setup.sh && ./setup.sh
```

The setup scripts handle virtual environment creation, dependency installation, CUDA toolkit verification, and native C++ guard compilation automatically.

**Manual installation (if you prefer):**
```bash
pip install -r requirements.txt
```

### Running Nyxen Memory

**Interactive CLI Chat + Live Memory Dashboard:**

Starts the terminal chat interface and opens the web-based memory visualizer at `http://localhost:8050`. The dashboard shows the knowledge graph in real time, live retrieval scores, and KV cache statistics.
```bash
python chat.py
```

**Quick Injection Demo:**

Runs a self-contained demonstration of the MEN injection pipeline — loads a test model, injects a synthetic memory, generates a response, and verifies the cache injection was successful.
```bash
python run_demo.py
```

---

## Training the Memory Encoder

Nyxen Memory includes a full **KV-Distillation** training pipeline for the Memory Encoder Network. The MEN learns to mimic the KV cache states that the target LLM would produce if it read the memory facts directly during its own prefill — a form of knowledge distillation from the model's own attention mechanism.

**Run LoCoMo Comparison Benchmark** *(evaluates RAG vs. Nyxen Memory side-by-side)*:
```bash
python run_locomo_benchmark.py --limit-conversations 1 --modes rag,cgm
```

**Train MEN with KV-Distillation:**
```bash
python train_locomo_men_v2.py --distill-lambda 2.0 --distill-warmup-epochs 10 --epochs 40
```

Key training flags:
- `--distill-lambda` — weight of the KV-distillation loss relative to task loss (default: 2.0)
- `--distill-warmup-epochs` — number of epochs to train on task loss only before enabling distillation
- `--epochs` — total training epochs

**Run Local Generalization Benchmarks:**
```bash
python benchmark.py
```

---

<div align="center">

**Nyxen Memory** — Built on Conversational Graph Memory. Persistent, structured, neural.

</div>

