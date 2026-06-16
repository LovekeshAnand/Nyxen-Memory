# Conversational Graph Memory (CGM-RAG)

Conversational Graph Memory (CGM) is a high-performance Retrieval-Augmented Generation (RAG) and neural context injection architecture that equips Large Language Models with persistent, structured long-term memory. By representing dialogue histories as dynamic, entity-linked knowledge graphs, CGM avoids the token accumulation overhead of prompt-stuffing. It utilizes a trainable **Memory Encoder Network (MEN)** to project retrieved graph concepts directly into the Key-Value (KV) cache layers of frozen local generative models, or falls back to chronologically sorted, distilled text-based RAG context for local APIs.

The framework is built from the ground up for edge workstations (such as laptops equipped with NVIDIA RTX GPUs). It features native C++ concurrency locks, GPU memory guards, and proportional thermal pacing to prevent Out-of-Memory (OOM) crashes and system degradation.

---

## Technical Architecture

The Conversational Graph Memory (CGM) pipeline is designed as an alternative to context stuffing. Rather than expanding prompt windows with historical text, CGM extracts structured knowledge graph triples from conversations, retrieves them semantically, projects them into virtual keys/values, and injects them directly into the LLM self-attention cache.

![Conversational Graph Memory Architecture](docs/cgm_llm_integration.png)

### Core Architectural Flow

```text
  [ User Dialogue Turn ] ──(Closed-Loop Extraction)──> [ SQLite Graph Store ]
                                                              │
                                                              ▼ (Top-k Subgraphs)
  [ User Active Query  ] ──────(TurboVec Index)──────> [ Dense Feature Construction ]
                                                              │
                                                              ▼
                                                   [ Memory Encoder Network ]
                                                              │
                                                              ▼ (Projected K & V)
                                                   [ Semantics-Aware Routing ]
                                                              │
                                                              ▼ (RoPE & Magnitude Match)
                                                   [ Cache Injection (past_kv) ]
                                                              │
                                                              ▼
                                                   [ Autoregressive Decoding ]
```

---

## Detailed Run-Time Execution Workflow

When you attach CGM to an LLM and start conversing, the system processes each turn through a six-stage loop:

1. **User Query & Vector Search:**
   * You input a query: *"What is my database password?"*
   * The query is embedded (384-dim) via SentenceTransformer (`all-MiniLM-L6-v2`).
   * The system queries the quantized **TurboVec** index under strict `tenant_id` and `user_id` filters to retrieve the top-k most semantically similar historical turns (e.g., Turn 14).

2. **Graph-Neighbor Concept Retrieval:**
   * The system looks up the retrieved turn IDs in the **SQLite Graph Store**.
   * It performs a **1-hop graph neighbor chain query** (forward and backward edges) to pull connected nodes. This ensures that negation statements (*"No, let's not use Redis"*), temporal updates (*"Change database to ClickHouse"*), and correction context are fetched together, rather than as isolated keywords.

3. **Neural Projection via the Memory Encoder Network (MEN):**
   * The retrieved subgraphs are represented as dense structural vectors (768-dim) in the form `[enc(Subject || Predicate); enc(Object)]`.
   * These structural vectors are passed to the **Memory Encoder Network (MEN)**. 
   * The MEN maps these vectors into the precise head-dim and head-count dimensions of the target LLM's attention layers.

4. **Semantics-Aware Routing & RoPE Alignment:**
   * The projected Key-Value states pass through the **Semantics-Aware KV Cache Routing (SA-KVR)** gating network, which scales head-wise and layer-wise activations based on a learned middle-layer prior.
   * Key states are rotated via **Rotary Position Embeddings (RoPE)** to align with the target LLM's positional encoding math.

5. **Dynamic KV Cache Injection:**
   * The projected and calibrated key-value tensors are prepended directly to the LLM's dynamic `past_key_values` cache as **virtual memory slots** (virtual tokens).
   * This bypasses the LLM's **Prefill Phase**. The prompt containing only the user query (~15 tokens) is tokenized, and the self-attention mask is padded to allow rectangular attention over both the virtual memory slots and the prompt tokens.

6. **Episodic Logging & Triple Extraction:**
   * The LLM generates the response immediately.
   * The user query and the assistant's reply are saved as a new turn in SQLite.
   * An NLP parser extracts new Subject-Predicate-Object triples from the interaction, seeding them back into the SQLite Graph Store.
   * The new turn text is embedded and indexed in TurboVec for future reference.

---

## The Memory Encoder Network (MEN) Architecture

The **Memory Encoder Network (MEN)** is the core neural translator of CGM. It maps symbolic knowledge representations into the latent key-value manifolds of the LLM.

```text
                        [ Input: Triples Vector (768-dim) ]
                                         │
                                         ▼
                           [ LowRankLinear Projector ]
                                  (Rank = 128)
                                 /            \
                                /              \
                        [ Key Path ]        [ Value Path ]
                             │                     │
                             ▼                     ▼
                       [ LayerNorm ]         [ LayerNorm ]
                     (Magnitude Match)     (Magnitude Match)
                             │                     │
                             ▼                     ▼
                        [ SA-KVR ]            [ SA-KVR ]
                     (Layer Gating)        (Layer Gating)
                             │                     │
                             ▼                     ▼
                          [ RoPE ]            [ Inject ]
                             │                     │
                             ▼                     ▼
                     [ Injected past_key_values Cache Tensors ]
```

### 1. Parameter Efficiency via LoRA-MEN (`LowRankLinear`)
To prevent the projection weights from overfitting on small conversation histories, MEN utilizes low-rank bottleneck projections instead of fully dense layers. 

For a projection matrix $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$, we decompose it into two low-rank matrices $A \in \mathbb{R}^{r \times d_{\text{in}}}$ and $B \in \mathbb{R}^{d_{\text{out}} \times r}$, where the rank $r \ll \min(d_{\text{in}}, d_{\text{out}})$:

$$W = B \cdot A$$

For a rank $r = 128$, this reduces the parameter footprint of the projection layer by **over 70%** (reducing MEN parameter count from 11.14M to 2.87M on Qwen 0.5B), regularizing the network to learn smooth semantic-to-attention mappings.

### 2. Semantics-Aware KV Cache Gating (SA-KVR)
A routing gating network predicts a scalar gating weight $g_l^h \in [0, 1]$ for each layer $l$ and attention head $h$ based on the input triple's embedding vector $x$:

$$g_l^h = \sigma\left( W_g \cdot x + b_g \right)$$

The projected key $K_{\text{proj}}$ and value $V_{\text{proj}}$ are scaled before injection:

$$K_{\text{routed}} = g_l^h \cdot K_{\text{proj}}, \quad V_{\text{routed}} = g_l^h \cdot V_{\text{proj}}$$

### 3. Layer-Depth Routing Prior Constraint (Gaussian Routing)
Empirical analysis of transformer activations indicates that factual knowledge is primarily concentrated in the middle layers, whereas tasks and contextual instructions reside in the later layers. 

To bias the routing network to inject memories where they are most relevant, we apply a layer-depth routing prior regularization. The routing gates are constrained toward a Gaussian distribution centered on the middle layers:

$$\mathcal{L}_{\text{routing\_prior}} = \frac{1}{L} \sum_{l=0}^{L-1} \left( \bar{g}_l - P(l) \right)^2$$

Where the target distribution $P(l)$ is defined as:

$$P(l) = \exp\left( -\frac{(l - \mu)^2}{2\sigma^2} \right)$$

For `Qwen2.5-0.5B-Instruct` (24 layers), we set the mean layer index $\mu = 11.5$ and standard deviation $\sigma = 4.0$. This guides the routing network to inject facts primarily within layers 8–16.

### 4. KV-Distillation Loss & Variance Normalization
The MEN is trained using a multi-task loss function combining next-token Cross-Entropy generation loss and Mean Squared Error (MSE) KV-distillation:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{generation}} + \lambda_{\text{distill}} \mathcal{L}_{\text{distill}}$$

The distillation loss forces the MEN projections to match the target LLM's own internal KV states when reading the triple's natural language representation:

$$\mathcal{L}_{\text{distill}} = \frac{1}{L} \sum_{l=0}^{L-1} \left( \frac{\text{MSE}(K_{\text{men}}^{(l)}, K_{\text{teacher}}^{(l)})}{\text{Var}(K_{\text{teacher}}^{(l)})} + \frac{\text{MSE}(V_{\text{men}}^{(l)}, V_{\text{teacher}}^{(l)})}{\text{Var}(V_{\text{teacher}}^{(l)})} \right)$$

Where $\text{Var}(K_{\text{teacher}}^{(l)})$ is the teacher key variance computed at layer $l$, which normalizes scale differences across different depths. Before computing MSE, the teacher's key states are **unrotated** to reverse the target LLM's Rotary Position Embedding (RoPE) functions, ensuring alignment with the static projections of the MEN.

### 5. Dynamic Magnitude Calibration
To match the magnitude of the LLM's activations, the MEN features custom LayerNorm scales calibrated dynamically before optimization. The normalization parameters are initialized to match the mean and standard deviation of the teacher model's KV states across the training set:

$$\gamma_l = \text{Std}(KV_{\text{teacher}}^{(l)}), \quad \beta_l = \text{Mean}(KV_{\text{teacher}}^{(l)})$$

This ensures that the injected key-value matrices do not cause attention weight explosions or numerical instability at early epochs.

---

## Folder Structure

The repository is organized as a modular, importable library package:

```text
Nyxen-Memory/
├── cgm/                              # Primary library package
│   ├── __init__.py                   # Package initialization, CUDA detection
│   ├── core/                         # Core execution logic
│   │   ├── __init__.py
│   │   ├── model.py                  # MEN and LowRankLinear architectures
│   │   ├── pipeline.py               # E2E generation and injection loops
│   │   └── compressor.py             # Double-buffered KV cache compression
│   ├── database/                     # Relational persistence
│   │   ├── __init__.py
│   │   └── schema.py                 # SQLite Graph Store & HMO definitions
│   ├── retrieval/                    # Retrieval layers
│   │   ├── __init__.py
│   │   ├── retriever.py              # Base retrieval class
│   │   └── rag_retriever.py          # Dense vector search using TurboVec
│   ├── safety/                       # Hardware and software guards
│   │   ├── __init__.py
│   │   ├── safety.py                 # Python ctypes safety wrapper
│   │   ├── safety_guard.cpp          # Win32 Memory & NVML VRAM C++ source
│   │   └── safety_guard.dll          # Pre-compiled 64-bit C++ library
│   ├── training/                     # Optimization and alignment
│   │   ├── __init__.py
│   │   └── train.py                  # MEGATrainer optimizer with KV-distillation
│   └── visualization/                # Web visualizer dashboard
│       ├── __init__.py
│       └── visualize.py              # Live visualizer server and endpoints
├── data/                             # Databases, TurboVec files, and reports
├── docs/                             # Architecture diagrams and deep-dives
├── requirements.txt                  # Python dependencies
├── chat.py                           # CLI chat interface and dashboard boot
├── run_demo.py                       # Training and injection verification demo
├── benchmark.py                      # Core evaluation benchmark runner
└── cgm_locomo_eval.py                # Public LoCoMo benchmark runner
```

---

## Empirical Benchmarks

### 1. Generalization Stress Test (`benchmark.py`)
This test evaluates temporal updates (e.g., changing DB configuration from ClickHouse to DuckDB) and negations (e.g., *"We decided not to use Redis"*).

| Base Model | Approach | Input Context / Virtual Tokens | Peak VRAM | Factual Recall | Negation / Temporal Generalization |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Qwen2.5-0.5B** | A. Context Stuffing (Baseline) | 523 / 0 | 1190.2 MB | 61.0% | 50.0% |
| | B. Standard RAG | 221 / 0 | 1153.6 MB | 53.7% | **66.7%** |
| | C. CGM (No Routing - Injected) | **20 / 6** | **1143.2 MB** | 0.0% | 0.0% |
| | D. CGM + SA-KVR (Routed - Injected) | **20 / 6** | **1143.2 MB** | **43.9%** | 16.7% |
| | E. CGM + Routing + Comp. (Ours) | **20 / 100** | **1143.2 MB** | **43.9%** | 16.7% |

* **Context Savings:** CGM + SA-KVR achieves a **96.2% reduction** in prompt token ingestion compared to context stuffing, while keeping latency minimal.
* **Unrouted Failure (Approach C):** Without the Semantics-Aware Routing network, un-gated KV projections collapse to 0% recall because out-of-distribution KV cache magnitudes disrupt the LLM's self-attention patterns.

### 2. Standard Public Benchmark: LoCoMo (`cgm_locomo_eval.py`)
To validate the model's capacity on industry-grade tasks, we evaluated the pipeline on the **LoCoMo** long-dialogue conversational dataset using `Qwen/Qwen2.5-0.5B-Instruct` on a local workstation GPU (4GB VRAM limit). 

The results are saved to [data/locomo_benchmark_rank128.json](file:///d:/Nyxen-Memory/data/locomo_benchmark_rank128.json):

| Approach | Avg Input Tokens | Factual Recall | Avg Latency | Notes / Performance Analysis |
| :--- | :---: | :---: | :---: | :--- |
| **Approach A (Context Stuffing)** | 14,753.4 | **0.0%** | 4.437s | **CUDA Out of Memory (OOM).** Attempting to load 35 dialogue sessions (~9,000+ tokens) exceeds the 4GB workstation GPU limit, causing crashes. |
| **Approach B (Standard RAG)** | 18.4 | **50.0%** | **1.324s** | Standard RAG retrieves the top-2 matching sessions and feeds them as text context. |
| **Approach D (CGM + SA-KVR)** | **13.4** | **53.3%** | 1.404s | **Ours.** Achieves the highest recall while reducing input prompt size by **27.1% vs RAG** and **99.9% vs Stuffing** without OOM errors. |

---

## Setup and Quickstart

Configure your environment using the setup scripts:

### Windows PowerShell (Admin Mode Recommended)
```powershell
./setup.ps1
```

### Git Bash / Linux / macOS
```bash
chmod +x setup.sh
./setup.sh
```

### Run Commands

* **CLI Chat and Live Dashboard:** Starts the interactive terminal chat and opens the web visualizer dashboard at `http://localhost:8050/`:
  ```bash
  python chat.py
  ```
* **Verify Cache Injection & MEN Training:**
  ```bash
  python run_demo.py
  ```
* **Run Local Generalization Benchmarks:**
  ```bash
  python benchmark.py
  ```
* **Run LoCoMo Public Benchmark:**
  ```bash
  python cgm_locomo_eval.py --rank 128
  ```
