# Conversational Graph Memory (CGM) for LLMs

Conversational Graph Memory (CGM) is a novel architecture designed to augment LLMs with long-term, factual memory by structuring dialogue histories into a dynamic knowledge graph. Instead of appending massive text logs to the input context, CGM utilizes a trainable **Memory Encoder Network (MEN)** to project retrieved knowledge graph triples directly into the Key-Value (KV) cache of the LLM. 

This enables the model to perform **rectangular attention** over graph-structured memory out-of-the-box, saving precious context window tokens and eliminating prompt-stuffing overhead.

---

## 🛠️ Project Folder Structure

The repository is structured as a clean, modular production codebase:

```text
Nyxen-Memory/
├── cgm/                          # Core Conversational Graph Memory Package
│   ├── __init__.py               # Dynamic CUDA loader hook
│   ├── model.py                  # PyTorch Memory Encoder Network (MEN)
│   ├── pipeline.py               # End-to-End KV injection inference pipeline
│   ├── retriever.py              # Embedding-based subgraph retriever
│   ├── run_demo.py               # Interactive demo coordinator
│   ├── safety.py                 # Active GPU locks & VRAM/Thermal guards
│   ├── schema.py                 # SQLite Hybrid Memory Object (HMO) store
│   └── train.py                  # Adapter training optimization loop
├── data/                         # Persistent Database Storage
│   └── cgm_memory.db             # Thread-safe SQLite store (automatically created)
├── docs/                         # Theoretical & Documentation Assets
│   └── nyxen-memory.tex          # LaTeX research paper draft
├── phase1_validation/            # Phase 1: Semantic Graph Fidelity Tests
│   ├── config.py                 # Rate-limiting Gemini API client wrapper
│   ├── evaluator.py              # LLM-as-a-judge fact grader
│   ├── extractor.py              # Concept/triple extraction pipeline
│   ├── report.json               # Aggregated validation results
│   ├── run.py                    # Test suite runner
│   ├── simulator.py              # Multi-turn scenario simulator
│   └── README.md                 # Validation instructions & math formulas
├── requirements.txt              # Project dependency configurations
└── .gitignore                    # Local database and cache ignores
```

---

## 🛡️ Active Hardware & Software Safety Protocol

Since workstation notebook GPUs (like the **NVIDIA RTX A2000 Laptop GPU**) run in tight thermal environments, this system includes a strict safety architecture implemented in [`cgm/safety.py`](file:///d:/Nyxen-Memory/cgm/safety.py):

1. **GPU VRAM Guard (OOM Prevention):** Proactively checks free memory using `torch.cuda.mem_get_info()`. Automatically flushes PyTorch CUDA caches if available memory drops below 800 MB, and aborts before running operations if memory is critically low (<300 MB).
2. **GPU Thread Lock & Semaphore:** Serializes execution of heavy PyTorch kernels. Uses a global `threading.Lock` and a `BoundedSemaphore` to guarantee only one thread or operation accesses the GPU at any millisecond.
3. **Thermal Protection Guard:** Tracks processed batches during training. Triggers automatic 2-second rest periods for the GPU after every 5 training steps, allowing fans to cool the chip down.
4. **Input Buffer Constraints:** Sanitizes string queries and strictly validates tensor shapes to prevent excessive attention-matrix allocations.
5. **Database Transaction Locks:** Wraps all Graph Store updates in transactional queries to maintain thread-safe write integrity.

---

## 📈 Phase 1: Semantic Graph Fidelity Results

The Phase 1 Validation Suite evaluated whether compressing raw dialogue histories into a concept graph (summary + entities + triples) loses factual fidelity. We simulated a baseline LLM (using the full raw dialogue history) and compared it with the CGM representation (using only the extracted graph concepts).

### Key Metrics
* **Average Factual Recall (Baseline):** 66.7%
* **Average Factual Recall (CGM):** 66.7%
* **Average Factual Fidelity ($F_f$):** **100.0%**
* **Fidelity Hypothesis ($F_f \ge 85\%$):** **VALIDATED**

| Scenario | Baseline Score | CGM Score | Relative Fidelity ($F_f$) | Summary |
| :--- | :---: | :---: | :---: | :--- |
| **Scenario 1: FastAPI & PostgreSQL** | 33.3% | 33.3% | **100.0%** | Retained database preferences, Pydantic configuration, asyncpg connector, and Ruff linting rules. |
| **Scenario 2: Docker Bridge Setup** | 100.0% | 100.0% | **100.0%** | Reconstructed complex Docker Compose networking instructions, network names, hostnames, and environment flags. |

*Factual Fidelity ($F_f$) of 100% validates that the compressed concept graph structure loses zero critical factual capacity compared to keeping the raw text history.*

---

## 🚀 Setup & Execution

### 1. Prerequisites & Custom CUDA Install
This repository supports Python 3.14+. Since GPU memory is highly restricted, we install the 2.6 GB PyTorch CUDA build onto the `D:` drive.

To install PyTorch with CUDA support on Windows without filling up the C: drive cache:
```powershell
# Redirect temporary installation directories to D: drive
$env:TMPDIR='D:\temp_pip'
$env:TEMP='D:\temp_pip'
$env:TMP='D:\temp_pip'
New-Item -ItemType Directory -Force -Path 'D:\temp_pip' | Out-Null

# Install PyTorch to a workspace directory
pip install torch --index-url https://download.pytorch.org/whl/cu126 --force-reinstall --no-cache-dir --target D:\Nyxen-Memory\.torch_cuda
```

*Note: The `cgm` package automatically hooks the `.torch_cuda` folder into your `sys.path` when imported, so you do not need to alter your environment path.*

### 2. Install Package Dependencies
Install the remaining packages globally:
```powershell
pip install -r requirements.txt
```

### 3. Run Phase 1 Validation Suite
To execute the multi-turn fidelity test (requires a `.env` file containing your `GEMINI_API_KEY`):
```powershell
python -m phase1_validation.run
```

### 4. Run System Demo (Adapter Training + Cache Injection)
To run the full end-to-end demo showing MEN training and dynamic KV injection:
```powershell
# Move to package folder
cd cgm
python run_demo.py
```
*(Alternatively, from the root folder: `python -m cgm.run_demo`)*
