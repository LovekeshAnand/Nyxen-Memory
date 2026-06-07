# Contributing to Conversational Graph Memory (CGM-RAG)

Thank you for your interest in contributing to CGM-RAG. This document provides guidelines for setting up your environment, coding standards, and submitting pull requests.

---

## Code of Conduct

We expect all contributors to adhere to standard professional code of conduct rules, treating all community members with respect and keeping collaboration constructive.

---

## Getting Started

1. **Fork the Repository:** Create a personal fork of the repository on GitHub.
2. **Clone the Fork:** Clone your fork to your local workstation.
3. **Run Setup:** Execute the setup scripts to install dependencies and configure local models:
   * **Windows PowerShell:** `./setup.ps1`
   * **Linux/Git Bash:** `./setup.sh`
4. **Create a Branch:** Create a branch for your work:
   ```bash
   git checkout -b feature/your-feature-name
   ```

---

## Coding Standards

To maintain a clean and maintainable codebase, please follow these conventions:

### Python Conventions
* **PEP 8:** Follow standard PEP 8 formatting guidelines.
* **Type Hinting:** Use type hints for all function signatures and complex variables.
* **Docstrings:** Provide descriptive docstrings for all public modules, classes, and methods.
* **Compatibility:** Force standard stream reconfigurations (`sys.stdout.reconfigure(encoding='utf-8')`) in interactive scripts to ensure compatibility with Windows CP1252 consoles. Do not output raw Unicode emojis to the standard console.

### Database and Indexing
* **ID Hashing:** When adding vectors to the retrieval index, use the 64-bit ID hashing scheme (`encode_id`) to prevent ID collisions between different conversations in the shared vector index.
* **Persistence:** Always release database locks cleanly using the SQLiteGraphStore lock manager.

### Hardware Safety Guards
* **Lock Management:** Wrap PyTorch model executions or GPU operations with the `GPULockManager` context manager to maintain thread safety.
* **VRAM Guard:** Ensure the `GPUMemoryGuard` is queried before allocating large attention matrices or compiling new model adapters.

---

## Running Verification

Before submitting a pull request, verify that your changes do not introduce regressions:

1. **Verify Basic Operations:**
   Run the seeding, training, and inference verification script:
   ```bash
   python run_demo.py
   ```
2. **Execute Performance Benchmarks:**
   Run the benchmark suite to ensure that KV cache injection and compressor latency metrics remain stable:
   ```bash
   python benchmark.py
   ```
3. **Verify Interactive Visualizer:**
   Launch the dashboard and interact with the CLI shell:
   ```bash
   python chat.py
   ```

---

## Submitting a Pull Request

1. **Commit Changes:** Write clear, concise commit messages.
2. **Push to GitHub:** Push your changes to your fork.
3. **Submit PR:** Open a Pull Request against the `main` branch of the upstream repository.
4. **Description:** Describe the problem solved, changes introduced, and how the changes were verified.
