# Literature Discovery Agent

A unified, agentic framework for discovering, triaging, and expanding scientific literature for chemical kinetics. This tool automates the "search and download" workflow by orchestrating multiple academic APIs and using LLMs for deep relevance filtering.

## 🚀 Key Features

- **Agentic Discovery**: Analyzes research mismatch reports to generate targeted search strategies.
- **Boolean Search Mode**: Uses "Exact Matching" (e.g., `"HOCHO" AND "oxidation"`) to preserve chemical formulas and technical terms.
- **Citation Snowballing**: Automatically expands discovery by following **bidirectional** citation links (`--snowball`).
    - **Forward**: Works that cite your high-priority results.
    - **Backward**: Works cited by your high-priority results (references).
- **Robust SI Mirroring**:
  - Prioritizes **Europe PMC** mirrors to bypass publisher anti-bot blocks (ACS, RSC, etc.).
  - Handles **Bulk SI ZIPs**: Detects and downloads binary ZIP responses from the EPMC API.
  - **Title Fallback**: Robustly matches preprints to their published versions by title if DOI search fails.
- **Multi-Source Power**: Integrates OpenAlex, Europe PMC, Crossref, and Semantic Scholar.
- **AI-Driven Triage**: Every discovered paper is analyzed by an LLM and assigned a priority:
    - ⚡ **High Priority**: Direct matches for your chemical system (e.g., HOCHO) that explicitly discuss reaction mechanisms, rate constants, or kinetic models. These papers are the only **seeds used for snowballing**.
    - 🔍 **Medium/Possible**: Relevant context (e.g., general burner studies, related species) but potentially lacks specific kinetic data or mechanism files.
- **Curated Results**: Automatically generates a "Manual Retrieval Queue" for paywalled content.

## 🔄 Core Workflow

The agent follows a multi-stage process to ensure high-recall and high-precision discovery:

1.  **Input Orchestration**: `runner.py` accepts natural language or raw keywords.
2.  **Parallel Discovery**: `RegistryBuilder` queries multiple APIs (`clients.py`) using **Boolean Exact Match** logic for chemical formulas.
3.  **Intelligent Enrichment**: Every candidate is cross-referenced with **Europe PMC**. If a match is found (by DOI or Title Fallback), SI metadata and PMCID links are attached.
4.  **AI-Driven Triage**: An LLM evaluates each paper's title and abstract for specific kinetics relevance, assigning priority and automated tags.
5.  **Citation Snowballing**: For "High Priority" hits, the agent follows the citation graph (`--snowball`) to find newer works that cite your target discoveries.
6.  **Robust SI Retrieval**:
    *   Resolves `PMC` identifiers to direct mirror links.
    *   Detects binary **Bulk ZIPs** and downloads them automatically.
    *   Downloads are saved with unique IDs to the nested `si_files/` folder.
7.  **Consolidated Reporting**: Generates a unified JSON/YAML registry and a human-readable curated summary.

## 🧠 AI Integration (Optional)

While the core discovery is driven by your manual keywords in `query.txt` (or CLI input), the system can optionally use an LLM (`--mode agent`) to act as an **Expert Critic**:

- **Role**: Performs deep triage on every candidate paper found by the search APIs.
- **Action**: It reads the title and abstract, evaluates them against expert chemical kinetics criteria, and:
    - Decides whether to **Keep** or **Discard** the paper based on actual kinetics content.
    - Assigns a **Priority** (High/Medium).
    - Writes a **Match Reason** explaining why the paper was selected.
    - Selects the most reliable papers to serve as **Citation Snowballing seeds**.

*If run in `--mode manual`, the system skips the LLM and keeps all search results.*

## 📂 Module Responsibilities

The project is structured as a modular package in `accelerator/`:

- **`runner.py`**: **Entry Point**. Handles CLI arguments, manages the output directory structure, and triggers the `RegistryBuilder`.
- **`registry_builder.py`**: **The Engine**. Implements the discovery loops, the LLM filtering logic, snowball expansion, and the SI resolution/download coordinator.
- **`clients.py`**: **The Eyes**. Contains unified wrappers for academic APIs. It handles data normalization and EPMC-specific mirror lookups.
- **`models.py`**: **The Schema**. Defines the `PaperRecord` and `SearchQuery` objects, ensuring data consistency across all discovery stages.
- **`utils.py`**: **The Tools**. General-purpose utilities for robust HTTP retries, LLM interactions (via LiteLLM/OpenAI), and safe file I/O.

## 🛠 Usage

Run the agent via **`runner.py`**:

```bash
# Full workflow: Search, Filter, Expand (Snowball), and Download SI
python -m accelerator.runner "HOCHO pyrolysis" --snowball --download-si --max-results 20
```

### Discovery Modes
- **Agentic (Default)**: Best for complex mismatch sentences.
- **Manual (`--mode manual`)**: Direct keyword searches.
- **File-based**: Pass a `.txt` (one query per line) or `.json` file as input.

## 📊 Outputs

Results are organized into timestamped subfolders under `outputs/run_[query]_[timestamp]/`:

- **`si_files/`**: Contains the actual supplementary materials (ZIPs, PDFs, mechanisms) that were successfully mirrored and downloaded.
- **`paper_registry.json` / `.yaml`**: The complete structured database of discovery. Contains metadata, AI-generated match reasons, priority scores, and provenance for every paper found.
- **`curated_search_results.txt`**: A "Quick Look" human-readable summary. Best for quickly scanning high-priority hits and their AI-summarized value.
- **`manual_retrieval_queue.yaml`**: A high-priority "To-Do" list. This contains papers that matched your research perfectly but could not be downloaded automatically (e.g., behind a strictly enforced paywall).

## ⚙️ Installation

```bash
pip install requests pyyaml pydantic openai python-dotenv duckduckgo-search
```
Ensure your `.env` file contains:
- `DEEPSEEK_API_KEY`
- `S2_API_KEY` (Optional)
