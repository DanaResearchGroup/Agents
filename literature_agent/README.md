# Paper Discovery and Triage Module

This module implements Step 1 of Lane 1 in the literature-model evaluation workflow. It searches for relevant papers, checks open access status, and triages them into a registry for downstream use.

## Features
- Multi-source search (Semantic Scholar, Crossref, Europe PMC)
- Hybrid Search Agent with agentic web search fallback for missing SI
- Heuristic-based relevance scoring and tagging
- Deep Supplementary Information (SI) discovery
- Optional automated downloading for OA SI files
- YAML/JSON registry export
- Manual retrieval queue generation for high-interest paywalled papers

## Installation
Ensure you have the required dependencies:
```bash
pip install requests pyyaml pydantic
```

## Usage

### Method 1: Registry Builder (CLI)
Best for building a comprehensive paper registry with metadata, scoring, and triage.
```bash
python -m accelerator.cli --query "HOCHO pyrolysis mechanism" --max-results 10
```

### Method 2: Workflow Accelerator (Runner)
Best for a step-by-step agentic workflow including intake, discovery, and automated SI downloading.
```bash
python -m accelerator.runner "At 900 K and 1 bar JSR, the model overpredicts CO and slightly underpredicts CO2 for HOCHO oxidation."
```

## Outputs
Results are saved to the `outputs/` directory (customizable via `--outdir`):
1. `paper_registry.yaml`: Full registry of discovered papers.
2. `manual_retrieval_queue.yaml`: High-interest papers requiring manual download.
3. `paper_registry.json`: Machine-readable version of the registry.

## Design Philosophy
This module focuses on **triage and retrieval planning**. It identifies what is available and what needs manual attention before any PDF parsing or simulation begins.
