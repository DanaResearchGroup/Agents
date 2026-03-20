# Paper Discovery and Triage Module

This module implements Step 1 of Lane 1 in the literature-model evaluation workflow. It searches for relevant papers, checks open access status, and triages them into a registry for downstream use.

## Features
- Multi-source search (Crossref, OpenAlex)
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

### Free-text Query
```bash
python -m accelerator.cli --query "HOCHO pyrolysis mechanism" --max-results 10
```

### Structured Search Object (JSON file)
Create a file named `search.json`:
```json
{
  "topic": "HOCHO pyrolysis",
  "keywords": ["formic acid", "pyrolysis", "mechanism"],
  "year_min": 2010,
  "max_results": 20
}
```
Run with:
```bash
python -m accelerator.cli --query search.json
```

## Outputs
Results are saved to the `outputs/` directory (customizable via `--outdir`):
1. `paper_registry.yaml`: Full registry of discovered papers.
2. `manual_retrieval_queue.yaml`: High-interest papers requiring manual download.
3. `paper_registry.json`: Machine-readable version of the registry.

## Design Philosophy
This module focuses on **triage and retrieval planning**. It identifies what is available and what needs manual attention before any PDF parsing or simulation begins.
