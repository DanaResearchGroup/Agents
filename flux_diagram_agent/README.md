# Flux Diagram Extraction Agent MVP

A Python-based pipeline for detecting and extracting pathway information from flux diagrams in computational chemistry literature.

## Features
- **PDF Parsing**: Robust text and figure extraction using PyMuPDF.
- **Candidate Filtering**: Keyword-based identification of relevant figures.
- **DeepSeek Integration**: LLM-powered figure classification and structured data extraction.
- **Automated Reporting**: Generates both human-readable Markdown and machine-readable JSON reports.

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
python main.py --pdf path/to/paper.pdf --outdir outputs/
```

## General Workflow
The agent follows a systematic pipeline to extract structured data from PDF literature:
1. **Parsing**: Uses PyMuPDF to extract text and identify figure captions. Entire pages containing potential figures are rendered as images.
2. **Filtering**: A keyword-based heuristic scores figures to identify "candidates" (e.g., matching "flux", "reaction path").
3. **Classification**: Uses a Vision-Language Model (DeepSeek) to confirm if a candidate is actually a flux diagram.
4. **Interpretation**: For confirmed flux diagrams, the LLM extracts structured kinetic data (species, conditions, pathways).
5. **Aggregation**: Synthesizes findings into a paper-level summary with recommended actions.
6. **Reporting**: Generates final JSON and Markdown reports.

## Module Responsibilities
- `agent/engine.py`: Core processing logic (Parsing, Filtering, Aggregation, and Reporting).
- `agent/brain.py`: AI task logic (LLM Provider, Figure Classification, and Flux Interpretation).
- `agent/prompts.py`: Centralized LLM prompt templates.

## Project Structure
- `agent/`: Core logic modules
- `main.py`: CLI entry point
- `prompts/`: LLM prompt templates
- `outputs/`: Default output directory
