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

## Project Structure
- `agent/`: Core logic modules
- `main.py`: CLI entry point
- `prompts/`: LLM prompt templates
- `outputs/`: Default output directory
