# Flux Diagram & Sensitivity Analysis Agent

A powerful Python-based pipeline for detecting, extracting, and prioritizing pathway and sensitivity information from chemical kinetics literature.

## Features
- **PDF Parsing & Smart Cropping**: Robust figure extraction using PyMuPDF with automated **Subplot Isolation** (Smart Cropping) for focused vision parsing.
- **Sensitivity Analysis Integration**: Automatically detects and interprets sensitivity bars/charts to extract key reaction coefficients.
- **Multi-Provider Support**: Supports **DeepSeek** (Reasoner/Chat) and **Zhipu AI (GLM-4V)** for state-of-the-art vision-based diagram interpretation.
- **Targeted Reaction Ranking**: Groups and ranks reactions by their influence on specific observables (e.g., HOCHO, CO2, laminar burning velocity).
- **Automated Reporting**: Generates timestamped and organized Markdown/JSON reports with embedded figure crops.

## Installation
1. Clone the repository and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Configure your `.env` file with your API keys:
   ```bash
   DEEPSEEK_API_KEY=your_key
   GLM_API_KEY=your_key
   ```

## Usage
The agent automatically derives the output directory from the PDF name and adds a timestamp for organization.

### Basic Usage (DeepSeek)
```bash
python main.py --pdf path/to/paper.pdf
```

### Advanced Usage (GLM-4V Vision)
Highly recommended for best precision in reading reaction equations from charts.
```bash
python main.py --pdf tests/SI.pdf --provider glm --model glm-4v-plus
```

### Configuration Flags
- `--pdf`: (Required) Path to the PDF paper.
- `--provider`: `deepseek` (default) or `glm`.
- `--model`: Specific model name (e.g., `glm-4v-plus`, `deepseek-reasoner`).
- `--outdir`: Custom output folder (defaults to `outputs/[paper_name]_[timestamp]/`).

## Project Structure
- `agent/engine.py`: Core processing (Parsing, Smart Cropping, Aggregation, Reporting).
- `agent/brain.py`: Multi-provider LLM logic and task-specific interpretation.
- `agent/prompts.py`: Optimized prompts for kinetics data extraction.
- `main.py`: CLI entry point.

## Workflow
1. **Parsing**: Identifies figure captions and extracts high-resolution horizontal "strips" (subplots) for each figure.
2. **Filtering**: Scores figures via keywords to identify flux or sensitivity candidates.
3. **Classification**: LLM confirms the diagram type using isolated subplot images.
4. **Interpretation**: Extracts species, conditions, and prioritized reaction lists with sensitivity coefficients.
5. **Aggregation**: Synthesizes a unified, target-grouped reaction ranking across the entire paper.
6. **Reporting**: Generates visual Markdown summaries and machine-readable JSON results.
