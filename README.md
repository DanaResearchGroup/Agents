# chem-agent

A multi-agent framework for evaluating and improving chemical kinetic models. It compares a user's kinetic model against published literature mechanisms and tests targeted reaction modifications — all backed by Cantera simulations and structured LLM reasoning.

## How it works

The system runs two independent analysis paths against user-provided experimental data:

- **Path 1 (Literature Evaluation)** compares the original model against a published mechanism. Models run in complete isolation — no reactions are ever merged.
- **Path 2 (Targeted Improvements)** mines candidate reactions from papers and tests each one as a modification to the original model.

```
User provides:
  original kinetic model       (Cantera YAML or Chemkin)
  experimental data            (YAML with conditions & measurements)
  paper                        (DOI, PDF upload, or search query)
           |
           v
     +-------------+
     | Orchestrator |
     +------+------+
            |
     +------+------+
     |             |
  Path 1        Path 2
  Literature    Targeted
  Evaluation    Improvements
     |             |
     +------+------+
            |
            v
     report.yaml
     (MAE per model, per condition)
```

## Pipeline workflow

```
chem-agent run
|
+-- 1. INGEST PAPER
|   +-- DOI --> OpenAlex/Crossref --> download PDF
|   +-- file: --> local PDF path
|   +-- search: --> OpenAlex --> user confirms --> download PDF
|       |
|       v
|   parse_pdf() --> PaperDocument
|       +-- pages: list[PageText]
|       +-- captions: list[FigureCaption]
|       +-- tables: list[ParsedTable]
|
+-- 2. READ PAPER (LLM)
|   +-- PaperReaderAgent
|       +-- tools: get_abstract, list_sections, get_section,
|       |          search_text, list_tables, get_table,
|       |          list_figures, get_figure_caption
|       +-- output: JSON --> PaperSummary
|       +-- fallback: markdown --> regex extraction --> PaperSummary
|           (reactor_types, T/P ranges, species, tables, figures)
|
+-- 3. PATH 1 -- Literature Model Evaluation
|   |
|   +-- 3a. Obtain literature mechanism
|   |   +-- --literature-model provided? --> use directly
|   |   +-- else: scan paper SI --> ChemkinConverter (T3) --> Cantera YAML
|   |       +-- extract_rates() --> {reaction: {A, n, Ea}}
|   |
|   +-- 3b. ModelIsolationValidator  <-- HARD STOP if shared reactions
|   |   +-- normalize equations --> set intersection --> must be empty
|   |
|   +-- 3c. Extract simulation conditions
|   |   +-- Deterministic pipeline (always runs first)
|   |   |   +-- evidence --> candidates --> scenarios --> plans
|   |   |
|   |   +-- ConditionReasoningAgent (LLM, validates plans)
|   |   |   +-- tools: get_evidence, validate_condition,
|   |   |   |          get_paper_summary, get_page
|   |   |   +-- output: JSON array --> list[SimConditions]
|   |   |
|   |   +-- Fallback: if agent returns 0 --> use deterministic plans
|   |
|   +-- 3d. Post-process conditions
|   |   +-- deduplicate
|   |   +-- filter by reactor family (match experimental data)
|   |
|   +-- 3e. Simulate (for each condition)
|   |   +-- run_simulation(literature_model, aliases=literature_aliases)
|   |   +-- run_simulation(original_model, aliases=species_aliases)
|   |       |
|   |       v dispatch by reactor_type
|   |       +-- shock_tube --> IDT (ignition delay time)
|   |       +-- jsr        --> species profiles (steady-state)
|   |       +-- pfr        --> species profiles (flow)
|   |       +-- flame      --> laminar flame speed
|   |       +-- rcm        --> IDT (compression)
|   |
|   +-- 3f. Match to experimental data --> compute MAE
|   |
|   +-- 3g. --> Path1Results
|       +-- conditions_tested, mae_results[]
|       +-- overall_literature_better: bool
|       +-- extracted_rates  -----> feeds Path 2
|
+-- 4. PATH 2 -- Targeted Model Improvements
|   |
|   +-- 4a. Mine candidate reactions (LLM)
|   |   +-- ReactionMiningAgent
|   |       +-- classify_figure() --> flux_diagram | sensitivity | other
|   |       +-- interpret_flux() --> candidate reactions
|   |       +-- interpret_sensitivity() --> candidate reactions
|   |
|   +-- 4b. Enrich with rates from Path 1
|   |   +-- match reaction strings --> attach {A, n, Ea}
|   |
|   +-- 4c. Baseline: simulate original model on all conditions
|   |
|   +-- 4d. Branch: one modified model per candidate reaction
|   |   +-- ModelBranchingAgent.create_branch(reaction, original_model)
|   |
|   +-- 4e. Simulate each branch, compute delta_mae vs baseline
|   |
|   +-- 4f. --> Path2Results
|       +-- baseline_mae, branches[], best_branch_id
|
+-- 5. REPORT
    +-- ReportGenerator --> data/reports/report_YYYYMMDD_HHMMSS.yaml
```

## Where LLMs are used (and aren't)

| Component | LLM? | Reason |
|---|---|---|
| Paper retrieval | No | HTTP APIs (OpenAlex, Crossref) |
| PDF parsing | No | PyMuPDF + regex |
| Paper reading | **Yes** | Unstructured paper --> structured summary |
| Condition extraction | **Yes** | Validates deterministic plans against paper |
| Deterministic pipeline | No | Evidence --> scenarios --> plans (runs first) |
| Reaction mining | **Yes** | Flux/sensitivity figures --> candidate reactions |
| Chemkin conversion | Error recovery only | T3 handles 95% of cases |
| Model isolation check | No | Set intersection on reaction strings |
| Cantera simulation | No | Deterministic |
| MAE computation | No | Pure math |
| Report generation | No | YAML templating |

## Quick start

### Prerequisites

- Python 3.11+
- [Cantera](https://cantera.org/) 3.0+
- An LLM provider: Anthropic API key, OpenAI API key, or local [Ollama](https://ollama.ai/)

### Install

```bash
conda create -n agent_env python=3.12
conda activate agent_env
conda install -c cantera cantera
pip install -e .
```

### Configure LLM

Copy and edit the example config:

```bash
cp config/llm_config.yaml config/llm_config.local.yaml
```

```yaml
# config/llm_config.local.yaml
provider: anthropic          # anthropic | openai | ollama
model: claude-sonnet-4-6
api_key: sk-ant-...

# Optional: use different models per agent
agent_overrides:
  condition_extraction:
    model: claude-haiku-4-5
  paper_reader:
    model: claude-haiku-4-5
```

For local Ollama:

```yaml
provider: ollama
model: qwen3.5:4b
base_url: http://localhost:11434
```

### Run

```bash
chem-agent run \
  --model data/models/your_mechanism.yaml \
  --experiment data/experimental/your_conditions.yaml \
  --paper file:data/papers/your_paper.pdf \
  --path1 \
  --output data/reports/ \
  --llm-config config/llm_config.local.yaml
```

With a pre-identified literature model (skips SI discovery):

```bash
chem-agent run \
  --model data/models/original.yaml \
  --experiment data/experimental/conditions.yaml \
  --paper file:data/papers/paper.pdf \
  --literature-model data/models/literature.yaml \
  --species-aliases config/species_aliases.yaml \
  --literature-aliases config/lit_aliases.yaml \
  --path1 \
  --output data/reports/ \
  --llm-config config/llm_config.local.yaml
```

### Species aliases

RMG-generated models and literature models often use non-standard species names. Alias files remap plain names to model-specific names:

```yaml
# config/species_aliases.yaml
NH3: "NH3(1)"
O2: "O2(1)"
H2O: "H2O(1)"
```

Pass `--species-aliases` for the original model and `--literature-aliases` for the literature model.

## Experimental data format

User-provided YAML validated against `ExperimentalDataset`:

```yaml
name: NH3_Stagni2021
source: "10.1016/j.proci.2020.06.XXX"

conditions:
  - reactor_type: shock_tube
    temperature_K: 1400
    pressure_atm: 1.2
    mixture:
      NH3: 0.02
      O2: 0.02
      Ar: 0.96
    observable_type: idt
    measured_value: 0.00045
    error_threshold: 0.0001

  - reactor_type: jsr
    temperature_K: [900, 1000, 1100, 1200]
    pressure_atm: 1.0
    mixture:
      NH3: 0.01
      O2: 0.0095
      N2: 0.9805
    observable_type: species_profile
    observable_label: NH3
    measured_value: [0.0095, 0.006, 0.003, 0.001]
    error_threshold: 0.002
```

## Output report

```yaml
run_id: run_20260409_190947
original_model: data/models/mechanism.yaml
experimental_data: NH3_Stagni2021

path1:
  literature_model: data/models/literature.yaml
  conditions_tested: 3
  results:
    - condition_id: cond_0
      original_mae: "0.5492"
      literature_mae: "0.3551"
      literature_better: true
  overall_literature_better: true

path2:
  baseline_mae: 0.043
  branches:
    - branch_id: branch_001
      reactions_added: ["NH2 + O <=> HNO + H"]
      mae: 0.031
      delta_mae: -0.012
      improved: true
  best_branch: branch_001
```

## Repository structure

```
src/
  cli.py                  CLI entry point (chem-agent command)
  orchestrator.py         Top-level coordinator
  report.py               YAML report generator

  schemas/                Pydantic models (single source of truth)
    experimental.py       ExperimentalConditions, SimResult, PaperSummary
    ingestion.py          PaperRecord, SearchQuery, RegistryResult
    report.py             ReportOutput

  agents/                 LLM-backed agents (all calls via llm_client.py)
    llm_client.py         LiteLLM wrapper — only place LLM is called
    provider.py           PydanticAI model factory (Anthropic/OpenAI/Ollama)
    paper_reader.py       PaperReaderAgent — paper --> PaperSummary
    condition_reasoning.py ConditionReasoningAgent — paper --> SimConditions
    reaction_mining.py    ReactionMiningAgent — figures --> CandidateReaction
    conversion.py         ChemkinConverter (T3 + LLM error recovery)
    validators.py         ModelIsolationValidator (code enforcement)
    utils.py              strip_thinking(), extract_json_object()
    tools/                PydanticAI tool functions for agents
    skills/               Skill prompts (markdown files)

  pipelines/              Pipeline orchestration (agents don't write files)
    path1.py              Literature Model Evaluation
    path2.py              Targeted Model Improvements

  simulation/             Deterministic, LLM-free
    core/
      runner.py           Dispatch to reactor by type
      mae.py              compute_mae()
      model_loader.py     Load Cantera model
    reactors/             One class per reactor type
      shock_tube.py       IDTReactor
      jsr.py              JSRReactor
      pfr.py              PFRReactor
      flame.py            FlamespeedReactor
    templates/            Jinja2 templates for Cantera code generation

  ingestion/              Paper retrieval & parsing (no LLM)
    retrieval.py          OpenAlex, Crossref, EuropePMC, Semantic Scholar
    registry_builder.py   Discover, filter, catalogue papers + extract SI
    pdf_parser/
      pdf_parser.py       PyMuPDF-based PDF --> PaperDocument
      section_detector.py Section boundary detection
      table_parser.py     Condition table extraction
      caption_extractor.py Figure/table caption parsing
    pipeline/             Deterministic extraction pipeline
      families/           Experiment family registry (shock_tube, jsr, etc.)

tests/                    Mirror of src/ structure
  agents/
  simulation/
  ingestion/
  fixtures/               PDFs, mechanisms, experimental YAML

config/
  llm_config.yaml         Example LLM config

data/
  models/                 Cantera/Chemkin mechanism files
  experimental/           Experimental condition YAML files
  papers/                 Downloaded PDFs
  reports/                Output YAML reports
```

## Tech stack

| Layer | Tool | Why |
|---|---|---|
| Simulation | [Cantera](https://cantera.org/) | Industry standard for combustion kinetics |
| LLM framework | [PydanticAI](https://ai.pydantic.dev/) | Typed agents with structured output |
| LLM routing | [LiteLLM](https://docs.litellm.ai/) | Provider-agnostic (Anthropic, OpenAI, Ollama) |
| Schemas | [Pydantic v2](https://docs.pydantic.dev/) | Validation of all agent I/O and YAML configs |
| PDF parsing | [PyMuPDF](https://pymupdf.readthedocs.io/) | Best for multi-column chemistry papers |
| Paper retrieval | OpenAlex / Crossref | Implemented in `src/ingestion/` |
| Format conversion | [T3](https://github.com/ReactionMechanismGenerator/T3) | Chemkin --> Cantera YAML |
| Testing | pytest | 300+ tests, mirrors `src/` structure |

## Design principles

- **Correctness over cleverness.** This is scientific software. Deterministic code runs first; LLMs validate and enrich.
- **Model isolation is non-negotiable.** Path 1 literature mechanisms are never merged with the original model. Enforced by code (`ModelIsolationValidator`), not prompts.
- **LLMs are bounded.** Only three agents call LLMs: paper reader, condition reasoning, and reaction mining. Everything else is deterministic.
- **Schemas are the contract.** All inter-agent data is a Pydantic model. No raw dicts cross boundaries.
- **Agents don't write files.** They return data. Pipelines and the orchestrator handle I/O.

## Tests

```bash
conda activate agent_env
pytest                        # all tests
pytest tests/agents/          # agent tests only
pytest tests/simulation/      # simulation tests only
pytest tests/ingestion/       # ingestion tests only
```

Tests never make real LLM API calls or require Cantera to be installed — all external dependencies are mocked.

## License

TBD
