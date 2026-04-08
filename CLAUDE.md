# CLAUDE.md — Session Context

Read this file at the start of every session. Do not skip it.

---

## What this project is

A multi-agent framework for evaluating and improving chemical kinetic models.
It runs two independent paths: (1) comparing an original model against literature
mechanisms, and (2) extracting candidate reactions from papers and testing them
as targeted modifications to the original model. All simulation is done via Cantera.
Results are reported as a structured YAML file with MAE values per model/condition.

This is a scientific software project. Correctness matters more than cleverness.
When in doubt, write boring, explicit, testable code.

---

## Tech stack

| Layer | Tool | Why |
|---|---|---|
| Simulation | Cantera (Python API) | Industry standard for combustion kinetics |
| LLM calls | LiteLLM | Provider-agnostic: Anthropic, OpenAI, Ollama all work |
| Schemas | Pydantic v2 | Validation of experimental YAML + all agent I/O |
| PDF parsing | PyMuPDF (fitz) | Best for multi-column chemistry papers |
| Paper retrieval | OpenAlex / Crossref API | Already implemented in src/ingestion/ |
| Format conversion | T3 fix_cantera.py (subprocess) | Chemkin → Cantera YAML |
| Package management | pyproject.toml (single file, repo root) | Replaced 3x legacy requirements.txt |
| Testing | pytest | Mirror src/ structure under tests/ |

---

## Repository structure

```
repo/
  CLAUDE.md              ← this file
  ARCHITECTURE.md        ← full system design — read before implementing anything
  DECISIONS.md           ← log of architectural decisions and rationale
  pyproject.toml         ← single unified dependencies

  src/
    schemas/             ← Pydantic models, shared by everything
      experimental.py    ← ExperimentalConditions, SimResult (source of truth)
      report.py          ← ReportOutput schema
    simulation/
      core/              ← mae.py, runner.py, model_loader.py (deterministic, no LLM)
      reactors/          ← IDT, JSR, PFR, Flame (one class per reactor type)
      templates/         ← Jinja2 .j2 templates for Cantera code generation
    ingestion/           ← paper retrieval (DOI/upload/search), PDF parsing, figures
      pipeline/          ← deterministic extraction: evidence → scenarios → plans
        families/        ← experiment family registry (shock_tube, jsr, etc.)
        builder/         ← candidate/scenario extraction, enrichment, validation
        report/          ← JSON/markdown output writers
    agents/
      llm_client.py      ← ONLY place LiteLLM is called. Never call LiteLLM directly elsewhere.
      condition_extraction.py
      reaction_mining.py
      conversion.py      ← Chemkin → Cantera, uses T3
      validators.py      ← ModelIsolationValidator (hard enforcement, not a prompt)
    pipelines/
      path1.py           ← Literature Model Evaluation pipeline
      path2.py           ← Targeted Model Improvements pipeline
    orchestrator.py      ← routes between paths, collects results
    report.py            ← YAML report generator

  tests/
    fixtures/            ← PDFs, example mechanisms, exp YAML files
    simulation/
    ingestion/
    agents/
  
  config/
    llm_config.yaml      ← user LLM settings (provider, model, api_key)
  
  data/
    models/              ← Cantera/Chemkin mechanism files
    experimental/        ← user experimental YAML files
    papers/              ← downloaded/uploaded PDFs
    reports/             ← output YAML reports
```

---

## Conventions — follow these exactly

### Naming
- Classes: `PascalCase`
- Functions/variables: `snake_case`
- Files: `snake_case.py`
- Pydantic models live in `src/schemas/` only — do not define data models inline in agents
- Agent classes are named `<Task>Agent` (e.g. `ConditionExtractionAgent`)
- Reactor classes are named `<Type>Reactor` (e.g. `IDTReactor`, `JSRReactor`)

### LLM calls
- All LLM calls go through `src/agents/llm_client.py` — nowhere else
- Never hardcode a model name outside of `llm_client.py` or `config/llm_config.yaml`
- Never import `anthropic`, `openai`, or `ollama` directly — always use LiteLLM via llm_client

### Simulation code
- `src/simulation/` is LLM-free. No LLM calls, no prompts, no API calls.
- Every reactor must implement the `BaseReactor` interface from `simulation/reactors/base.py`
- `run(conditions: SimConditions) -> SimResult` is the only public method reactors expose

### Schemas
- All inter-agent data must be a Pydantic model, not a raw dict
- Never pass raw dicts between agents or pipelines
- If you find yourself typing `Dict[str, Any]`, stop and define a proper schema

### Path 1 hard rule — MODEL ISOLATION
- The literature mechanism must NEVER be merged with the original model
- `ModelIsolationValidator` in `src/agents/validators.py` must be called before any Path 1 simulation
- This is a code enforcement, not a prompt instruction
- If the validator raises `ModelIsolationViolation`, the pipeline must halt — do not catch and continue

### File I/O
- All output goes to `data/reports/` or a user-specified path
- Never write output files from inside `src/simulation/` or `src/agents/`
- Only `orchestrator.py`, `pipelines/`, and `report.py` write files

### Tests
- Every new function gets a test
- Tests go in `tests/` mirroring the `src/` path (e.g. `src/simulation/core/mae.py` → `tests/simulation/core/test_mae.py`)
- Use fixtures from `tests/fixtures/` — do not hardcode paths inside test functions
- Tests must not make real LLM API calls — mock `llm_client.py`
- Tests must not require Cantera to be installed to pass (use mocks for simulation tests)

---

## What is already implemented (as of project start)

These exist and largely work — understand before modifying:
- `src/simulation/reactors/` — reactor family classes (migrated from `simulator_agent/literature_support/families/`)
- `src/simulation/templates/` — Jinja2 templates for all reactor types
- `src/simulation/core/mae.py` — MAE computation
- `src/simulation/core/runner.py` — Cantera runner
- `src/ingestion/` — OpenAlex/Crossref paper retrieval (migrated from `literature_agent/accelerator/`)
- `src/ingestion/pdf_parser.py` — PDF parsing pipeline
- `src/agents/reaction_mining.py` — flux diagram / reaction extraction (migrated from `flux_diagram_agent/`)

These are NEW and need to be built:
- `src/agents/llm_client.py` — LiteLLM wrapper
- `src/agents/conversion.py` — Chemkin → Cantera via T3
- `src/agents/validators.py` — ModelIsolationValidator
- `src/pipelines/path1.py` and `path2.py`
- `src/schemas/report.py`
- `src/orchestrator.py` (rework of existing orchestrator)

---

## What NOT to do

- Do not add new top-level directories without updating ARCHITECTURE.md first
- Do not add new dependencies without updating pyproject.toml and DECISIONS.md
- Do not make LLM calls from simulation code
- Do not merge the literature model into the original model under any circumstances
- Do not re-implement paper retrieval — it exists in `src/ingestion/`
- Do not define Pydantic models outside `src/schemas/`
- Do not write to disk from agent classes — return data, let pipelines/orchestrator write
- Do not use `Dict[str, Any]` as an agent interface
- Do not make architectural decisions during a coding session — raise them in DECISIONS.md and discuss separately

---

## Current status

### Legacy directories (DELETED in Phase 8b)
All code migrated to src/. Legacy dirs removed: simulator_agent/, literature_agent/, flux_diagram_agent/, Agent_demo/

### Phase 1 — Foundation (complete) 
- [x] src/schemas/experimental.py — unified Pydantic models
- [x] src/agents/llm_client.py — LiteLLM wrapper

### Phase 2 — Migration (blocked on Phase 1)
- [X] src/simulation/ — move deterministic code
- [X] src/ingestion/ — move retrieval + parsing

### Phase 3 — LLM boundary fix (blocked on Phase 1)
- [x] src/agents/reaction_mining.py — replace DeepSeek/GLM with LiteLLM

### Phase 4 — Orchestrator rework
- [x] src/orchestrator.py

### Phase 5 — New components
- [x] src/agents/conversion.py
- [x] src/agents/validators.py
- [x] src/pipelines/path1.py
- [x] src/pipelines/path2.py

### Phase 7 — Report generator + Chemkin rate extraction
- [x] src/agents/conversion.py — extract_rates() for Chemkin A/n/Ea parsing
- [x] src/schemas/experimental.py — extracted_rates field on Path1Results
- [x] src/pipelines/path1.py — wired extract_rates after conversion
- [x] src/pipelines/path2.py — enriches candidates from extracted_rates
- [x] src/report.py — ReportGenerator (YAML output per ARCHITECTURE.md 3.7)
- [x] src/orchestrator.py — wired ReportGenerator replacing placeholder

### Phase 8a — Unified dependencies, CLI, and example config
- [x] pyproject.toml — single unified dependencies replacing 3x legacy files
- [x] src/cli.py — argparse CLI with run/validate-model/convert commands
- [x] config/llm_config.yaml — example LLM config with agent_overrides
- [x] tests/test_cli.py — 16 tests covering arg parsing, paper sources, all commands

### Phase 8b — Deterministic extraction pipeline + legacy cleanup
- [x] src/ingestion/pipeline/ — full deterministic extraction pipeline (10 modules)
- [x] src/ingestion/pipeline/families/ — experiment family registry (5 families)
- [x] src/ingestion/pipeline/extractor.py — PaperExtractionPipeline coordinator
- [x] src/pipelines/path1.py — two-stage extraction (deterministic-first, LLM fallback)
- [x] src/schemas/experimental.py — validate() compatibility methods on 4 Pydantic models
- [x] Legacy directories deleted (125 files, 33k lines removed)
- [x] 157/157 tests pass

---

## Key contacts / references

- T3 fix_cantera.py: https://github.com/ReactionMechanismGenerator/T3/blob/main/t3/utils/fix_cantera.py
- Cantera Python docs: https://cantera.org/documentation/
- OpenAlex API: https://docs.openalex.org/
- LiteLLM docs: https://docs.litellm.ai/

