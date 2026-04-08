# ARCHITECTURE.md — System Design

This document is the authoritative design reference. Claude Code implements what is
written here. If a coding decision contradicts this document, update this document
first and record the rationale in DECISIONS.md.

---

## 1. System overview

The system evaluates and improves chemical kinetic models by running two independent
analysis paths against user-provided experimental data. It does not perform continuous
automated iteration — each run is a structured, one-shot comparison with a final report.

```
User provides:
  ├── Original kinetic model (Cantera YAML or Chemkin format)
  ├── Experimental data (YAML, schema in Section 4)
  ├── Papers (DOI | uploaded PDF | search query)
  └── Config (LLM provider, paths to run)
           │
           ▼
     Orchestrator
     (src/orchestrator.py)
           │
     ┌─────┴──────┐
     ▼            ▼
  Path 1       Path 2
(Literature  (Targeted
 Evaluation) Improvements)
     │            │
     └─────┬──────┘
           ▼
     Report Generator
     → report.yaml
```

---

## 2. Two-path design

### Path 1 — Literature Model Evaluation

**Objective:** Compare the original model against a published literature mechanism.
Models run independently. No reactions are merged. Ever.

```
PaperIngestionAgent
  → FormatConversionAgent        (Chemkin → Cantera YAML via T3)
  → ConditionExtractionAgent     (LLM: paper text → SimConditions)
  → ModelIsolationValidator      (hard check: no shared reactions)
  → Path1SimulationAgent         (run literature model + original model separately)
  → MAEEvaluator                 (shared module)
  → Path1Results
```

**Hard rule:** `ModelIsolationValidator` must pass before any simulation runs.
If it raises `ModelIsolationViolation`, the entire path halts. This is enforced
in code, not via prompting.

### Path 2 — Targeted Model Improvements

**Objective:** Test whether specific reactions from literature improve the original model.

```
PaperIngestionAgent
  → ReactionMiningAgent          (LLM: parse sensitivity/flux sections → candidate reactions)
  → ModelBranchingAgent          (create one modified model per candidate reaction/set)
  → Path2SimulationAgent         (simulate each branch under same conditions)
  → MAEEvaluator                 (shared module)
  → Path2Results
```

**Note:** Path 2 branches always descend from the original model. The validator
confirms this — a Path 2 branch that does not contain the original model's reactions
is an error.

### Path independence

Paths can run independently or sequentially. If both run, Path 1 results are
available as optional context to Path 2 (e.g. if Path 1 identified a better
literature model, Path 2 may mine reactions from that model's source paper).
This is opt-in, not automatic.

---

## 3. Component map

### 3.1 Shared schemas (`src/schemas/`)

All inter-component data structures. Defined once, used everywhere.
Nothing outside `src/schemas/` should define Pydantic models.

```python
# src/schemas/experimental.py
class ReactorType(str, Enum):
    SHOCK_TUBE = "shock_tube"
    JSR = "jsr"
    PFR = "pfr"
    FLAME = "flame"
    RCM = "rcm"

class ObservableType(str, Enum):
    IDT = "idt"
    FLAME_SPEED = "flame_speed"
    SPECIES_PROFILE = "species_profile"
    K_EXT = "k_ext"

class ExperimentalCondition(BaseModel):
    reactor_type: ReactorType
    temperature_K: float | list[float]
    pressure_atm: float | list[float]
    mixture: dict[str, float]          # species: mole fraction
    observable_type: ObservableType
    observable_label: str | None       # species label if species_profile
    measured_value: float | list[float]
    error_threshold: float             # acceptable MAE

class ExperimentalDataset(BaseModel):
    name: str
    conditions: list[ExperimentalCondition]
    source: str | None                 # paper DOI or reference

class SimConditions(BaseModel):
    """Validated, simulation-ready form of ExperimentalCondition."""
    reactor_type: ReactorType
    T: float
    P: float
    X: dict[str, float]               # Cantera-compatible mole fraction dict
    observable_type: ObservableType
    observable_label: str | None

class SimResult(BaseModel):
    conditions: SimConditions
    simulated_value: float
    units: str
    success: bool
    error_message: str | None

class MAEResult(BaseModel):
    model_name: str
    conditions_id: str
    mae: float
    threshold: float
    passed: bool
```

### 3.2 Simulation layer (`src/simulation/`)

**No LLM calls in this layer. Ever.**

```
src/simulation/
  core/
    mae.py             ← compute_mae(simulated, experimental) -> MAEResult
    runner.py          ← run_simulation(model_path, conditions) -> SimResult
    model_loader.py    ← load_cantera_model(path) -> ct.Solution
    observable_extractor.py  ← extract_observable(sim, observable_type) -> float
  reactors/
    base.py            ← abstract BaseReactor with run(SimConditions) -> SimResult
    shock_tube.py      ← IDTReactor
    jsr.py             ← JSRReactor
    pfr.py             ← PFRReactor (flow reactor)
    flame.py           ← FlamespeedReactor
  templates/
    shock_tube_idt_const_uv.py.j2
    jsr_const_tp.py.j2
    flow_reactor_const_tp.py.j2
    flame_speed.py.j2
```

Each reactor implements:
```python
class BaseReactor(ABC):
    @abstractmethod
    def run(self, conditions: SimConditions, model_path: Path) -> SimResult:
        ...
```

The runner dispatches to the correct reactor based on `conditions.reactor_type`.

### 3.3 Ingestion layer (`src/ingestion/`)

Handles all paper acquisition. Output is always a `Paper` object.

```
src/ingestion/
  retrieval.py       ← DOI → fetch PDF (Crossref/OpenAlex/Unpaywall)
                       search query → ranked results → user confirms → fetch
                       upload → receive PDF bytes
  pdf_parser.py      ← PDF → structured text (sections, tables, captions)
  figure_handler.py  ← extract figures, panel text
```

```python
class Paper(BaseModel):
    doi: str | None
    title: str
    full_text: str
    sections: dict[str, str]     # section_name: text
    tables: list[str]
    figures: list[FigureData]
    source_path: Path
    si_path: Path | None
```

Three ingestion modes, same output:
```python
async def ingest_doi(doi: str) -> Paper: ...
async def ingest_upload(pdf_bytes: bytes) -> Paper: ...
async def ingest_search(query: str) -> Paper: ...   # prompts user to confirm match
```

### 3.4 Agent layer (`src/agents/`)

LLM-backed agents. All LLM calls go through `llm_client.py`.

#### llm_client.py — single LLM interface

```python
class LLMClient:
    """Single point of LLM access. Wraps LiteLLM."""
    
    def __init__(self, config: LLMConfig): ...
    
    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        response_model: type[BaseModel] | None = None,  # structured output
        agent_name: str | None = None,   # for per-agent model override
    ) -> str | BaseModel: ...
```

Config schema (`config/llm_config.yaml`):
```yaml
provider: anthropic          # anthropic | openai | ollama
model: claude-sonnet-4-6
api_key: sk-...              # omit for ollama
base_url: ~                  # http://localhost:11434 for ollama

agent_overrides:             # optional: use cheaper model for fast tasks
  condition_extraction:
    model: claude-haiku-4-5
  reaction_mining:
    model: claude-sonnet-4-6
```

#### condition_extraction.py

```
Input:  Paper + ExperimentalDataset (user-provided, for schema reference)
Output: list[SimConditions]
LLM:    Yes — extracts T, P, mixture, reactor type, observable from paper text
```

#### reaction_mining.py

```
Input:  Paper
Output: list[CandidateReaction]
LLM:    Yes — parses sensitivity analysis, flux diagrams, rate discussions
```

```python
class CandidateReaction(BaseModel):
    reaction_string: str          # e.g. "H + O2 <=> O + OH"
    rate_params: dict | None      # Arrhenius A, n, Ea if found
    source_section: str           # where in paper it was found
    justification: str            # why LLM flagged it as important
    confidence: float             # 0-1
```

#### conversion.py

```
Input:  Path to Chemkin mechanism file
Output: Path to converted Cantera YAML file
LLM:    Only for error recovery (T3 conversion errors → LLM suggests fix)
```

Uses T3's `fix_cantera.py` as primary converter. If conversion fails,
the error message is passed to LLM for diagnosis and retry (max 3 attempts).

#### validators.py — ModelIsolationValidator

```python
class ModelIsolationViolation(Exception): ...

class ModelIsolationValidator:
    def validate_path1(
        self,
        literature_model: Path,
        original_model: Path
    ) -> None:
        """Raise ModelIsolationViolation if models share any reactions."""
        
    def validate_path2_branch(
        self,
        branch_model: Path,
        original_model: Path
    ) -> None:
        """Raise ValueError if branch does not descend from original."""
```

Comparison is done on reaction equation strings (normalised), not file paths.

### 3.5 Pipelines (`src/pipelines/`)

Orchestrate the agent sequence for each path. Pipelines write output files.
Agent classes do not write files.

#### path1.py

```python
async def run_path1(
    paper: Paper,
    original_model: Path,
    experimental_data: ExperimentalDataset,
    llm_client: LLMClient,
) -> Path1Results:
    
    # 1. Convert literature mechanism (from SI)
    # 2. Extract simulation conditions from paper
    # 3. Validate isolation (hard stop if violated)
    # 4. Run literature model simulations
    # 5. Run original model simulations (same conditions)
    # 6. Compute MAE for both
    # Return Path1Results
```

#### path2.py

```python
async def run_path2(
    paper: Paper,
    original_model: Path,
    experimental_data: ExperimentalDataset,
    llm_client: LLMClient,
    path1_results: Path1Results | None = None,  # optional context
) -> Path2Results:
    
    # 1. Mine candidate reactions from paper
    # 2. Create model branch per candidate
    # 3. Validate each branch descends from original
    # 4. Run original model baseline
    # 5. Run each branch simulation
    # 6. Compute MAE for each branch vs original baseline
    # Return Path2Results
```

### 3.6 Orchestrator (`src/orchestrator.py`)

Top-level entry point. Takes user inputs, decides which paths to run,
calls pipelines, collects results, triggers report generation.

```python
async def run(config: RunConfig) -> Path:
    """Returns path to output report YAML."""
    paper = await ingest_paper(config.paper_source)
    
    results = {}
    if config.run_path1:
        results["path1"] = await run_path1(...)
    if config.run_path2:
        results["path2"] = await run_path2(...)
    
    report_path = generate_report(results, config.output_dir)
    return report_path
```

### 3.7 Report generator (`src/report.py`)

Purely deterministic. Takes results structs, writes YAML.

```yaml
# Output report structure
run_id: run_20260408_143022
original_model: original_model.yaml
experimental_data: exp_nh3_jsr.yaml

path1:
  literature_model: wang2023_nh3.yaml
  source_doi: 10.1016/j.combustflame.2023.xxxxx
  conditions_tested: 12
  results:
    - condition_id: T1200_P1atm_phi1
      original_mae: 0.043
      literature_mae: 0.021
      threshold: 0.05
      literature_better: true
  overall_literature_better: true

path2:
  baseline_mae: 0.043
  branches:
    - branch_id: branch_001
      reactions_added:
        - "NH2 + O <=> HNO + H"
      mae: 0.031
      improvement: true
      delta_mae: -0.012
  best_branch: branch_001
```

---

## 4. Experimental data YAML schema

User-provided input file. Validated against `ExperimentalDataset` on load.

```yaml
name: NH3_JSR_Dagaut2000
source: "10.1016/S0010-2180(99)00122-3"

conditions:
  - reactor_type: jsr
    temperature_K: [900, 1000, 1100, 1200, 1300]
    pressure_atm: 1.0
    mixture:
      NH3: 0.01
      O2: 0.0095
      N2: 0.9805
    phi: 1.0
    residence_time_s: 0.1
    observable_type: species_profile
    observable_label: NH3
    measured_value: [0.0095, 0.006, 0.003, 0.001, 0.0005]
    error_threshold: 0.002

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
```

---

## 5. Where LLMs are and are not used

| Component | LLM? | Reason |
|---|---|---|
| Paper retrieval | No | HTTP + API |
| PDF parsing | No | PyMuPDF |
| Chemkin conversion | Error recovery only | T3 handles 95% of cases |
| Condition extraction | **Yes** | Unstructured paper text → structured schema |
| Reaction mining | **Yes** | Requires chemical reasoning about paper content |
| Cantera simulation | No | Deterministic code |
| MAE computation | No | Pure math |
| Model isolation check | No | Set intersection on reaction strings |
| Report generation | No | Template/schema |

The LLM budget is two agents: `ConditionExtractionAgent` and `ReactionMiningAgent`.
All other LLM usage is error recovery, not primary logic.

---

## 6. LLM provider abstraction

All LLM calls go through `src/agents/llm_client.py` which wraps LiteLLM.
This gives users three modes:

```
Anthropic API    ANTHROPIC_API_KEY set          → claude-sonnet-4-6 (default)
OpenAI API       OPENAI_API_KEY set             → gpt-4o or user choice
Local (Ollama)   base_url: localhost:11434      → ollama/llama3 or user choice
```

No other file imports `anthropic`, `openai`, or `litellm` directly.

---

## 7. Model isolation enforcement (Path 1)

Three layers of enforcement — this is non-negotiable:

**Layer 1 — File isolation**
The orchestrator passes a read-only path reference for the original model to Path 1.
Path 1 agents never receive a writable handle to the original model.

**Layer 2 — Reaction set validator**
`ModelIsolationValidator.validate_path1()` loads both mechanisms, extracts all
reaction equation strings (normalised: sorted stoichiometry, canonical species names),
and asserts the intersection is empty. Raises `ModelIsolationViolation` on any overlap.
This runs before any simulation.

**Layer 3 — Audit trail**
The output report records the exact file path and SHA256 hash of every model used
in every simulation. Any violation would be visible in the report.

---

## 8. What is explicitly out of scope (for now)

- Continuous automated iteration / closed-loop optimisation
- Automatic mechanism generation (RMG integration)
- Rate coefficient optimisation
- Multi-objective optimisation across conditions
- Web UI
- Database persistence of results
- Parallel simulation execution (sequential is fine for v1)

---

## 9. Migration from original repo structure

The original repo had three disconnected sub-projects:
- `simulator_agent/` → primary source, most of `src/simulation/` and `src/ingestion/`
- `literature_agent/` → paper retrieval, migrated to `src/ingestion/retrieval.py`
- `flux_diagram_agent/` → reaction mining, migrated to `src/agents/reaction_mining.py`
- `Agent_demo/` → throwaway demo code, not migrated

See DECISIONS.md entry "2026-04-08 — Repo restructure" for the full migration log.
