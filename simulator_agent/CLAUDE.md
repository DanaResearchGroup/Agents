# CLAUDE.md (simulator_agent)

## What This Module Is

The simulator agent evaluates kinetic models against experimental data.

Primary question it answers:
> "Given this experiment YAML and these model(s), how well does each model reproduce the experiment?"

It also contains a **literature support layer** for extracting conditions from published papers.

---

## Package Structure

```
simulator_agent/
  orchestrator/            # AGENT LAYER — thin coordination
    agent.py               # SimulatorAgent: evaluate_models, extract_experiment, evaluate_from_literature
    spec_builder.py        # SimulationPlan → SimulationSpec conversion (unit normalization)
    review.py              # ReviewQueue for draft/blocked plans

  core/                    # ENGINE — simulation + evaluation
    exp_loader.py          # YAML → ExperimentalDataset
    simulation_spec.py     # SimulationSpec (canonical internal format, SI units)
    model_loader.py        # Cantera mechanism loading
    runner.py              # reactor-type dispatch → SimulationResult
    observable_extractor.py # extract observables from results
    mae.py                 # MAE / relative MAE
    report_writer.py       # structured JSON evaluation reports

  literature_support/      # SUPPORT — PDF parsing pipeline
    parser/                # PDF → structured document
    extractor/             # text → evidence snippets
    builder/               # evidence → scenarios
    planner/               # scenarios → simulation plans
    validator/             # physical/logical consistency
    families/              # experiment family registry
    figure_handler/        # OCR panel extraction
    report/                # literature-specific JSON/markdown output

  generator/               # SimulationPlan → executable code
  models.py                # shared data models (literature pipeline)
  main.py                  # CLI entry point
```

---

## CLI Usage

### Primary: model evaluation

```bash
python main.py evaluate --exp-yaml experiments.yaml --model mechanism.yaml
python main.py evaluate --exp-yaml experiments.yaml --model original.yaml --compare-model literature.yaml
python main.py --exp-yaml experiments.yaml --model mechanism.yaml  # shorthand
```

### Secondary: PDF extraction

```bash
python main.py extract --pdf paper.pdf [--outdir outputs/] [--debug]
python main.py --pdf paper.pdf  # shorthand
```

---

## Orchestrator (Agent Layer)

The `SimulatorAgent` class provides three entry points:

```python
from orchestrator import SimulatorAgent
agent = SimulatorAgent()

# Path 1: YAML experiments + model(s)
report = agent.evaluate_models("experiments.yaml", ["mech.yaml"])

# Path 2: PDF → scenarios + plans + review queue
result = agent.extract_experiment("paper.pdf")

# Path 3: PDF → evaluate executable plans against model(s)
lit_result = agent.evaluate_from_literature("paper.pdf", ["mech.yaml"])
```

### Plan routing

- **executable** → auto-convert to SimulationSpec → simulate → score
- **draft** → ReviewQueue (blocking reasons, missing fields)
- **blocked** → ReviewQueue (skipped, recorded)

### spec_builder

Bridges literature pipeline output (`SimulationPlan`) to core engine input (`SimulationSpec`):
- template_family → reactor_type mapping
- Pressure unit conversion (atm/bar/kPa → Pa)
- Time unit conversion (ms → s)
- Temperature range → temperature_list

---

## Core Responsibilities

### 1. Load experimental data (YAML)
- `core/exp_loader.py` loads experiment YAML with conditions + measured data
- Produces `ExperimentalDataset` (spec + data points)

### 2. Normalize into SimulationSpec
- All values in SI units (K, Pa, s, mole fractions)
- Supports sweeps via `temperature_list` / `pressure_list`

### 3. Run simulation
- `core/runner.py` dispatches by reactor type: shock_tube, jsr, flow_reactor, rcm, flame
- Returns `SimulationResult` with time series + species histories

### 4. Extract observable
- `core/observable_extractor.py` maps observable type to extraction logic
- Supports: ignition_delay, species_profile, flame_speed, half_life, conversion, temperature_profile

### 5. Compute MAE
- `core/mae.py` for absolute and relative MAE

### 6. Write report
- `core/report_writer.py` produces machine-readable JSON with per-model scores, ranking, threshold comparison

---

## Literature Support (Secondary)

Pipeline: parse PDF → extract evidence → build scenarios → validate → build plans

Used when:
- experimental YAML is incomplete
- reproducing literature experiments
- extracting conditions from papers

### Evidence Precedence
panel text > caption > nearby paragraph > methods > general text

Never overwrite higher-confidence values.

### Status System
- Scenario: accepted / needs_review / rejected
- Plan: executable / draft / blocked

### Confidence Policy
- overall = min(composition, temperature, pressure)
- < 0.7 → needs_review
- missing → rejected or draft

---

## Scope Boundaries

This module does NOT:
- define repository-wide workflow
- manage literature retrieval
- perform reaction selection (flux analysis)

---

## Outputs

### Evaluation path
- `evaluation_report.json` — per-model MAE, ranking, threshold results

### Literature path
- `scenarios.json`, `simulation_plans.json`, `run_summary.md`

---

## Success Criteria

- correct simulation execution
- correct observable extraction
- accurate MAE calculation
- reliable, traceable plans
