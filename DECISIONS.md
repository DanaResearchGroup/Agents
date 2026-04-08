# DECISIONS.md — Architectural Decision Log

Every non-obvious decision gets recorded here with rationale.
This prevents re-litigating settled questions across sessions.

Format:
  ## YYYY-MM-DD — Short title
  **Decision:** what was decided
  **Reason:** why
  **Rejected alternatives:** what was considered and rejected
  **Implications:** what this decision constrains downstream

---

## 2026-04-08 — Repo restructure from three sub-projects to unified src/

**Decision:** Merge `simulator_agent/`, `literature_agent/`, and `flux_diagram_agent/`
into a single `src/` package under one `pyproject.toml`. `Agent_demo/` is discarded.

**Reason:** The three sub-projects could not communicate — no shared schemas, no shared
LLM client, no shared interfaces. Building Path 1 and Path 2 pipelines requires all three
to work together.

**Rejected alternatives:**
- Keep as separate packages with inter-package imports: adds packaging complexity with
  no benefit at this scale.
- Keep as separate processes communicating over HTTP/sockets: massively over-engineered
  for a single-user research tool.

**Implications:** One virtual environment, one `pyproject.toml`, one test suite.
All existing code migrated by moving files, not rewriting logic.

---

## 2026-04-08 — LiteLLM as LLM abstraction layer

**Decision:** All LLM calls go through LiteLLM via a single `LLMClient` class in
`src/agents/llm_client.py`. No other file imports `anthropic`, `openai`, or `litellm`.

**Reason:** Users need to provide their own API key (Anthropic or OpenAI) or run
a local LLM via Ollama. LiteLLM gives a single interface for all three without
any conditional branching in agent code.

**Rejected alternatives:**
- Raw Anthropic SDK: locks out OpenAI and local LLM users.
- LangChain: adds significant abstraction overhead for what is essentially one
  function call (`complete(prompt) -> str`). Introduces framework-level complexity
  we don't need.

**Implications:** `config/llm_config.yaml` is the only place provider/model is
specified. Per-agent model overrides are supported (e.g. Haiku for cheap tasks,
Sonnet for reaction mining).

---

## 2026-04-08 — Model isolation is code enforcement, not prompt instruction

**Decision:** Path 1's requirement that the literature mechanism not be merged with
the original model is enforced by `ModelIsolationValidator` — a deterministic code
check — not by instructing the LLM not to merge them.

**Reason:** Prompt instructions are not reliable safety mechanisms for hard constraints.
A code check that raises an exception is deterministic and auditable.

**Rejected alternatives:**
- Trust the LLM prompt: unacceptable for a scientific correctness requirement.
- File system permissions: too OS-specific and fragile.

**Implications:** `ModelIsolationValidator.validate_path1()` must be called in
`pipelines/path1.py` before any simulation step. Catching `ModelIsolationViolation`
and continuing is not permitted.

---

## 2026-04-08 — LLMs used only for condition extraction and reaction mining

**Decision:** LLM calls are restricted to two agents: `ConditionExtractionAgent`
(paper text → SimConditions) and `ReactionMiningAgent` (paper text → candidate reactions).
All other components are deterministic code.

**Reason:** LLM calls add latency, cost, and non-determinism. They are justified only
where the task is genuinely unstructured-text-to-structured-data extraction that cannot
be done with a parser.

**Rejected alternatives:**
- LLM-driven Cantera code generation: too risky, Cantera API has subtle correctness
  requirements (solver tolerances, reactor setup order) that need hand-coded templates.
- LLM orchestration: the pipeline structure is fixed and known; a rule-based orchestrator
  is simpler and more reliable.

**Implications:** If a new agent is proposed that calls an LLM, the justification must
be that the task is irreducibly unstructured. Add the decision here before implementing.

---

## 2026-04-08 — Reactor templates over LLM-generated Cantera code

**Decision:** Four hand-coded Jinja2 templates cover all reactor types (shock tube/IDT,
JSR, PFR/flow reactor, laminar flame speed). Cantera simulation code is generated from
these templates, not written by an LLM at runtime.

**Reason:** Cantera reactor setup has non-obvious correctness requirements (JSR steady-
state convergence, flame solver tolerances, IDT detection methods). Getting these wrong
produces silently incorrect results. Hand-coded templates that are validated once are
safer than runtime LLM code generation.

**Rejected alternatives:**
- MCP server for Cantera docs + LLM code generation: adds significant complexity for
  marginal flexibility. Can be revisited if a reactor type arises that doesn't fit
  any template.
- Single generic reactor: not possible, each reactor type needs fundamentally different
  Cantera setup.

**Implications:** Adding a new reactor type requires a new Jinja2 template and a new
reactor class in `src/simulation/reactors/`. It does not require LLM changes.

---

## 2026-04-08 — PyMuPDF for PDF parsing

**Decision:** Use PyMuPDF (`fitz`) for PDF text and figure extraction.

**Reason:** Chemistry journal papers typically use multi-column layouts, complex tables,
and embedded figures. PyMuPDF handles these better than pdfplumber for this document type.
Already in use in the existing `simulator_agent/` codebase.

**Rejected alternatives:**
- pdfplumber: weaker on multi-column layouts.
- pypdf: text-only, no figure extraction.
- LLM vision: expensive, slow, and unnecessary for structured section extraction.

**Implications:** PyMuPDF is a binary dependency. Document in pyproject.toml.

---

## 2026-04-08 — Pydantic v2 as the single schema layer (Option A)

**Decision:** All data models across the entire codebase use Pydantic v2.
`simulator_agent/models.py` stdlib dataclasses (`PaperDocument`, `EvidenceSnippet`,
`ExperimentCandidate`, `ExperimentalScenario`, `SimulationPlan`, `NormalizedCondition`,
`NormalizedComposition`, `FieldConfidence`) are converted to Pydantic during migration.
`literature_agent/accelerator/models.py` is already Pydantic and moves as-is.

**Reason:** Two type systems (dataclasses + Pydantic) in the same codebase require
conversion boilerplate at every boundary. All agent interfaces must be Pydantic for
YAML/JSON validation, serialisation, and consistent `.model_dump()` / `.model_validate()`
across the pipeline. The existing dataclasses contain no custom validation logic that
would be lost — the conversion is mechanical field-for-field.

**Rejected alternatives:**
- Option B (keep dataclasses internally, Pydantic only at boundaries): produces
  conversion boilerplate at every agent handoff. Two mental models for the same thing.
- Keep stdlib dataclasses everywhere: lose free YAML validation, JSON serialisation,
  and schema introspection needed for the report generator.

**Implications:**
- `src/schemas/experimental.py` is the single file where all Pydantic models live.
- No `dataclass` imports anywhere in `src/`. If you see one, it's a migration gap.
- The conversion from `simulator_agent/models.py` dataclasses to Pydantic is done
  once during Phase 1 and never revisited.
- Field names from the original dataclasses are preserved exactly to avoid breaking
  the existing parsing/extraction logic that populates them.

---

## 2026-04-08 — Paths can run independently or sequentially

**Decision:** Path 1 and Path 2 are independent pipelines. They can be run individually
or both in sequence. If both run, Path 1 results are available as optional context to
Path 2, but Path 2 does not require Path 1 results to run.

**Reason:** A user may want to evaluate literature models without testing modifications
(Path 1 only), or test targeted improvements without a full literature comparison
(Path 2 only).

**Rejected alternatives:**
- Always run both sequentially: limits flexibility, increases runtime cost.
- Merge into one pipeline: confuses the distinct purposes of each path.

**Implications:** `orchestrator.py` reads `config.run_path1` and `config.run_path2`
flags. Both default to True. The `path1_results` parameter in `run_path2()` is
`Optional` and defaults to `None`.

--

## 2026-04-08 — SimulationSpec and SimulationResult remain as dataclasses (temporary)

**Decision:** src/simulation/core/ retains stdlib dataclasses for SimulationSpec 
and SimulationResult rather than converting to Pydantic during Phase 2 migration.

**Reason:** These types operate at a different abstraction level than the 
orchestrator-facing Pydantic models (SimConditions/SimResult). Forcing the 
conversion during a move-only phase risked breaking Cantera logic.

**Implications:** Two type systems still exist internally. These must be converted 
to Pydantic and unified with SimConditions/SimResult before Path 1/2 pipelines 
are built (Phase 5), because the pipelines need a single consistent type at the 
simulation boundary.

**Revisit in:** Phase 5, before pipelines/path1.py is written.

**RESOLVED (2026-04-08):** Converted both SimulationSpec (simulation_spec.py) and
SimulationResult (runner.py) from stdlib dataclasses to Pydantic v2 BaseModel.
Same fields, same names, `dataclasses.field(default_factory=...)` →
`pydantic.Field(default_factory=...)`. All 17 simulation tests pass unchanged.

---

## 2026-04-08 — search ingestion uses input() for now (non-interactive TODO)

**Decision:** ingest_paper() search mode uses input() for user confirmation.

**Reason:** Sufficient for interactive single-user use in Phase 4.

**Implications:** Must be replaced before any CI/batch/headless use.
Replace with an injected confirmation callback:
  confirm_fn: Callable[[list[PaperRecord]], int] = default_input_confirm
This makes it testable and swappable without changing the orchestrator logic.

**Revisit in:** Phase 5, or when first non-interactive run is needed.

**RESOLVED (2026-04-08):** Extracted input() logic into module-level
`_default_confirm(records) -> int`. Orchestrator.__init__ now accepts optional
`confirm_fn: Callable[[list[PaperRecord]], int]`, defaults to `_default_confirm`.
Tests inject lambdas directly instead of monkeypatching builtins.input.

---

## 2026-04-08 — Path 2 rate_params gap: mined reactions have no rate parameters

**Decision:** Deferred. ReactionMiningAgent extracts reaction identities 
(strings) from flux/sensitivity figures but not rate parameters. 
ModelBranchingAgent skips reactions with rate_params=None, so Path 2 
currently produces zero branches in a real run.

**Reason:** Rate parameters are not reliably present in figures — they 
appear in tables, supplementary Chemkin files, or paper body text. 
Extracting them requires a separate step not yet implemented.

**What is needed:** One of:
  a) A RateExtractionAgent that parses paper tables/text for Arrhenius 
     parameters (A, n, Ea) matched to reaction strings — LLM-assisted
  b) A database lookup (NIST, ReSpecTh) given a reaction string
  c) Extracting rate params directly from a literature Chemkin SI file

**Revisit in:** Phase 7, before live smoke test. Path 2 is blocked 
on this for real runs. Unit tests pass because rate_params is mocked


---

**Resolution:** Option C selected. Extract rate parameters directly from 
the literature Chemkin SI file during Path 1's conversion step. 
ChemkinConverter already processes this file — add a rate parameter 
extraction pass that maps reaction strings to Arrhenius params (A, n, Ea) 
before the file is converted to Cantera YAML format. Pass the resulting 
dict into Path 2 via Path1Results.

**Implication:** Path1Results needs a new field:
  extracted_rates: dict[str, dict] | None
  ← maps normalised reaction string to {A, n, Ea}
Path 2 uses this to populate CandidateReaction.rate_params before 
branching. If Path 1 did not run, Path 2 rate_params remain None 
and branches are skipped with a warning.

---

## 2026-04-08 — Deterministic extraction pipeline retained alongside LLM agent

**Decision:** simulator_agent/literature_support/ intelligence layer 
(evidence extraction → scenario building → enrichment → validation → 
planning) is migrated to src/ingestion/pipeline/ rather than deleted.

**Reason:** This deterministic pipeline is more reliable than a single 
LLM call for standard paper formats. It was incorrectly omitted from 
the Phase 2 migration scope.

**Architecture:** Deterministic pipeline runs first. LLM ConditionExtractionAgent 
runs as fallback when deterministic pipeline returns zero results.
Both outputs are merged before passing to Path 1/2 pipelines.

**Revisit in:** Phase 8b.

