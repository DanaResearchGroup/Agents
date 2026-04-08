"""Shared Pydantic v2 models for the entire pipeline.

Section 1: Simulation-facing models (from ARCHITECTURE.md 3.1)
Section 2: Ingestion-facing models (migrated from simulator_agent/models.py)

Every inter-component data structure lives here. Nothing outside
src/schemas/ should define Pydantic models.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


# ═══════════════════════════════════════════════════════════════════════════
# Section 1 — Simulation-facing models (ARCHITECTURE.md §3.1)
# ═══════════════════════════════════════════════════════════════════════════


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
    mixture: dict[str, float]
    observable_type: ObservableType
    observable_label: str | None = None
    measured_value: float | list[float]
    error_threshold: float


class ExperimentalDataset(BaseModel):
    name: str
    conditions: list[ExperimentalCondition]
    source: str | None = None


class SimConditions(BaseModel):
    """Validated, simulation-ready form of ExperimentalCondition."""
    reactor_type: ReactorType
    T: float
    P: float
    X: dict[str, float]
    observable_type: ObservableType
    observable_label: str | None = None


class SimResult(BaseModel):
    conditions: SimConditions
    simulated_value: float
    units: str
    success: bool
    error_message: str | None = None


class MAEResult(BaseModel):
    model_name: str
    conditions_id: str
    mae: float
    threshold: float
    passed: bool


# ═══════════════════════════════════════════════════════════════════════════
# Section 2 — Ingestion-facing models (migrated from simulator_agent/models.py)
# ═══════════════════════════════════════════════════════════════════════════


# ── Validation helper ────────────────────────────────────────────────────


class ValidationResult(BaseModel):
    """Result of domain-specific validation on a model instance."""
    valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


# ── Parser models ────────────────────────────────────────────────────────


class PageText(BaseModel):
    """A single page of extracted text."""
    page_num: int
    text: str


class FigureCaption(BaseModel):
    """A figure or table caption extracted from the paper."""
    figure_id: str
    label_type: str
    page_num: int
    caption: str


# ── Table models ─────────────────────────────────────────────────────────


ConditionValueMode = Literal["single", "range", "set", "formula"]


class TableConditionValue(BaseModel):
    """A structured condition value from a table cell."""
    mode: ConditionValueMode
    raw_text: str
    values: list[float] = Field(default_factory=list)
    unit: str | None = None
    formula: str | None = None


class TableColumn(BaseModel):
    """Column definition for a parsed condition table."""
    header: str
    role: str
    species: str | None = None
    unit: str | None = None


class TableRow(BaseModel):
    """One row of a parsed condition table."""
    row_index: int
    row_label: str | None = None
    cells: dict[str, TableConditionValue]


class ParsedTable(BaseModel):
    """A structured condition table extracted from the PDF."""
    table_id: str
    page_num: int
    caption: str
    columns: list[TableColumn]
    rows: list[TableRow]


class PaperDocument(BaseModel):
    """Structured representation of a parsed PDF paper."""
    pdf_path: str
    title: str | None = None
    abstract: str | None = None
    pages: list[PageText]
    captions: list[FigureCaption]
    tables: list[ParsedTable] = Field(default_factory=list)

    def full_text(self) -> str:
        """Concatenate all page texts for whole-document searches."""
        return "\n".join(p.text for p in self.pages)


class SectionSpan(BaseModel):
    """A detected section boundary in the paper."""
    name: str
    page_start: int
    page_end: int
    line_start: int
    weight: float


# ── Evidence models ──────────────────────────────────────────────────────


EvidenceKind = Literal[
    "experiment_family",
    "observable_type",
    "temperature",
    "pressure",
    "residence_time",
    "equivalence_ratio",
    "composition",
    "mechanism",
    "flow_rate",
    "reactor_volume",
]

EVIDENCE_KINDS: tuple[EvidenceKind, ...] = (
    "experiment_family",
    "observable_type",
    "temperature",
    "pressure",
    "residence_time",
    "equivalence_ratio",
    "composition",
    "mechanism",
    "flow_rate",
    "reactor_volume",
)


EvidenceRole = Literal[
    "setup_range",
    "setpoint",
    "measurement",
    "constraint",
    "instrument",
    "background",
    "unclassified",
]

ROLE_PRIORITY: dict[str, int] = {
    "setup_range": 5,
    "setpoint": 4,
    "measurement": 2,
    "constraint": 1,
    "instrument": 0,
    "background": 0,
    "unclassified": 1,
}


class EvidenceSnippet(BaseModel):
    """A simulation-relevant piece of evidence extracted from the paper."""
    kind: EvidenceKind
    value_text: str
    page_num: int
    source_text: str
    confidence: float = Field(ge=0.0, le=1.0)
    normalized_value: str | float | dict | None = None
    section: str | None = None
    section_weight: float = 1.0
    role: EvidenceRole = "unclassified"


# ── Builder models ───────────────────────────────────────────────────────


ExperimentFamily = Literal[
    "jsr", "shock_tube", "flow_reactor", "rcm", "flame", "pfr", "unknown",
]

LegacyObservableType = Literal[
    "ignition_delay", "species_profile", "conversion", "temperature_profile",
    "flame_speed", "half_life", "unknown",
]

PlanningPriority = Literal["high", "medium", "low"]


class ExperimentCandidate(BaseModel):
    """A grouped cluster of evidence for one experiment family."""
    candidate_id: str
    experiment_family: ExperimentFamily
    confirmed_observables: list[LegacyObservableType]
    possible_observables: list[LegacyObservableType]
    section: str | None = None
    page_start: int
    page_end: int
    evidence: list[EvidenceSnippet]
    confidence: float = Field(ge=0.0, le=1.0)
    is_primary: bool
    simulatable: bool
    planning_priority: PlanningPriority
    notes: list[str] = Field(default_factory=list)


# ── Scenario models ──────────────────────────────────────────────────────


ScenarioType = Literal["single_point", "range", "sweep"]
ScenarioSource = Literal["figure_caption", "table", "methods_mixture", "results_cluster"]
ScenarioStatus = Literal["accepted", "needs_review", "rejected"]


class FieldConfidence(BaseModel):
    """Per-field confidence breakdown for a scenario."""
    composition: float = Field(default=0.0, ge=0.0, le=1.0)
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    pressure: float = Field(default=0.0, ge=0.0, le=1.0)
    family: float = Field(default=0.0, ge=0.0, le=1.0)
    observable: float = Field(default=0.0, ge=0.0, le=1.0)
    panel_extraction: float | None = Field(default=None, ge=0.0, le=1.0)

    @property
    def overall(self) -> float:
        """Minimum of core fields — weakest link determines reliability."""
        core = [self.composition, self.temperature, self.pressure]
        return min(core) if all(c > 0 for c in core) else 0.0


class ExperimentalScenario(BaseModel):
    """One coherent experimental condition set that can be simulated."""
    scenario_id: str
    experiment_family: ExperimentFamily
    scenario_type: ScenarioType
    source: ScenarioSource
    source_label: str
    source_page: int
    temperature_text: str | None = None
    pressure_text: str | None = None
    composition_text: str | None = None
    residence_time_text: str | None = None
    equivalence_ratio_text: str | None = None
    observable: str | None = None
    simulatable: bool
    status: ScenarioStatus = "accepted"
    confidence: FieldConfidence = Field(default_factory=FieldConfidence)
    review_reasons: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    table_row_id: str | None = None
    table_conditions: dict | None = None
    expansion_state: str | None = None


# ── Normalizer models ────────────────────────────────────────────────────


ConditionMode = Literal["setpoint", "range", "sweep"]


class NormalizedCondition(BaseModel):
    """A structured, machine-readable condition value."""
    raw_text: str
    kind: str
    role: str
    is_range: bool
    min_value: float
    max_value: float
    unit: str
    source_section: str | None = None
    source_page: int

    @property
    def mode(self) -> ConditionMode:
        if self.is_range and self.min_value != self.max_value:
            return "range"
        return "setpoint"

    @model_validator(mode="after")
    def _check_min_max(self) -> NormalizedCondition:
        if self.min_value > self.max_value:
            raise ValueError(
                f"min_value ({self.min_value}) > max_value ({self.max_value})"
            )
        return self


class NormalizedComposition(BaseModel):
    """Structured composition with species fractions."""
    species: dict[str, float]
    balance_species: str | None = None
    composition_complete: bool
    raw_mentions: list[str]
    source_pages: list[int]

    @model_validator(mode="after")
    def _check_fractions(self) -> NormalizedComposition:
        for sp, frac in self.species.items():
            if frac < 0:
                raise ValueError(f"negative fraction for {sp}: {frac}")
            if frac > 1.0:
                raise ValueError(f"fraction > 1.0 for {sp}: {frac}")
        total = sum(self.species.values())
        if total > 1.01:
            raise ValueError(f"species fractions sum to {total:.4f} (> 1.0)")
        return self


# ── Planner models ───────────────────────────────────────────────────────


TemplateFamily = Literal[
    "idt_const_uv",
    "idt_const_hp",
    "jsr_const_tp",
    "pfr_const_tp",
    "flame_speed",
    "batch_const_tp",
    "unsupported",
]

PlanStatus = Literal["executable", "draft", "blocked"]


class SimulationPlan(BaseModel):
    """The planner's output: a simulation setup with explicit uncertainty.

    Status policy:
      executable — all fields present, auto-generation safe
      draft      — plan exists but some fields need confirmation
      blocked    — cannot build a plan (missing critical fields or unsupported)
    """
    scenario_id: str
    experiment_family: ExperimentFamily
    template_family: TemplateFamily
    plan_status: PlanStatus
    temperature: NormalizedCondition | None = None
    pressure: NormalizedCondition | None = None
    composition: NormalizedComposition | None = None
    residence_time: NormalizedCondition | None = None
    equivalence_ratio: NormalizedCondition | None = None
    target_observables: list[LegacyObservableType]
    mechanism: str | None = None
    assumptions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    blocking_fields: list[str] = Field(default_factory=list)
    auto_generation_allowed: bool

    @model_validator(mode="after")
    def _check_status_consistency(self) -> SimulationPlan:
        if self.plan_status == "executable":
            if self.missing_fields:
                raise ValueError(
                    f"executable plan has missing fields: {self.missing_fields}"
                )
            if self.blocking_fields:
                raise ValueError(
                    f"executable plan has blocking fields: {self.blocking_fields}"
                )
            if not self.auto_generation_allowed:
                raise ValueError("executable plan has auto_generation_allowed=False")
        if self.template_family == "unsupported" and self.plan_status != "blocked":
            raise ValueError("unsupported template but plan not blocked")
        return self


# ── Reaction mining models ───────────────────────────────────────────────


class FigureClassification(BaseModel):
    """Result of classifying a scientific figure by type."""
    label: str
    confidence: float = 0.0
    reasoning: str = ""


class FluxPathway(BaseModel):
    """A single dominant pathway extracted from a flux diagram."""
    rank: int
    from_species: str = Field(alias="from")
    to_species: str = Field(alias="to")
    importance: str = "unknown"
    evidence: str = ""

    model_config = {"populate_by_name": True}


class FluxConditions(BaseModel):
    """Operating conditions shown in a flux diagram."""
    temperature: str = "unknown"
    pressure: str = "unknown"
    equivalence_ratio: str = "unknown"
    residence_time: str = "unknown"


class FluxInterpretation(BaseModel):
    """Structured interpretation of a flux / reaction-path diagram."""
    system: str = "unknown"
    conditions: FluxConditions = Field(default_factory=FluxConditions)
    major_species: list[str] = Field(default_factory=list)
    dominant_pathways: list[FluxPathway] = Field(default_factory=list)
    quantitative_info: bool = False
    usefulness: str = "unknown"
    use_cases: list[str] = Field(default_factory=list)
    uncertainty: str = ""
    confidence: float = 0.0


class SensitiveReaction(BaseModel):
    """A single reaction from a sensitivity analysis ranking."""
    rank: int
    reaction: str
    coefficient: float | str = "unknown"
    impact: str = "unknown"


class SensitivityConditions(BaseModel):
    """Operating conditions for a sensitivity analysis."""
    temperature: str = "unknown"
    pressure: str = "unknown"
    equivalence_ratio: str = "unknown"


class SensitivityInterpretation(BaseModel):
    """Structured interpretation of a sensitivity analysis figure."""
    target_property: str = "unknown"
    conditions: SensitivityConditions = Field(default_factory=SensitivityConditions)
    top_reactions: list[SensitiveReaction] = Field(default_factory=list)
    usefulness: str = "unknown"
    uncertainty: str = ""
    confidence: float = 0.0


# ── Orchestrator models ────────────────────────────────────────────────────


class PaperSource(BaseModel):
    """How to locate the paper: by DOI, local file upload, or search query."""
    mode: Literal["doi", "upload", "search"]
    value: str = Field(..., description="DOI string, file path, or search query")


class ConversionResult(BaseModel):
    """Result of a Chemkin → Cantera YAML conversion attempt."""

    success: bool
    output_path: Path | None = None
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    attempts: int = 0


class RunConfig(BaseModel):
    """Top-level configuration for an orchestrator run."""
    original_model: Path
    experimental_data: Path = Field(description="Path to experimental YAML file")
    paper_source: PaperSource
    run_path1: bool = True
    run_path2: bool = True
    output_dir: Path = Path("data/reports")
    llm_config: Path = Path("config/llm_config.yaml")
