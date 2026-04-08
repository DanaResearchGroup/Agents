"""Shared data models for the simulator agent pipeline.

parse_pdf → extract_evidence → build_experiment_candidates → build_simulation_plan
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Literal


# ── Validation result (used by all validate() methods) ────────────────────

@dataclass
class ValidationResult:
    """Result of validating a model instance."""
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ── Parser models ─────────────────────────────────────────────────────────

@dataclass
class PageText:
    """A single page of extracted text."""
    page_num: int  # 1-indexed
    text: str


@dataclass
class FigureCaption:
    """A figure or table caption extracted from the paper."""
    figure_id: str       # e.g. "Figure 1", "Table 2"
    label_type: str      # "figure" or "table"
    page_num: int        # 1-indexed
    caption: str


# ── Table models ─────────────────────────────────────────────────────────

ConditionValueMode = Literal["single", "range", "set", "formula"]


@dataclass
class TableConditionValue:
    """A structured condition value from a table cell."""
    mode: ConditionValueMode
    raw_text: str
    values: list[float] = field(default_factory=list)
    unit: str | None = None
    formula: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TableColumn:
    """Column definition for a parsed condition table."""
    header: str          # raw header text, e.g. "CH4 (ppm)"
    role: str            # "temperature", "pressure", "residence_time", "equivalence_ratio", "species", "identifier"
    species: str | None = None   # for species columns, e.g. "CH4"
    unit: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TableRow:
    """One row of a parsed condition table."""
    row_index: int
    row_label: str | None        # e.g. "1", "Case 1"
    cells: dict[str, TableConditionValue]  # keyed by role or "species:CH4"

    def to_dict(self) -> dict:
        return {
            "row_index": self.row_index,
            "row_label": self.row_label,
            "cells": {k: v.to_dict() for k, v in self.cells.items()},
        }


@dataclass
class ParsedTable:
    """A structured condition table extracted from the PDF."""
    table_id: str        # e.g. "Table 2"
    page_num: int
    caption: str
    columns: list[TableColumn]
    rows: list[TableRow]

    def to_dict(self) -> dict:
        return {
            "table_id": self.table_id,
            "page_num": self.page_num,
            "caption": self.caption,
            "columns": [c.to_dict() for c in self.columns],
            "rows": [r.to_dict() for r in self.rows],
        }


@dataclass
class PaperDocument:
    """Structured representation of a parsed PDF paper."""
    pdf_path: str
    title: str | None
    abstract: str | None
    pages: list[PageText]
    captions: list[FigureCaption]
    tables: list[ParsedTable] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def full_text(self) -> str:
        """Concatenate all page texts for whole-document searches."""
        return "\n".join(p.text for p in self.pages)


@dataclass
class SectionSpan:
    """A detected section boundary in the paper."""
    name: str         # "abstract", "introduction", "methods", "results", "conclusion", "references"
    page_start: int   # 1-indexed
    page_end: int     # 1-indexed, inclusive
    line_start: int   # approximate line offset on page_start
    weight: float     # trust weight for evidence scoring


# ── Extractor models ──────────────────────────────────────────────────────

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
    "setup_range",   # "experiments were conducted over 2100-3000 K" — highest priority
    "setpoint",      # "at 1 atm", "T = 1000 K" — operating condition
    "measurement",   # "ignition delay at 1200 K" — result, not setup
    "constraint",    # "below 1200 K" — bound, not value
    "instrument",    # "gauge range 0-1.333 kPa" — equipment spec
    "background",    # from intro/literature discussion
    "unclassified",  # not yet role-classified
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


@dataclass
class EvidenceSnippet:
    """A simulation-relevant piece of evidence extracted from the paper."""
    kind: EvidenceKind
    value_text: str                                    # raw extracted, e.g. "950-1150 K"
    page_num: int                                      # 1-indexed
    source_text: str                                   # surrounding context for traceability
    confidence: float                                  # 0.0 to 1.0
    normalized_value: str | float | dict | None = None # e.g. 101325.0 for "1 atm"
    section: str | None = None                         # e.g. "methods", "introduction"
    section_weight: float = 1.0                        # section-based trust multiplier
    role: EvidenceRole = "unclassified"                # semantic role of this evidence

    def to_dict(self) -> dict:
        return asdict(self)


# ── Builder models ────────────────────────────────────────────────────────

ExperimentFamily = Literal[
    "jsr", "shock_tube", "flow_reactor", "rcm", "flame", "pfr", "unknown",
]

ObservableType = Literal[
    "ignition_delay", "species_profile", "conversion", "temperature_profile",
    "flame_speed", "half_life", "unknown",
]

PlanningPriority = Literal["high", "medium", "low"]


@dataclass
class ExperimentCandidate:
    """A grouped cluster of evidence for one experiment family."""
    candidate_id: str
    experiment_family: ExperimentFamily
    confirmed_observables: list[ObservableType]  # strong evidence in methods/results
    possible_observables: list[ObservableType]   # mentioned nearby but weaker
    section: str | None               # primary section where anchor was found
    page_start: int
    page_end: int
    evidence: list[EvidenceSnippet]
    confidence: float                 # aggregate confidence
    is_primary: bool                  # True if from methods, not lit review
    simulatable: bool                 # True if enough evidence for planning
    planning_priority: PlanningPriority
    notes: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


# ── Scenario models ──────────────────────────────────────────────────────

ScenarioType = Literal["single_point", "range", "sweep"]
ScenarioSource = Literal["figure_caption", "table", "methods_mixture", "results_cluster"]
ScenarioStatus = Literal["accepted", "needs_review", "rejected"]


@dataclass
class FieldConfidence:
    """Per-field confidence breakdown for a scenario."""
    composition: float = 0.0
    temperature: float = 0.0
    pressure: float = 0.0
    family: float = 0.0
    observable: float = 0.0
    panel_extraction: float | None = None  # only for multi-panel scenarios

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.panel_extraction is None:
            del d["panel_extraction"]
        return d

    @property
    def overall(self) -> float:
        """Minimum of core fields — weakest link determines reliability."""
        core = [self.composition, self.temperature, self.pressure]
        return min(core) if all(c > 0 for c in core) else 0.0

    def validate(self) -> ValidationResult:
        errors = []
        for name in ("composition", "temperature", "pressure", "family", "observable"):
            val = getattr(self, name)
            if not (0.0 <= val <= 1.0):
                errors.append(f"{name} confidence {val} not in [0, 1]")
        if self.panel_extraction is not None and not (0.0 <= self.panel_extraction <= 1.0):
            errors.append(f"panel_extraction confidence {self.panel_extraction} not in [0, 1]")
        return ValidationResult(valid=len(errors) == 0, errors=errors)


@dataclass
class ExperimentalScenario:
    """One coherent experimental condition set that can be simulated."""
    scenario_id: str
    experiment_family: ExperimentFamily
    scenario_type: ScenarioType
    source: ScenarioSource
    source_label: str                     # e.g. "Figure 1", "Table 2 row 3"
    source_page: int
    temperature_text: str | None          # raw, e.g. "2217 K" or "2096-2487 K"
    pressure_text: str | None             # raw, e.g. "1.22 atm"
    composition_text: str | None          # raw, e.g. "0.45% NH3 in Ar"
    residence_time_text: str | None
    equivalence_ratio_text: str | None
    observable: str | None                # e.g. "species_profile", "ignition_delay"
    simulatable: bool
    status: ScenarioStatus = "accepted"
    confidence: FieldConfidence = field(default_factory=FieldConfidence)
    review_reasons: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    # Table-specific: structured conditions preserving multi-valued cells
    table_row_id: str | None = None
    table_conditions: dict | None = None   # {role: {mode, values, unit, ...}}
    expansion_state: str | None = None     # "unexpanded" for multi-valued rows

    def to_dict(self) -> dict:
        d = {}
        for k, v in self.__dict__.items():
            if hasattr(v, 'to_dict'):
                d[k] = v.to_dict()
            else:
                d[k] = v
        return d


# ── Normalizer models ─────────────────────────────────────────────────────

ConditionMode = Literal["setpoint", "range", "sweep"]


@dataclass
class NormalizedCondition:
    """A structured, machine-readable condition value."""
    raw_text: str
    kind: str                          # "temperature", "pressure", etc.
    role: str                          # "setup_range", "setpoint", etc.
    is_range: bool
    min_value: float
    max_value: float
    unit: str
    source_section: str | None
    source_page: int

    @property
    def mode(self) -> ConditionMode:
        if self.is_range and self.min_value != self.max_value:
            return "range"
        return "setpoint"

    def validate(self) -> ValidationResult:
        errors = []
        if self.min_value > self.max_value:
            errors.append(f"min_value ({self.min_value}) > max_value ({self.max_value})")
        if not self.unit:
            errors.append("unit is empty")
        if not self.kind:
            errors.append("kind is empty")
        return ValidationResult(valid=len(errors) == 0, errors=errors)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["mode"] = self.mode
        return d


@dataclass
class NormalizedComposition:
    """Structured composition with species fractions."""
    species: dict[str, float]          # e.g. {"NH3": 0.005, "Ar": 0.995}
    balance_species: str | None        # e.g. "Ar" if remainder assigned
    composition_complete: bool         # all fractions sum to ~1.0?
    raw_mentions: list[str]            # original text evidence
    source_pages: list[int]

    def validate(self) -> ValidationResult:
        errors, warnings = [], []
        for sp, frac in self.species.items():
            if frac < 0:
                errors.append(f"negative fraction for {sp}: {frac}")
            if frac > 1.0:
                errors.append(f"fraction > 1.0 for {sp}: {frac}")
        total = sum(self.species.values())
        if total > 1.01:
            errors.append(f"species fractions sum to {total:.4f} (> 1.0)")
        if not self.composition_complete and total < 0.99:
            warnings.append(f"composition incomplete: fractions sum to {total:.4f}")
        return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)

    def to_dict(self) -> dict:
        return asdict(self)


# ── Planner models ────────────────────────────────────────────────────────

TemplateFamily = Literal[
    "idt_const_uv",       # shock tube IDT (constant volume, adiabatic)
    "idt_const_hp",       # RCM IDT (constant pressure, adiabatic)
    "jsr_const_tp",       # JSR (constant T, P — steady-state)
    "pfr_const_tp",       # plug flow reactor
    "flame_speed",        # freely propagating flame
    "batch_const_tp",     # generic batch reactor
    "unsupported",
]


PlanStatus = Literal["executable", "draft", "blocked"]


@dataclass
class SimulationPlan:
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
    temperature: NormalizedCondition | None
    pressure: NormalizedCondition | None
    composition: NormalizedComposition | None
    residence_time: NormalizedCondition | None
    equivalence_ratio: NormalizedCondition | None
    target_observables: list[ObservableType]
    mechanism: str | None
    assumptions: list[str]
    warnings: list[str]
    missing_fields: list[str]
    blocking_fields: list[str]             # fields that prevent auto-execution
    auto_generation_allowed: bool

    def validate(self) -> ValidationResult:
        errors, warnings = [], []
        # Status consistency
        if self.plan_status == "executable":
            if self.missing_fields:
                errors.append(f"executable plan has missing fields: {self.missing_fields}")
            if self.blocking_fields:
                errors.append(f"executable plan has blocking fields: {self.blocking_fields}")
            if not self.auto_generation_allowed:
                errors.append("executable plan has auto_generation_allowed=False")
        elif self.plan_status == "draft":
            if not self.missing_fields and not self.blocking_fields:
                warnings.append("draft plan has no missing or blocking fields — could be executable")
        # Condition validation
        for cond in (self.temperature, self.pressure, self.residence_time, self.equivalence_ratio):
            if cond is not None:
                r = cond.validate()
                errors.extend(r.errors)
        if self.composition is not None:
            r = self.composition.validate()
            errors.extend(r.errors)
            warnings.extend(r.warnings)
        # Template check
        if self.template_family == "unsupported" and self.plan_status != "blocked":
            errors.append("unsupported template but plan not blocked")
        return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)

    def to_dict(self) -> dict:
        d = {}
        for key, val in self.__dict__.items():
            if val is None:
                d[key] = None
            elif hasattr(val, 'to_dict'):
                d[key] = val.to_dict()
            elif isinstance(val, list) and val and hasattr(val[0], 'to_dict'):
                d[key] = [v.to_dict() for v in val]
            else:
                d[key] = val
        return d
