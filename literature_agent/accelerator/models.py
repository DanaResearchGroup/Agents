from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, Field

class SearchQuery(BaseModel):
    topic: str
    keywords: list[str] = Field(default_factory=list)
    year_min: int | None = None
    max_results: int = 30


class PaperRecord(BaseModel):
    id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    journal: str | None = None
    doi: str | None = None
    abstract: str | None = None
    source_url: str | None = None
    oa_status: Literal["oa", "non_oa", "unknown"] = "unknown"
    oa_url: str | None = None
    relevance_score: float = Field(0.0, ge=0.0, le=1.0)
    priority: Literal["high", "medium", "low"] = "low"
    tags: list[str] = Field(default_factory=list)
    relevance_reason: list[str] = Field(default_factory=list)
    si_link_found: bool = False
    si_links: list[str] = Field(default_factory=list)
    mechanism_link_found: bool = False
    mechanism_links: list[str] = Field(default_factory=list)
    access_notes: str | None = None
    manual_review_needed: bool = False
    keep_for_manual_review: bool = False


class RegistryResult(BaseModel):
    papers: list[PaperRecord] = Field(default_factory=list)


class SourceHit(BaseModel):
    title: str
    doi: str | None = None
    url: str | None = None
    year: int | None = None
    authors: list[str] = Field(default_factory=list)
    venue: str | None = None
    provenance: str | None = None
    abstract: str | None = None
    suggested_rate: str | None = None
    reaction_formula: str | None = None
    evidence_type: list[str] = Field(default_factory=list)
    score: float = 0.0
    reasons: list[str] = Field(default_factory=list)
    match_reason: str | None = None
    likely_contains: str | None = None
    best_use: str | None = None
    si_links: list[str] = Field(default_factory=list)
    si_link_found: bool = False


class ReactionMapping(BaseModel):
    mechanism_label: str
    physical_model: str | None = None
    note: str | None = None


class RecommendedKinetics(BaseModel):
    bath_gas: str | None = None
    low_pressure_limit_k0_cm6_molecule2_s: dict[str, Any] | str | float | None = None
    high_pressure_limit_kinf_cm3_molecule_s: dict[str, Any] | str | float | None = None
    broadening: dict[str, Any] | float | str | None = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class CandidateUpdate(BaseModel):
    taxonomy: str
    action: str
    confidence: float = Field(ge=0.0, le=1.0)
    proposed_update: str | None = None
    rationale: str
    supporting_sources: list[SourceHit] = Field(default_factory=list)
    target_mismatch_link: dict[str, Any] = Field(default_factory=dict)
    reaction_mapping: list[ReactionMapping] = Field(default_factory=list)
    recommended_kinetics: RecommendedKinetics | None = None
    validity: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class Target(BaseModel):
    observable: str
    direction: str
    severity: float
    notes: str | None = None


class Conditions(BaseModel):
    reactor: str
    fuel: str
    T_K: float | None = None
    P_bar: float | None = None
    oxidizer: str | None = None


class FocusSet(BaseModel):
    species: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class FailureBrief(BaseModel):
    raw_sentence: str
    targets: list[Target] = Field(default_factory=list)
    conditions: Conditions
    focus_set: FocusSet = Field(default_factory=FocusSet)
    evidence_needs: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    case_id: str | None = None
