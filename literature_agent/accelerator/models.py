from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field

class PaperRecord(BaseModel):
    id: str = Field(..., description="Unique paper ID")
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    journal: str | None = None
    venue: str | None = None
    doi: str | None = None
    abstract: str | None = None
    source_url: str | None = None
    oa_status: Literal["oa", "non_oa", "unknown"] = "unknown"
    oa_url: str | None = None
    priority: Literal["high", "medium", "low"] = "low"
    tags: list[str] = Field(default_factory=list)
    si_link_found: bool = False
    si_links: list[str] = Field(default_factory=list)
    mechanism_link_found: bool = False
    mechanism_links: list[str] = Field(default_factory=list)
    access_notes: str | None = None
    manual_review_needed: bool = False
    keep_for_manual_review: bool = False
    provenance: str | None = None
    match_reason: str | None = None
    likely_contains: str | None = None
    best_use: str | None = None
    references: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)

class SearchQuery(BaseModel):
    topic: str
    max_results: int = 30
    keywords: list[str] = Field(default_factory=list)
    snowball: bool = False

class RegistryResult(BaseModel):
    papers: list[PaperRecord] = Field(default_factory=list)

class FailureBrief(BaseModel):
    raw_sentence: str
    case_id: str = "default"
    targets: list[dict] = Field(default_factory=list)
    conditions: dict = Field(default_factory=dict)
    focus_set: dict = Field(default_factory=dict)
    evidence_needs: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
