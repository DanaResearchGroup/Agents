"""Condition extraction tools for the ConditionReasoningAgent.

These wrap the deterministic extraction pipeline as callable PydanticAI
tools so the reasoning agent can gather evidence before making decisions.
"""

from __future__ import annotations

import json
import logging

from pydantic import BaseModel, Field

from src.schemas.experimental import (
    EvidenceSnippet,
    PaperDocument,
    PaperSummary,
    SimConditions,
    SimulationPlan,
)
from src.ingestion.pipeline.extractor import PaperExtractionPipeline
from src.ingestion.pipeline.evidence_extractor import extract_evidence

logger = logging.getLogger(__name__)


class ConditionDeps(BaseModel):
    """Dependencies passed to condition tools via RunContext."""

    paper: PaperDocument
    model_species: list[str] = Field(default_factory=list)
    paper_summary: PaperSummary | None = None
    executable_plans: list[SimulationPlan] = Field(default_factory=list)

    # Cached pipeline results (populated lazily by tools).
    _evidence: list[EvidenceSnippet] | None = None
    _plans: list[SimulationPlan] | None = None

    model_config = {"arbitrary_types_allowed": True}


def _truncate(text: str, limit: int, label: str = "") -> str:
    """Truncate text to limit chars, logging if truncated."""
    if len(text) <= limit:
        return text
    logger.debug("Truncated %s from %d to %d chars", label, len(text), limit)
    return text[:limit] + f"\n... [truncated, {len(text) - limit} chars omitted]"


# ── Tool implementations ────────────────────────────────────────────────────
# Plain functions — registered on the Agent via @agent.tool in the agent module.


def get_evidence(deps: ConditionDeps, kind: str, page: int | None = None) -> str:
    """Extract evidence snippets of a given kind from the paper.

    Args:
        kind: One of temperature, pressure, composition, experiment_family,
              observable_type, residence_time.
        page: If provided, only return evidence from this page number.

    Returns:
        Formatted evidence snippets: "Page N (conf=X.XX): {source_text}"
    """
    try:
        if deps._evidence is None:
            deps._evidence = extract_evidence(deps.paper)

        filtered = [s for s in deps._evidence if s.kind == kind]
        if page is not None:
            filtered = [s for s in filtered if s.page_num == page]
        if not filtered:
            msg = f"No {kind} evidence found"
            if page is not None:
                msg += f" on page {page}"
            return msg + "."

        filtered_top = sorted(filtered, key=lambda x: x.confidence, reverse=True)[:5]
        lines: list[str] = []
        for s in filtered_top:
            display = s.normalized_value if s.normalized_value is not None else s.value_text
            lines.append(_truncate(f"Page {s.page_num} (conf={s.confidence:.2f}): {display}", 200, "evidence"))
        return "\n".join(lines)
    except Exception as e:
        return f"Tool error: {e}"


def get_page(deps: ConditionDeps, page_num: int) -> str:
    """Return text of a specific page from the paper.

    Args:
        page_num: 1-based page number.

    Returns:
        Page text truncated to 1500 chars.
    """
    try:
        for page in deps.paper.pages:
            if page.page_num == page_num:
                return _truncate(page.text, 1500, f"page:{page_num}")
        return f"Page {page_num} not found. Paper has {len(deps.paper.pages)} pages."
    except Exception as e:
        return f"Tool error: {e}"


def get_executable_plans(deps: ConditionDeps) -> str:
    """Run the deterministic extraction pipeline and return formatted plans.

    Returns:
        Formatted list of executable SimulationPlans.
    """
    try:
        if deps._plans is None:
            pipeline = PaperExtractionPipeline()
            deps._plans = pipeline.extract(deps.paper)

        if not deps._plans:
            return "No executable simulation plans found."

        lines: list[str] = []
        for i, plan in enumerate(deps._plans[:10]):
            parts = [f"Plan {i + 1}: {plan.experiment_family}"]

            if plan.temperature:
                if plan.temperature.min_value == plan.temperature.max_value:
                    parts.append(f"T={plan.temperature.min_value}{plan.temperature.unit}")
                else:
                    parts.append(
                        f"T={plan.temperature.min_value}-"
                        f"{plan.temperature.max_value}{plan.temperature.unit}"
                    )

            if plan.pressure:
                parts.append(f"P={plan.pressure.min_value}{plan.pressure.unit}")

            if plan.composition:
                species = ", ".join(
                    f"{k}={v}" for k, v in plan.composition.species.items()
                )
                parts.append(f"composition={{{species}}}")

            if plan.target_observables:
                parts.append(f"observable={plan.target_observables[0]}")

            lines.append(_truncate(" | ".join(parts), 150, f"plan:{i+1}"))

        return "\n".join(lines)
    except Exception as e:
        return f"Tool error: {e}"


def validate_condition(deps: ConditionDeps, condition_json: str) -> str:
    """Validate a condition JSON string against SimConditions schema and model species.

    Args:
        condition_json: JSON string representing a single condition.

    Returns:
        "VALID" or a description of what's wrong.
    """
    try:
        try:
            data = json.loads(condition_json)
        except json.JSONDecodeError as e:
            return f"Invalid JSON: {e}"

        try:
            cond = SimConditions.model_validate(data)
        except Exception as e:
            return f"Schema validation error: {e}"

        # Check species against model if species list is provided.
        if deps.model_species:
            unknown = [s for s in cond.X if s not in deps.model_species]
            if unknown:
                return f"Unknown species not in model: {unknown}"

        # Basic sanity checks.
        x_sum = sum(cond.X.values())
        if abs(x_sum - 1.0) > 0.01:
            return f"Mole fractions sum to {x_sum:.4f}, expected 1.0 ±0.01"

        return "VALID"
    except Exception as e:
        return f"Tool error: {e}"


def get_paper_summary(deps: ConditionDeps) -> str:
    """Return the paper summary as formatted text.

    Falls back to the paper abstract if no summary is available.
    """
    try:
        if deps.paper_summary is not None:
            parts = []
            if deps.paper_summary.reactor_types:
                parts.append(f"Reactor types: {', '.join(deps.paper_summary.reactor_types)}")
            if deps.paper_summary.species_studied:
                parts.append(f"Species: {', '.join(deps.paper_summary.species_studied)}")
            if deps.paper_summary.temperature_range:
                parts.append(f"Temperature: {deps.paper_summary.temperature_range}")
            if deps.paper_summary.pressure_range:
                parts.append(f"Pressure: {deps.paper_summary.pressure_range}")
            if deps.paper_summary.observable_types:
                parts.append(f"Observables: {', '.join(deps.paper_summary.observable_types)}")
            if deps.paper_summary.experimental_setup:
                parts.append(f"Setup: {deps.paper_summary.experimental_setup}")
            return _truncate("\n".join(parts), 800, "paper_summary") if parts else "Paper summary available but empty."

        if deps.paper.abstract:
            return _truncate(f"Abstract: {deps.paper.abstract}", 800, "paper_summary")

        if deps.paper.pages:
            return _truncate(f"First page: {deps.paper.pages[0].text}", 800, "paper_summary")

        return "No paper summary available."
    except Exception as e:
        return f"Tool error: {e}"
