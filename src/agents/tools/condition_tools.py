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

    # Cached pipeline results (populated lazily by tools).
    _evidence: list[EvidenceSnippet] | None = None
    _plans: list[SimulationPlan] | None = None

    model_config = {"arbitrary_types_allowed": True}


# ── Tool implementations ────────────────────────────────────────────────────
# Plain functions — registered on the Agent via @agent.tool in the agent module.


def get_evidence(deps: ConditionDeps, kind: str) -> str:
    """Extract evidence snippets of a given kind from the paper.

    Args:
        kind: One of temperature, pressure, composition, experiment_family,
              observable_type, residence_time.

    Returns:
        Formatted evidence snippets: "Page N (conf=X.XX): {source_text}"
    """
    try:
        if deps._evidence is None:
            deps._evidence = extract_evidence(deps.paper)

        filtered = [s for s in deps._evidence if s.kind == kind]
        if not filtered:
            return f"No {kind} evidence found in the paper."

        lines: list[str] = []
        for s in sorted(filtered, key=lambda x: x.confidence, reverse=True):
            display = s.normalized_value if s.normalized_value is not None else s.value_text
            lines.append(f"Page {s.page_num} (conf={s.confidence:.2f}): {display}")
        return "\n".join(lines)
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
        for i, plan in enumerate(deps._plans):
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

            lines.append(" | ".join(parts))

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
            return "\n".join(parts) if parts else "Paper summary available but empty."

        if deps.paper.abstract:
            return f"Abstract: {deps.paper.abstract}"

        if deps.paper.pages:
            return f"First page: {deps.paper.pages[0].text[:500]}"

        return "No paper summary available."
    except Exception as e:
        return f"Tool error: {e}"
