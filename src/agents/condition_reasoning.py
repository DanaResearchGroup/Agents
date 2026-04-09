"""ConditionReasoningAgent — extracts simulation conditions using tool-augmented reasoning.

Unlike the old ConditionExtractionAgent (which either ran deterministic-only
or LLM-only), this agent always runs the LLM but gives it access to the
deterministic pipeline as tools.  The agent calls get_executable_plans(),
get_evidence(), and validate_condition() to gather and verify conditions
before returning a structured result.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from src.agents.llm_client import LLMConfig
from src.agents.provider import make_model
from src.agents.tools.condition_tools import (
    ConditionDeps,
    get_evidence,
    get_executable_plans,
    get_paper_summary,
    validate_condition,
)
from src.schemas.experimental import PaperDocument, PaperSummary, SimConditions

logger = logging.getLogger(__name__)

_SKILL = (Path(__file__).parent / "skills" / "condition_reasoning_skill.md").read_text()


class ConditionExtractionResult(BaseModel):
    """Structured output of the ConditionReasoningAgent."""

    conditions: list[SimConditions]
    reasoning_trace: str = Field(
        default="", description="Summary of what the agent did"
    )
    source: str = Field(default="reasoning_agent")


# ── Agent definition ────────────────────────────────────────────────────────

condition_reasoning_agent: Agent[ConditionDeps, ConditionExtractionResult] = Agent(
    "test",  # placeholder model, overridden at runtime
    deps_type=ConditionDeps,
    output_type=ConditionExtractionResult,
    system_prompt=_SKILL,
    output_retries=3,
)


# ── Register tools on the agent ─────────────────────────────────────────────


@condition_reasoning_agent.tool
async def tool_get_evidence(ctx: RunContext[ConditionDeps], kind: str) -> str:
    """Extract evidence snippets of a given kind from the paper.

    kind: one of temperature, pressure, composition, experiment_family,
          observable_type, residence_time.
    Returns formatted evidence: "Page N (conf=X.XX): {value}"
    """
    return get_evidence(ctx.deps, kind)


@condition_reasoning_agent.tool
async def tool_get_executable_plans(ctx: RunContext[ConditionDeps]) -> str:
    """Run the deterministic extraction pipeline on the paper.

    Returns a formatted list of executable simulation plans with
    temperature, pressure, composition, and observable for each.
    """
    return get_executable_plans(ctx.deps)


@condition_reasoning_agent.tool
async def tool_validate_condition(
    ctx: RunContext[ConditionDeps], condition_json: str
) -> str:
    """Validate a condition JSON against the schema and model species list.

    Pass a JSON object with reactor_type, T, P, X, observable_type.
    Returns "VALID" or an error description.
    """
    return validate_condition(ctx.deps, condition_json)


@condition_reasoning_agent.tool
async def tool_get_paper_summary(ctx: RunContext[ConditionDeps]) -> str:
    """Get an overview of the paper: reactor types, species, T/P ranges.

    Returns a formatted summary or falls back to the paper abstract.
    """
    return get_paper_summary(ctx.deps)


# ── Public API ───────────────────────────────────────────────────────────────


async def extract_conditions(
    paper: PaperDocument,
    config: LLMConfig,
    model_species: list[str] | None = None,
    paper_summary: PaperSummary | None = None,
) -> list[SimConditions]:
    """Run the ConditionReasoningAgent and return extracted conditions.

    Args:
        paper: Parsed paper document.
        config: LLM configuration.
        model_species: Species names from the Cantera model (for validation).
        paper_summary: Optional PaperSummary from a prior PaperReaderAgent run.

    Returns:
        List of validated SimConditions.
    """
    model = make_model(config, agent_name="condition_extraction")
    deps = ConditionDeps(
        paper=paper,
        model_species=model_species or [],
        paper_summary=paper_summary,
    )

    result = await condition_reasoning_agent.run(
        "Extract all simulation conditions from this paper.",
        deps=deps,
        model=model,
    )

    extraction = result.output
    logger.info(
        "ConditionReasoningAgent: %d conditions | %s",
        len(extraction.conditions),
        extraction.reasoning_trace[:100] if extraction.reasoning_trace else "no trace",
    )
    return extraction.conditions
