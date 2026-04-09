"""ConditionReasoningAgent — validates pre-extracted plans into SimConditions.

Uses a plan-first architecture: the deterministic extraction pipeline runs
BEFORE the agent, and the agent receives compact plan summaries in its initial
prompt. The agent only calls tools when a plan has missing or uncertain values.

The agent returns a raw JSON string which is parsed with _extract_json_array()
to handle models (e.g. qwen3) that leak thinking text into their output.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from pydantic_ai import Agent, RunContext

from src.agents.llm_client import LLMConfig
from src.agents.provider import make_model
from src.agents.utils import strip_thinking
from src.agents.tools.condition_tools import (
    ConditionDeps,
    get_evidence,
    get_page,
    get_paper_summary,
    validate_condition,
)
from src.ingestion.pipeline.extractor import PaperExtractionPipeline
from src.schemas.experimental import (
    PaperDocument,
    PaperSummary,
    SimConditions,
    SimulationPlan,
)

logger = logging.getLogger(__name__)

_SKILL = (Path(__file__).parent / "skills" / "condition_reasoning_skill.md").read_text()


# ── JSON extraction helper ──────────────────────────────────────────────────


def _extract_json_array(text: str) -> str:
    """Extract a JSON array from text that may contain preamble or markdown.

    Handles models that wrap valid JSON in explanatory text, markdown fences,
    or thinking tokens.
    """
    text = text.strip()

    # Strip markdown code fences if present.
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    # Direct parse — already a JSON array.
    if text.startswith("["):
        return text

    # Find JSON array in text.
    match = re.search(r"\[[\s\S]*\]", text)
    if match:
        return match.group(0)

    # Find JSON object and wrap it.
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        return f"[{match.group(0)}]"

    return "[]"


# ── Agent definition ────────────────────────────────────────────────────────

condition_reasoning_agent: Agent[ConditionDeps, str] = Agent(
    "test",  # placeholder model, overridden at runtime
    deps_type=ConditionDeps,
    output_type=str,
    system_prompt=_SKILL,
    output_retries=3,
)


# ── Register tools on the agent ─────────────────────────────────────────────


@condition_reasoning_agent.tool
async def tool_get_evidence(
    ctx: RunContext[ConditionDeps], kind: str, page: int | None = None
) -> str:
    """Extract evidence snippets of a given kind from the paper.

    kind: one of temperature, pressure, composition, experiment_family,
          observable_type, residence_time.
    page: optional page number to restrict search to.
    Returns formatted evidence: "Page N (conf=X.XX): {value}"
    """
    return get_evidence(ctx.deps, kind, page=page)


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


@condition_reasoning_agent.tool
async def tool_get_page(ctx: RunContext[ConditionDeps], page_num: int) -> str:
    """Return text of a specific page from the paper (max 1500 chars).

    Use when you need to verify a specific value from a plan's source page.
    """
    return get_page(ctx.deps, page_num)


# ── Plan formatting ─────────────────────────────────────────────────────────


def format_plans(plans: list[SimulationPlan]) -> str:
    """Format plans compactly for the initial prompt (~400 tokens for 20 plans)."""
    lines: list[str] = []
    for i, p in enumerate(plans):
        t = p.temperature
        pr = p.pressure
        c = p.composition
        T = f"{t.min_value}K" if t else "T=?"
        P = f"{pr.min_value}atm" if pr else "P=?"
        X = str(c.species) if c else "X=?"
        obs = p.target_observables[0] if p.target_observables else "?"
        src = f"p.{t.source_page}" if t else ""
        lines.append(
            f"Plan {i}: {p.experiment_family} | "
            f"T={T} P={P} | X={X} | obs={obs} {src}"
        )
    return "\n".join(lines)


# ── Public API ───────────────────────────────────────────────────────────────


async def extract_conditions(
    paper: PaperDocument,
    config: LLMConfig,
    model_species: list[str] | None = None,
    paper_summary: PaperSummary | None = None,
) -> list[SimConditions]:
    """Run the ConditionReasoningAgent and return extracted conditions.

    Runs the deterministic extraction pipeline first, then passes the
    compact plan summaries to the agent for validation and conversion.
    """
    model = make_model(config, agent_name="condition_extraction")

    # Pre-extract plans deterministically.
    pipeline = PaperExtractionPipeline()
    plans = pipeline.extract(paper)
    logger.info("Deterministic pipeline: %d plans extracted", len(plans))

    deps = ConditionDeps(
        paper=paper,
        model_species=model_species or [],
        paper_summary=paper_summary,
        executable_plans=plans,
    )

    # Build prompt with compact plan summaries.
    if plans:
        plans_text = format_plans(plans)
        prompt = (
            f"Review these {len(plans)} pre-extracted simulation "
            f"plans and convert valid ones to SimConditions.\n\n"
            f"{plans_text}\n\n"
            f"For each plan: validate species names against the "
            f"model, resolve any uncertainties using tools, "
            f"then return the validated conditions."
        )
    else:
        prompt = (
            "No pre-extracted plans found. Use get_paper_summary() "
            "and get_evidence() tools to find simulation conditions."
        )

    result = await condition_reasoning_agent.run(
        prompt,
        deps=deps,
        model=model,
    )

    # Parse the raw string output — handles models that wrap JSON in text.
    raw_text = strip_thinking(result.output)
    json_str = _extract_json_array(raw_text)

    try:
        conditions_data = json.loads(json_str)
        conditions = [SimConditions(**c) for c in conditions_data]
    except Exception as e:
        logger.warning("Failed to parse conditions: %s, raw: %s", e, raw_text[:200])
        conditions = []

    logger.info("ConditionReasoningAgent: %d conditions", len(conditions))
    return conditions
