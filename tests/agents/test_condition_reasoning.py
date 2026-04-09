"""Tests for ConditionReasoningAgent."""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.condition_reasoning import (
    ConditionExtractionResult,
    condition_reasoning_agent,
    extract_conditions,
)
from src.agents.llm_client import LLMConfig
from src.schemas.experimental import (
    ObservableType,
    PageText,
    PaperDocument,
    PaperSummary,
    ReactorType,
    SimConditions,
)

# All tests mock agent.run() and make_model to avoid real LLM calls.
MOCK_MAKE_MODEL = patch(
    "src.agents.condition_reasoning.make_model", return_value=MagicMock()
)


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture()
def paper() -> PaperDocument:
    return PaperDocument(
        pdf_path="/tmp/paper.pdf",
        title="Test Paper",
        abstract="NH3 combustion study",
        pages=[PageText(page_num=1, text="T=1200K P=1atm")],
        captions=[],
    )


@pytest.fixture()
def config() -> LLMConfig:
    return LLMConfig()


TWO_CONDITIONS = [
    SimConditions(
        reactor_type=ReactorType.SHOCK_TUBE,
        T=1200.0,
        P=1.0,
        X={"NH3": 0.01, "Ar": 0.99},
        observable_type=ObservableType.IDT,
    ),
    SimConditions(
        reactor_type=ReactorType.SHOCK_TUBE,
        T=1500.0,
        P=1.0,
        X={"NH3": 0.01, "Ar": 0.99},
        observable_type=ObservableType.IDT,
    ),
]


# ── Test: extract_conditions returns conditions from agent ──────────────────


@pytest.mark.asyncio
async def test_extract_conditions_returns_list(
    paper: PaperDocument, config: LLMConfig
):
    """extract_conditions returns list of SimConditions from agent output."""
    mock_result = MagicMock()
    mock_result.output = ConditionExtractionResult(
        conditions=TWO_CONDITIONS,
        reasoning_trace="Called get_executable_plans, found 2 plans",
        source="reasoning_agent",
    )

    with (
        MOCK_MAKE_MODEL,
        patch.object(
            condition_reasoning_agent, "run", new=AsyncMock(return_value=mock_result)
        ),
    ):
        result = await extract_conditions(paper, config)

    assert len(result) == 2
    assert result[0].T == 1200.0
    assert result[1].T == 1500.0
    assert all(isinstance(c, SimConditions) for c in result)


# ── Test: extract_conditions with model_species ─────────────────────────────


@pytest.mark.asyncio
async def test_extract_conditions_passes_model_species(
    paper: PaperDocument, config: LLMConfig
):
    """model_species is forwarded to the agent deps."""
    mock_result = MagicMock()
    mock_result.output = ConditionExtractionResult(conditions=TWO_CONDITIONS)

    mock_run = AsyncMock(return_value=mock_result)

    with (
        MOCK_MAKE_MODEL,
        patch.object(condition_reasoning_agent, "run", new=mock_run),
    ):
        await extract_conditions(
            paper, config, model_species=["NH3(1)", "O2(6)", "Ar"]
        )

    # Check deps passed to agent.run()
    call_kwargs = mock_run.call_args
    deps = call_kwargs.kwargs["deps"]
    assert deps.model_species == ["NH3(1)", "O2(6)", "Ar"]


# ── Test: extract_conditions with paper_summary ─────────────────────────────


@pytest.mark.asyncio
async def test_extract_conditions_passes_paper_summary(
    paper: PaperDocument, config: LLMConfig
):
    """paper_summary is forwarded to the agent deps."""
    summary = PaperSummary(
        reactor_types=["shock_tube"],
        species_studied=["NH3"],
    )

    mock_result = MagicMock()
    mock_result.output = ConditionExtractionResult(conditions=TWO_CONDITIONS)

    mock_run = AsyncMock(return_value=mock_result)

    with (
        MOCK_MAKE_MODEL,
        patch.object(condition_reasoning_agent, "run", new=mock_run),
    ):
        await extract_conditions(paper, config, paper_summary=summary)

    deps = mock_run.call_args.kwargs["deps"]
    assert deps.paper_summary is not None
    assert deps.paper_summary.reactor_types == ["shock_tube"]


# ── Test: extract_conditions returns empty list ─────────────────────────────


@pytest.mark.asyncio
async def test_extract_conditions_empty_result(
    paper: PaperDocument, config: LLMConfig
):
    """Agent returning empty conditions returns empty list."""
    mock_result = MagicMock()
    mock_result.output = ConditionExtractionResult(
        conditions=[],
        reasoning_trace="No conditions found in paper",
    )

    with (
        MOCK_MAKE_MODEL,
        patch.object(
            condition_reasoning_agent, "run", new=AsyncMock(return_value=mock_result)
        ),
    ):
        result = await extract_conditions(paper, config)

    assert result == []


# ── Test: logging includes condition count ──────────────────────────────────


@pytest.mark.asyncio
async def test_extract_conditions_logs_count(
    paper: PaperDocument, config: LLMConfig, caplog
):
    """Log message includes number of conditions extracted."""
    mock_result = MagicMock()
    mock_result.output = ConditionExtractionResult(
        conditions=TWO_CONDITIONS,
        reasoning_trace="Found 2 conditions via plans",
    )

    with (
        MOCK_MAKE_MODEL,
        patch.object(
            condition_reasoning_agent, "run", new=AsyncMock(return_value=mock_result)
        ),
        caplog.at_level(logging.INFO, logger="src.agents.condition_reasoning"),
    ):
        await extract_conditions(paper, config)

    assert any("2 conditions" in r.message for r in caplog.records)


# ── Test: ConditionExtractionResult schema ──────────────────────────────────


def test_extraction_result_defaults():
    """ConditionExtractionResult has sensible defaults."""
    result = ConditionExtractionResult(conditions=[])
    assert result.reasoning_trace == ""
    assert result.source == "reasoning_agent"


def test_extraction_result_with_conditions():
    """ConditionExtractionResult holds SimConditions list."""
    result = ConditionExtractionResult(
        conditions=TWO_CONDITIONS,
        reasoning_trace="Used deterministic plans as primary source",
    )
    assert len(result.conditions) == 2
    assert result.conditions[0].reactor_type == ReactorType.SHOCK_TUBE
