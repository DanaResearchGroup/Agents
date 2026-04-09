"""Tests for ConditionReasoningAgent."""

from __future__ import annotations

import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.condition_reasoning import (
    _extract_json_array,
    condition_reasoning_agent,
    extract_conditions,
    format_plans,
)
from src.agents.llm_client import LLMConfig
from src.schemas.experimental import (
    NormalizedComposition,
    NormalizedCondition,
    ObservableType,
    PageText,
    PaperDocument,
    PaperSummary,
    ReactorType,
    SimConditions,
    SimulationPlan,
)

# All tests mock agent.run() and make_model to avoid real LLM calls.
MOCK_MAKE_MODEL = patch(
    "src.agents.condition_reasoning.make_model", return_value=MagicMock()
)

# Mock the pipeline so we control what plans are passed.
MOCK_PIPELINE = patch(
    "src.agents.condition_reasoning.PaperExtractionPipeline",
)

# A valid JSON array string that the mocked agent returns.
TWO_CONDITIONS_JSON = json.dumps([
    {
        "reactor_type": "shock_tube", "T": 1200.0, "P": 1.0,
        "X": {"NH3": 0.01, "Ar": 0.99}, "observable_type": "idt",
    },
    {
        "reactor_type": "shock_tube", "T": 1500.0, "P": 1.0,
        "X": {"NH3": 0.01, "Ar": 0.99}, "observable_type": "idt",
    },
])


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


def _make_plan(**overrides) -> SimulationPlan:
    defaults = dict(
        scenario_id="scenario_1",
        experiment_family="shock_tube",
        template_family="idt_const_uv",
        plan_status="executable",
        temperature=NormalizedCondition(
            raw_text="1200 K", kind="temperature", role="setpoint",
            is_range=False, min_value=1200.0, max_value=1200.0,
            unit="K", source_page=1,
        ),
        pressure=NormalizedCondition(
            raw_text="1 atm", kind="pressure", role="setpoint",
            is_range=False, min_value=1.0, max_value=1.0,
            unit="atm", source_page=1,
        ),
        composition=NormalizedComposition(
            species={"NH3": 0.01, "Ar": 0.99},
            balance_species=None, composition_complete=True,
            raw_mentions=["1% NH3 in Ar"], source_pages=[1],
        ),
        target_observables=["ignition_delay"],
        auto_generation_allowed=True,
    )
    defaults.update(overrides)
    return SimulationPlan(**defaults)


def _mock_agent_run(raw_output: str) -> AsyncMock:
    """Create a mock agent.run() that returns raw_output as result.output."""
    mock_result = MagicMock()
    mock_result.output = raw_output
    return AsyncMock(return_value=mock_result)


# ── Test: _extract_json_array ───────────────────────────────────────────────


def test_extract_json_array_clean():
    """Extracts array from clean JSON."""
    raw = '[{"T": 1200}]'
    assert _extract_json_array(raw) == raw


def test_extract_json_array_markdown_fences():
    """Extracts array from markdown-wrapped JSON."""
    raw = '```json\n[{"T": 1200}]\n```'
    assert _extract_json_array(raw) == '[{"T": 1200}]'


def test_extract_json_array_preamble():
    """Extracts array from text with preamble."""
    raw = 'Based on the search results, here are the conditions:\n[{"T": 1200}]'
    result = _extract_json_array(raw)
    assert result == '[{"T": 1200}]'


def test_extract_json_array_single_object():
    """Wraps a single JSON object in an array."""
    raw = 'Here is the condition: {"T": 1200}'
    result = _extract_json_array(raw)
    assert result == '[{"T": 1200}]'


def test_extract_json_array_no_json():
    """Returns empty array when no JSON found."""
    assert _extract_json_array("No conditions found in this paper.") == "[]"


def test_extract_json_array_empty_input():
    """Returns empty array for empty string."""
    assert _extract_json_array("") == "[]"


def test_extract_json_array_whitespace():
    """Handles leading/trailing whitespace."""
    raw = '  \n  [{"T": 1200}]  \n  '
    assert _extract_json_array(raw) == '[{"T": 1200}]'


# ── Test: extract_conditions returns conditions from agent ──────────────────


@pytest.mark.asyncio
async def test_extract_conditions_returns_list(
    paper: PaperDocument, config: LLMConfig
):
    """extract_conditions returns list of SimConditions from agent output."""
    with (
        MOCK_MAKE_MODEL,
        MOCK_PIPELINE as mock_pipe_cls,
        patch.object(
            condition_reasoning_agent, "run",
            new=_mock_agent_run(TWO_CONDITIONS_JSON),
        ),
    ):
        mock_pipe_cls.return_value.extract.return_value = [_make_plan()]
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
    mock_run = _mock_agent_run("[]")

    with (
        MOCK_MAKE_MODEL,
        MOCK_PIPELINE as mock_pipe_cls,
        patch.object(condition_reasoning_agent, "run", new=mock_run),
    ):
        mock_pipe_cls.return_value.extract.return_value = []
        await extract_conditions(
            paper, config, model_species=["NH3(1)", "O2(6)", "Ar"]
        )

    deps = mock_run.call_args.kwargs["deps"]
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
    mock_run = _mock_agent_run("[]")

    with (
        MOCK_MAKE_MODEL,
        MOCK_PIPELINE as mock_pipe_cls,
        patch.object(condition_reasoning_agent, "run", new=mock_run),
    ):
        mock_pipe_cls.return_value.extract.return_value = []
        await extract_conditions(paper, config, paper_summary=summary)

    deps = mock_run.call_args.kwargs["deps"]
    assert deps.paper_summary is not None
    assert deps.paper_summary.reactor_types == ["shock_tube"]


# ── Test: extract_conditions returns empty list ─────────────────────────────


@pytest.mark.asyncio
async def test_extract_conditions_empty_result(
    paper: PaperDocument, config: LLMConfig
):
    """Agent returning empty array returns empty list."""
    with (
        MOCK_MAKE_MODEL,
        MOCK_PIPELINE as mock_pipe_cls,
        patch.object(
            condition_reasoning_agent, "run",
            new=_mock_agent_run("[]"),
        ),
    ):
        mock_pipe_cls.return_value.extract.return_value = []
        result = await extract_conditions(paper, config)

    assert result == []


# ── Test: extract_conditions handles preamble text ──────────────────────────


@pytest.mark.asyncio
async def test_extract_conditions_handles_preamble(
    paper: PaperDocument, config: LLMConfig
):
    """Agent returning JSON wrapped in preamble still parses correctly."""
    raw = f"Based on my analysis:\n{TWO_CONDITIONS_JSON}\nThese are valid."
    with (
        MOCK_MAKE_MODEL,
        MOCK_PIPELINE as mock_pipe_cls,
        patch.object(
            condition_reasoning_agent, "run",
            new=_mock_agent_run(raw),
        ),
    ):
        mock_pipe_cls.return_value.extract.return_value = [_make_plan()]
        result = await extract_conditions(paper, config)

    assert len(result) == 2


# ── Test: extract_conditions handles garbage output gracefully ──────────────


@pytest.mark.asyncio
async def test_extract_conditions_handles_garbage(
    paper: PaperDocument, config: LLMConfig
):
    """Agent returning unparseable text returns empty list."""
    with (
        MOCK_MAKE_MODEL,
        MOCK_PIPELINE as mock_pipe_cls,
        patch.object(
            condition_reasoning_agent, "run",
            new=_mock_agent_run("I cannot help with that."),
        ),
    ):
        mock_pipe_cls.return_value.extract.return_value = []
        result = await extract_conditions(paper, config)

    assert result == []


# ── Test: logging includes condition count ──────────────────────────────────


@pytest.mark.asyncio
async def test_extract_conditions_logs_count(
    paper: PaperDocument, config: LLMConfig, caplog
):
    """Log message includes number of conditions extracted."""
    with (
        MOCK_MAKE_MODEL,
        MOCK_PIPELINE as mock_pipe_cls,
        patch.object(
            condition_reasoning_agent, "run",
            new=_mock_agent_run(TWO_CONDITIONS_JSON),
        ),
        caplog.at_level(logging.INFO, logger="src.agents.condition_reasoning"),
    ):
        mock_pipe_cls.return_value.extract.return_value = [_make_plan()]
        await extract_conditions(paper, config)

    assert any("2 conditions" in r.message for r in caplog.records)


# ── Test: executable_plans passed in deps ───────────────────────────────────


@pytest.mark.asyncio
async def test_extract_conditions_passes_executable_plans(
    paper: PaperDocument, config: LLMConfig
):
    """Pre-extracted plans are passed as deps.executable_plans."""
    plans = [_make_plan(), _make_plan(scenario_id="scenario_2")]
    mock_run = _mock_agent_run("[]")

    with (
        MOCK_MAKE_MODEL,
        MOCK_PIPELINE as mock_pipe_cls,
        patch.object(condition_reasoning_agent, "run", new=mock_run),
    ):
        mock_pipe_cls.return_value.extract.return_value = plans
        await extract_conditions(paper, config)

    deps = mock_run.call_args.kwargs["deps"]
    assert len(deps.executable_plans) == 2


# ── Test: format_plans produces compact string ──────────────────────────────


def test_format_plans_compact_output():
    """format_plans produces one line per plan with key fields."""
    plans = [_make_plan()]
    result = format_plans(plans)
    assert "Plan 0: shock_tube" in result
    assert "T=1200.0K" in result
    assert "P=1.0atm" in result
    assert "NH3" in result
    assert "ignition_delay" in result
    assert "p.1" in result  # source page


def test_format_plans_marks_missing_fields():
    """format_plans shows ? for missing T, P, or X."""
    plan = _make_plan(temperature=None, pressure=None, composition=None)
    result = format_plans([plan])
    assert "T=?" in result
    assert "P=?" in result
    assert "X=?" in result


def test_format_plans_empty():
    """format_plans returns empty string for no plans."""
    assert format_plans([]) == ""


# ── Test: initial prompt contains plan summary ──────────────────────────────


@pytest.mark.asyncio
async def test_prompt_contains_plan_summary(
    paper: PaperDocument, config: LLMConfig
):
    """When plans exist, the prompt includes formatted plan summaries."""
    plans = [_make_plan()]
    mock_run = _mock_agent_run("[]")

    with (
        MOCK_MAKE_MODEL,
        MOCK_PIPELINE as mock_pipe_cls,
        patch.object(condition_reasoning_agent, "run", new=mock_run),
    ):
        mock_pipe_cls.return_value.extract.return_value = plans
        await extract_conditions(paper, config)

    prompt = mock_run.call_args.args[0]
    assert "1 pre-extracted" in prompt
    assert "Plan 0: shock_tube" in prompt
    assert "T=1200.0K" in prompt


@pytest.mark.asyncio
async def test_prompt_no_plans_fallback(
    paper: PaperDocument, config: LLMConfig
):
    """When no plans, prompt tells agent to use tools."""
    mock_run = _mock_agent_run("[]")

    with (
        MOCK_MAKE_MODEL,
        MOCK_PIPELINE as mock_pipe_cls,
        patch.object(condition_reasoning_agent, "run", new=mock_run),
    ):
        mock_pipe_cls.return_value.extract.return_value = []
        await extract_conditions(paper, config)

    prompt = mock_run.call_args.args[0]
    assert "No pre-extracted plans" in prompt
    assert "get_paper_summary()" in prompt


# ── Test: prompt does NOT contain full paper text ───────────────────────────


@pytest.mark.asyncio
async def test_prompt_excludes_paper_text():
    """Even with a large paper, prompt stays reasonable."""
    paper = PaperDocument(
        pdf_path="/tmp/paper.pdf", title="Big Paper",
        pages=[PageText(page_num=i, text="X" * 5000) for i in range(10)],
        captions=[],
    )
    config = LLMConfig(provider="anthropic", model="claude-sonnet-4-6", api_key="sk-test")
    mock_run = _mock_agent_run("[]")

    with (
        MOCK_MAKE_MODEL,
        MOCK_PIPELINE as mock_pipe_cls,
        patch.object(condition_reasoning_agent, "run", new=mock_run),
    ):
        mock_pipe_cls.return_value.extract.return_value = [_make_plan()]
        await extract_conditions(paper=paper, config=config)

    prompt = mock_run.call_args.args[0]
    assert "XXXXX" not in prompt
    assert len(prompt) < 2000


# ── Test: agent has correct tools registered ────────────────────────────────


def test_agent_has_expected_tools():
    tool_names = set(condition_reasoning_agent._function_toolset.tools.keys())
    expected = {
        "tool_get_evidence",
        "tool_validate_condition",
        "tool_get_paper_summary",
        "tool_get_page",
    }
    assert expected == tool_names


def test_agent_output_type_is_str():
    """Agent returns raw string, not a structured Pydantic model."""
    # The output type is checked via the agent's output toolset being None
    # (str output doesn't need a tool).
    assert condition_reasoning_agent._output_type is str
