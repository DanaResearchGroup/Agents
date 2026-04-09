"""Tests for PaperReaderAgent."""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.llm_client import LLMConfig
from src.agents.paper_reader import paper_reader_agent, read_paper
from src.agents.tools.paper_tools import PaperDeps
from src.schemas.experimental import (
    FigureCaption,
    PageText,
    PaperDocument,
    PaperSummary,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_paper() -> PaperDocument:
    return PaperDocument(
        pdf_path="/fake/paper.pdf",
        title="Shock Tube Study of H2/O2",
        abstract="We measured ignition delay times for H2/O2.",
        pages=[
            PageText(page_num=1, text="Abstract\nWe measured ignition delay times for H2/O2."),
            PageText(page_num=2, text="2. Experimental Methods\nShock tube, 10 cm ID."),
        ],
        captions=[
            FigureCaption(
                figure_id="Figure 1", label_type="Figure", page_num=2,
                caption="IDT vs temperature.",
            ),
        ],
    )


@pytest.fixture
def config() -> LLMConfig:
    return LLMConfig(provider="anthropic", model="claude-sonnet-4-6", api_key="sk-test")


@pytest.fixture
def mock_summary() -> PaperSummary:
    return PaperSummary(
        reactor_types=["shock tube"],
        species_studied=["H2", "O2"],
        temperature_range="1000-1400 K",
        pressure_range="1-10 atm",
        key_tables=["Table 1: experimental conditions"],
        key_figures=["Figure 1: IDT vs temperature"],
        experimental_setup="Shock tube with 10 cm internal diameter.",
        observable_types=["IDT"],
    )


# ── Agent has correct tools registered ───────────────────────────────────────

def test_agent_has_all_tools():
    tool_names = set(paper_reader_agent._function_toolset.tools.keys())
    expected = {
        "tool_get_abstract",
        "tool_get_section",
        "tool_search_text",
        "tool_get_figure_caption",
        "tool_get_table",
        "tool_list_figures",
        "tool_list_tables",
        "tool_list_sections",
    }
    assert expected == tool_names


def test_agent_output_type():
    assert paper_reader_agent._output_toolset is not None


# ── read_paper integration ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_read_paper_returns_summary(
    sample_paper: PaperDocument,
    config: LLMConfig,
    mock_summary: PaperSummary,
):
    mock_result = AsyncMock()
    mock_result.output = mock_summary

    with patch.object(paper_reader_agent, "run", return_value=mock_result) as mock_run:
        result = await read_paper(paper=sample_paper, config=config)

    assert isinstance(result, PaperSummary)
    assert result.reactor_types == ["shock tube"]
    assert result.temperature_range == "1000-1400 K"
    assert len(result.key_tables) == 1

    # Verify agent was called with correct deps.
    call_kwargs = mock_run.call_args
    assert isinstance(call_kwargs.kwargs["deps"], PaperDeps)
    assert call_kwargs.kwargs["deps"].paper is sample_paper


@pytest.mark.asyncio
async def test_read_paper_passes_model_species(
    sample_paper: PaperDocument,
    config: LLMConfig,
    mock_summary: PaperSummary,
):
    mock_result = AsyncMock()
    mock_result.output = mock_summary

    with patch.object(paper_reader_agent, "run", return_value=mock_result) as mock_run:
        await read_paper(
            paper=sample_paper, config=config, model_species=["H2", "O2"]
        )

    deps = mock_run.call_args.kwargs["deps"]
    assert deps.model_species == ["H2", "O2"]


@pytest.mark.asyncio
async def test_read_paper_uses_agent_name_override(
    sample_paper: PaperDocument,
    mock_summary: PaperSummary,
):
    """make_model should be called with agent_name='paper_reader'."""
    mock_result = AsyncMock()
    mock_result.output = mock_summary

    config = LLMConfig(
        provider="anthropic",
        model="claude-sonnet-4-6",
        api_key="sk-test",
        agent_overrides={"paper_reader": {"model": "claude-haiku-4-5"}},
    )

    with (
        patch.object(paper_reader_agent, "run", return_value=mock_result),
        patch("src.agents.paper_reader.make_model") as mock_make_model,
    ):
        mock_make_model.return_value = "mock-model"
        await read_paper(paper=sample_paper, config=config)

    mock_make_model.assert_called_once_with(config, agent_name="paper_reader")


@pytest.mark.asyncio
async def test_read_paper_logs_summary(
    sample_paper: PaperDocument,
    config: LLMConfig,
    mock_summary: PaperSummary,
    caplog,
):
    mock_result = AsyncMock()
    mock_result.output = mock_summary

    with patch.object(paper_reader_agent, "run", return_value=mock_result):
        import logging
        with caplog.at_level(logging.INFO, logger="src.agents.paper_reader"):
            await read_paper(paper=sample_paper, config=config)

    assert "PaperReaderAgent" in caplog.text
    assert "shock tube" in caplog.text


@pytest.mark.asyncio
async def test_read_paper_prompt_under_100_words(
    sample_paper: PaperDocument,
    config: LLMConfig,
    mock_summary: PaperSummary,
):
    """Initial prompt must not contain paper text (under 100 words)."""
    mock_result = AsyncMock()
    mock_result.output = mock_summary

    with patch.object(paper_reader_agent, "run", return_value=mock_result) as mock_run:
        await read_paper(paper=sample_paper, config=config)

    prompt = mock_run.call_args.args[0]
    assert len(prompt.split()) < 100, f"Prompt too long ({len(prompt.split())} words)"
