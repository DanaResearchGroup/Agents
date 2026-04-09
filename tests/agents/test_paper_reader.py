"""Tests for PaperReaderAgent."""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.llm_client import LLMConfig
from src.agents.paper_reader import _parse_markdown_fallback, paper_reader_agent, read_paper
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
def mock_summary_json() -> str:
    return (
        '{"reactor_types":["shock tube"],"species_studied":["H2","O2"],'
        '"temperature_range":"1000-1400 K","pressure_range":"1-10 atm",'
        '"key_tables":["Table 1: experimental conditions"],'
        '"key_figures":["Figure 1: IDT vs temperature"],'
        '"experimental_setup":"Shock tube with 10 cm internal diameter.",'
        '"observable_types":["IDT"]}'
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


def test_agent_output_type_is_str():
    """Agent uses str output to allow manual JSON parsing."""
    assert paper_reader_agent._output_type is str


# ── read_paper integration ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_read_paper_returns_summary(
    sample_paper: PaperDocument,
    config: LLMConfig,
    mock_summary_json: str,
):
    mock_result = AsyncMock()
    mock_result.output = mock_summary_json

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
async def test_read_paper_parses_markdown_wrapped_json(
    sample_paper: PaperDocument,
    config: LLMConfig,
    mock_summary_json: str,
):
    """Agent output wrapped in markdown fences still parses correctly."""
    mock_result = AsyncMock()
    mock_result.output = f"```json\n{mock_summary_json}\n```"

    with patch.object(paper_reader_agent, "run", return_value=mock_result):
        result = await read_paper(paper=sample_paper, config=config)

    assert isinstance(result, PaperSummary)
    assert result.reactor_types == ["shock tube"]


@pytest.mark.asyncio
async def test_read_paper_handles_think_tags(
    sample_paper: PaperDocument,
    config: LLMConfig,
    mock_summary_json: str,
):
    """Think tags before JSON are stripped before parsing."""
    mock_result = AsyncMock()
    mock_result.output = f"<think>Let me analyze...</think>\n{mock_summary_json}"

    with patch.object(paper_reader_agent, "run", return_value=mock_result):
        result = await read_paper(paper=sample_paper, config=config)

    assert isinstance(result, PaperSummary)
    assert result.temperature_range == "1000-1400 K"


@pytest.mark.asyncio
async def test_read_paper_markdown_fallback(
    sample_paper: PaperDocument,
    config: LLMConfig,
):
    """Markdown output triggers fallback parser instead of empty summary."""
    mock_result = AsyncMock()
    mock_result.output = (
        "## Summary\n\n"
        "**Reactors:** Shock tube\n\n"
        "**Species:** NH3, H2, Ar\n\n"
        "**Temperature range:** 2100–3100 K\n\n"
        "**Pressure:** ~1 atm\n"
    )

    with patch.object(paper_reader_agent, "run", return_value=mock_result):
        result = await read_paper(paper=sample_paper, config=config)

    assert isinstance(result, PaperSummary)
    assert "shock_tube" in result.reactor_types
    assert result.temperature_range == "2100-3100 K"


@pytest.mark.asyncio
async def test_read_paper_passes_model_species(
    sample_paper: PaperDocument,
    config: LLMConfig,
    mock_summary_json: str,
):
    mock_result = AsyncMock()
    mock_result.output = mock_summary_json

    with patch.object(paper_reader_agent, "run", return_value=mock_result) as mock_run:
        await read_paper(
            paper=sample_paper, config=config, model_species=["H2", "O2"]
        )

    deps = mock_run.call_args.kwargs["deps"]
    assert deps.model_species == ["H2", "O2"]


@pytest.mark.asyncio
async def test_read_paper_uses_agent_name_override(
    sample_paper: PaperDocument,
    mock_summary_json: str,
):
    """make_model should be called with agent_name='paper_reader'."""
    mock_result = AsyncMock()
    mock_result.output = mock_summary_json

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
    mock_summary_json: str,
    caplog,
):
    mock_result = AsyncMock()
    mock_result.output = mock_summary_json

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
    mock_summary_json: str,
):
    """Initial prompt must not contain paper text (under 100 words)."""
    mock_result = AsyncMock()
    mock_result.output = mock_summary_json

    with patch.object(paper_reader_agent, "run", return_value=mock_result) as mock_run:
        await read_paper(paper=sample_paper, config=config)

    prompt = mock_run.call_args.args[0]
    assert len(prompt.split()) < 100, f"Prompt too long ({len(prompt.split())} words)"


# ── _parse_markdown_fallback ─────────────────────────────────────────────────


def test_markdown_fallback_extracts_temperature_range():
    text = "Experiments at 1200-2400 K and 1-10 atm."
    result = _parse_markdown_fallback(text)
    assert result.temperature_range == "1200-2400 K"
    assert result.pressure_range == "1-10 atm"


def test_markdown_fallback_extracts_single_pressure():
    text = "Shock tube study at ~1 atm, 2100–3100 K."
    result = _parse_markdown_fallback(text)
    assert result.pressure_range == "1 atm"
    assert result.temperature_range == "2100-3100 K"


def test_markdown_fallback_extracts_reactor_types():
    text = "Jet-stirred reactor and shock tube experiments."
    result = _parse_markdown_fallback(text)
    assert "jsr" in result.reactor_types
    assert "shock_tube" in result.reactor_types


def test_markdown_fallback_extracts_species():
    text = "Species studied: NH3, H2, O2, and Ar in mixtures."
    result = _parse_markdown_fallback(text)
    assert "NH3" in result.species_studied
    assert "H2" in result.species_studied
    assert "Ar" in result.species_studied


def test_markdown_fallback_extracts_tables_and_figures():
    text = (
        "Table 1 lists conditions. Table 2 shows rate constants.\n"
        "Figure 3 plots IDT vs temperature."
    )
    result = _parse_markdown_fallback(text)
    assert len(result.key_tables) == 2
    assert len(result.key_figures) == 1


def test_markdown_fallback_empty_text():
    result = _parse_markdown_fallback("No useful content here at all.")
    assert isinstance(result, PaperSummary)
    assert result.temperature_range == ""
    assert result.reactor_types == []


def test_markdown_fallback_real_qwen_output():
    """Test against the actual output format qwen3.5:4b produces."""
    text = (
        '## Summary\n\n'
        'The paper "NH₃ Pyrolysis" focuses on ammonia kinetics.\n\n'
        '**Reactors:** Shock tube with laser absorption\n\n'
        '**Species studied:** NH₃, H₂, and Ar\n\n'
        '**Temperature range:** 2100–3100 K\n\n'
        '**Pressure:** Near atmospheric pressure, with variations between 0.88–1.26 atm\n\n'
        '**Key tables:**\n'
        '- **Table 1** (p.6): Reactions in the mechanism\n'
        '- **Table 2** (p.10): Experimental data\n\n'
        '**Key figures:**\n'
        '- **Figure 5** (p.7): NH₃ speciation profiles\n'
    )
    result = _parse_markdown_fallback(text)
    assert "shock_tube" in result.reactor_types
    assert result.temperature_range == "2100-3100 K"
    assert "0.88-1.26 atm" in result.pressure_range
    assert "NH3" in result.species_studied  # unicode subscripts normalised
    assert "H2" in result.species_studied
    assert "Ar" in result.species_studied
    assert len(result.key_tables) >= 1
    assert "**" not in result.key_tables[0]  # bold markers stripped
    assert len(result.key_figures) >= 1


def test_markdown_fallback_atmospheric_pressure():
    """'atmospheric pressure' with no number maps to '1 atm'."""
    text = "Experiments at atmospheric pressure, 1200-1800 K."
    result = _parse_markdown_fallback(text)
    assert result.pressure_range == "1 atm"


def test_markdown_fallback_unicode_species():
    """Unicode subscript digits are normalised to ASCII."""
    text = "Species: N₂H₂, H₂O, NO₂ at 1000-2000 K."
    result = _parse_markdown_fallback(text)
    assert "N2H2" in result.species_studied
    assert "H2O" in result.species_studied
    assert "NO2" in result.species_studied
