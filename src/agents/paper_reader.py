"""PaperReaderAgent — reads a paper and produces a structured PaperSummary.

This is the first pass over a paper. It uses tools to explore the document
and outputs a summary that downstream agents (condition extraction, reaction
mining) consume as context.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from pydantic_ai import Agent, RunContext

from src.agents.llm_client import LLMConfig
from src.agents.provider import make_model
from src.agents.utils import extract_json_object, strip_thinking
from src.agents.tools.paper_tools import (
    PaperDeps,
    get_abstract,
    get_figure_caption,
    get_section,
    get_table,
    list_figures,
    list_sections,
    list_tables,
    search_text,
)
from src.schemas.experimental import PaperDocument, PaperSummary

logger = logging.getLogger(__name__)

_SKILL = (Path(__file__).parent / "skills" / "paper_reader_skill.md").read_text()

paper_reader_agent: Agent[PaperDeps, str] = Agent(
    "test",  # placeholder model, overridden at runtime
    deps_type=PaperDeps,
    output_type=str,
    system_prompt=_SKILL,
    output_retries=3,
)


# ── Register tools on the agent ─────────────────────────────────────────────


@paper_reader_agent.tool
async def tool_get_abstract(ctx: RunContext[PaperDeps]) -> str:
    """Return the paper abstract or the first 500 characters of page 1."""
    return get_abstract(ctx.deps)


@paper_reader_agent.tool
async def tool_get_section(ctx: RunContext[PaperDeps], section_name: str) -> str:
    """Find and return a section by name (case-insensitive partial match)."""
    return get_section(ctx.deps, section_name)


@paper_reader_agent.tool
async def tool_search_text(ctx: RunContext[PaperDeps], query: str) -> str:
    """Search all page text for passages containing query terms (max 5 hits)."""
    return search_text(ctx.deps, query)


@paper_reader_agent.tool
async def tool_get_figure_caption(ctx: RunContext[PaperDeps], figure_id: str) -> str:
    """Look up a figure caption. Accepts 'Figure 1', 'Fig. 1', or '1'."""
    return get_figure_caption(ctx.deps, figure_id)


@paper_reader_agent.tool
async def tool_get_table(ctx: RunContext[PaperDeps], table_id: str) -> str:
    """Return a parsed table as formatted text. Accepts 'Table 1' or '1'."""
    return get_table(ctx.deps, table_id)


@paper_reader_agent.tool
async def tool_list_figures(ctx: RunContext[PaperDeps]) -> str:
    """List all figure IDs and first 100 chars of each caption."""
    return list_figures(ctx.deps)


@paper_reader_agent.tool
async def tool_list_tables(ctx: RunContext[PaperDeps]) -> str:
    """List all table IDs and captions."""
    return list_tables(ctx.deps)


@paper_reader_agent.tool
async def tool_list_sections(ctx: RunContext[PaperDeps]) -> str:
    """List all detected section names with page ranges."""
    return list_sections(ctx.deps)


# ── Markdown fallback parser ─────────────────────────────────────────────────

_REACTOR_ALIASES: dict[str, str] = {
    "shock tube": "shock_tube",
    "shock-tube": "shock_tube",
    "st": "shock_tube",
    "jet-stirred reactor": "jsr",
    "jet stirred reactor": "jsr",
    "plug flow reactor": "pfr",
    "flow reactor": "pfr",
    "rapid compression machine": "rcm",
    "laminar flame": "flame",
    "flame speed": "flame",
}

# Unicode subscript digits → ASCII (e.g. NH₃ → NH3)
_SUBSCRIPT_MAP = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")


def _normalize_unicode(text: str) -> str:
    """Replace Unicode subscripts with ASCII digits."""
    return text.translate(_SUBSCRIPT_MAP)


def _clean_md_bold(text: str) -> str:
    """Strip markdown bold markers from a string."""
    return text.replace("**", "")


def _parse_markdown_fallback(text: str) -> PaperSummary:
    """Best-effort extraction from markdown when JSON parsing fails.

    Scrapes temperature/pressure ranges, reactor types, and species
    from the free-text output that small models produce instead of JSON.
    """
    logger.info("Attempting markdown fallback parse")
    # Normalize unicode subscripts so H₂ → H2, NH₃ → NH3
    text = _normalize_unicode(text)

    # Temperature range — look for patterns like "2100-3100 K" or "2100–3100 K"
    temp_match = re.search(
        r"(\d{3,4})\s*[-–—to]+\s*(\d{3,4})\s*K", text
    )
    temperature_range = f"{temp_match.group(1)}-{temp_match.group(2)} K" if temp_match else ""

    # Pressure range — "1-10 atm" or "~1 atm" or "0.88-1.26 atm"
    press_match = re.search(
        r"([\d.]+)\s*[-–—to]+\s*([\d.]+)\s*atm", text
    )
    if press_match:
        pressure_range = f"{press_match.group(1)}-{press_match.group(2)} atm"
    else:
        single_press = re.search(r"~?\s*([\d.]+)\s*atm", text)
        if single_press:
            pressure_range = f"{single_press.group(1)} atm"
        elif re.search(r"atmospheric\s+pressure", text, re.IGNORECASE):
            pressure_range = "1 atm"
        else:
            pressure_range = ""

    # Reactor types
    reactor_types: list[str] = []
    text_lower = text.lower()
    for alias, canonical in _REACTOR_ALIASES.items():
        if alias in text_lower and canonical not in reactor_types:
            reactor_types.append(canonical)

    # Species — chemical formulas (start with uppercase, contain digits or lowercase)
    species_matches = re.findall(
        r"\b([A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)*)\b", text
    )
    # Filter to plausible species (contain a digit or are known short names)
    known_short = {"Ar", "He", "Ne", "Kr", "Xe", "N2", "O2", "H2"}
    species = []
    seen: set[str] = set()
    for s in species_matches:
        if s in seen:
            continue
        seen.add(s)
        if (re.search(r"\d", s) or s in known_short) and len(s) <= 10:
            species.append(s)

    # Tables and figures — strip markdown bold markers before extracting.
    # Stop at sentence boundary (period + space) or newline, not bare periods
    # so "Table 1 (p.6): description" is captured fully.
    clean = _clean_md_bold(text)
    key_tables = re.findall(r"(Table\s+\d+.*?)(?=\.\s|\n|$)", clean, re.IGNORECASE)
    key_figures = re.findall(r"(Fig(?:ure)?\.?\s+\d+.*?)(?=\.\s|\n|$)", clean, re.IGNORECASE)

    summary = PaperSummary(
        reactor_types=reactor_types,
        species_studied=species[:15],
        temperature_range=temperature_range,
        pressure_range=pressure_range,
        key_tables=[t.strip()[:100] for t in key_tables[:10]],
        key_figures=[f.strip()[:100] for f in key_figures[:10]],
    )
    logger.info(
        "Markdown fallback extracted: reactors=%s T=%s P=%s species=%d",
        summary.reactor_types,
        summary.temperature_range,
        summary.pressure_range,
        len(summary.species_studied),
    )
    return summary


# ── Public API ───────────────────────────────────────────────────────────────


async def read_paper(
    paper: PaperDocument,
    config: LLMConfig,
    model_species: list[str] | None = None,
) -> PaperSummary:
    """Run the PaperReaderAgent and return a structured summary."""
    model = make_model(config, agent_name="paper_reader")
    deps = PaperDeps(paper=paper, model_species=model_species or [])

    prompt = (
        "Use tools to read the paper and produce a summary. "
        "Start with get_abstract(), then list_sections(), "
        "then read the experimental section. "
        "When done, respond with ONLY a JSON object — "
        "first character must be { and last must be }. "
        "No markdown, no headers, no explanation."
    )
    assert len(prompt.split()) < 100, "Initial prompt must not contain paper text"
    result = await paper_reader_agent.run(
        prompt,
        deps=deps,
        model=model,
    )

    raw = strip_thinking(result.output)
    logger.debug("PaperReaderAgent raw output (%d chars): %s", len(raw), raw[:500])
    json_str = extract_json_object(raw)
    summary: PaperSummary | None = None
    try:
        parsed = PaperSummary.model_validate_json(json_str)
        # Check the parse produced something useful — not just defaults from "{}".
        if parsed.reactor_types or parsed.temperature_range or parsed.species_studied:
            summary = parsed
    except Exception as e:
        logger.warning("PaperSummary JSON parse failed: %s", e)

    if summary is None:
        summary = _parse_markdown_fallback(raw)

    logger.info(
        "PaperReaderAgent: %s | T=%s | tables=%d figures=%d",
        summary.reactor_types,
        summary.temperature_range,
        len(summary.key_tables),
        len(summary.key_figures),
    )
    return summary
