"""PaperReaderAgent — reads a paper and produces a structured PaperSummary.

This is the first pass over a paper. It uses tools to explore the document
and outputs a summary that downstream agents (condition extraction, reaction
mining) consume as context.
"""

from __future__ import annotations

import logging

from pydantic_ai import Agent, RunContext

from src.agents.llm_client import LLMConfig
from src.agents.provider import make_model
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

_SYSTEM_PROMPT = """\
You are a combustion chemistry expert reading a kinetics paper.
Your job is to summarise the experimental setup so other agents
can extract simulation conditions.

Use the available tools to read the paper systematically:
1. Start with the abstract
2. Find the experimental/methods section
3. List all tables and figures
4. Identify what reactor was used, what species, what conditions

Be specific about temperature and pressure ranges.
List all tables that contain experimental conditions.
Note the species names exactly as written in the paper."""

paper_reader_agent: Agent[PaperDeps, PaperSummary] = Agent(
    "test",  # placeholder model, overridden at runtime
    deps_type=PaperDeps,
    output_type=PaperSummary,
    system_prompt=_SYSTEM_PROMPT,
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


# ── Public API ───────────────────────────────────────────────────────────────


async def read_paper(
    paper: PaperDocument,
    config: LLMConfig,
    model_species: list[str] | None = None,
) -> PaperSummary:
    """Run the PaperReaderAgent and return a structured summary."""
    model = make_model(config, agent_name="paper_reader")
    deps = PaperDeps(paper=paper, model_species=model_species or [])

    result = await paper_reader_agent.run(
        "Read this paper and produce a structured summary.",
        deps=deps,
        model=model,
    )

    summary = result.data
    logger.info(
        "PaperReaderAgent: %s | T=%s | tables=%d figures=%d",
        summary.reactor_types,
        summary.temperature_range,
        len(summary.key_tables),
        len(summary.key_figures),
    )
    return summary
