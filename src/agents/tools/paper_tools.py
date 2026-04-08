"""Paper-reading tools for PydanticAI agents.

Each tool wraps existing ingestion data structures so agents can
explore a parsed paper incrementally rather than ingesting it whole.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from src.schemas.experimental import PaperDocument, SectionSpan
from src.ingestion.pdf_parser.section_detector import detect_sections


class PaperDeps(BaseModel):
    """Dependency object passed to tools via RunContext."""

    paper: PaperDocument
    model_species: list[str] = Field(default_factory=list)

    # Lazily computed section spans.
    _sections: list[SectionSpan] | None = None

    model_config = {"arbitrary_types_allowed": True}

    @property
    def sections(self) -> list[SectionSpan]:
        if self._sections is None:
            self._sections = detect_sections(self.paper.pages)
        return self._sections


# ── Helpers ──────────────────────────────────────────────────────────────────

_FIG_RE = re.compile(r"(?:fig(?:ure)?\.?\s*)?(\d+)", re.IGNORECASE)
_TABLE_RE = re.compile(r"(?:table\.?\s*)?(\d+)", re.IGNORECASE)


def _normalize_figure_id(raw: str) -> str:
    """Extract the numeric part from 'Figure 1', 'Fig. 1', or '1'."""
    m = _FIG_RE.search(raw)
    return m.group(1) if m else raw.strip()


def _normalize_table_id(raw: str) -> str:
    """Extract the numeric part from 'Table 1' or '1'."""
    m = _TABLE_RE.search(raw)
    return m.group(1) if m else raw.strip()


def _caption_num(caption_id: str) -> str:
    """Extract trailing digits from a caption figure_id like 'Figure 1'."""
    m = re.search(r"(\d+)", caption_id)
    return m.group(1) if m else caption_id


# ── Tool implementations ────────────────────────────────────────────────────
# These are plain functions. They get registered on a PydanticAI Agent via
# agent.tool decorator in the agent module (paper_reader.py).


def get_abstract(deps: PaperDeps) -> str:
    """Return the paper abstract, or the first 500 characters of page 1."""
    if deps.paper.abstract:
        return deps.paper.abstract
    if deps.paper.pages:
        return deps.paper.pages[0].text[:500]
    return "No abstract or text available."


def get_section(deps: PaperDeps, section_name: str) -> str:
    """Find and return the text of a section by name.

    Case-insensitive partial match (e.g. 'experimental' matches
    'Experimental Methods').
    """
    query = section_name.lower()

    # Find matching section span.
    match: SectionSpan | None = None
    for span in deps.sections:
        if query in span.name.lower():
            match = span
            break

    if match is None:
        available = ", ".join(s.name for s in deps.sections)
        return f"Section not found. Available sections: {available}"

    # Collect text from the matching page range.
    parts: list[str] = []
    for page in deps.paper.pages:
        if match.page_start <= page.page_num <= match.page_end:
            parts.append(page.text)
    return "\n".join(parts) if parts else "Section found but no text available."


def search_text(deps: PaperDeps, query: str) -> str:
    """Search all page text for passages containing query terms.

    Returns up to 5 matching passages with page numbers.
    """
    terms = query.lower().split()
    hits: list[str] = []

    for page in deps.paper.pages:
        lines = page.text.split("\n")
        for i, line in enumerate(lines):
            lower = line.lower()
            if all(t in lower for t in terms):
                # Include surrounding context (1 line before/after).
                start = max(0, i - 1)
                end = min(len(lines), i + 2)
                passage = " ".join(lines[start:end]).strip()
                hits.append(f"Page {page.page_num}: {passage}")
                if len(hits) >= 5:
                    break
        if len(hits) >= 5:
            break

    return "\n\n".join(hits) if hits else "No matches found."


def get_figure_caption(deps: PaperDeps, figure_id: str) -> str:
    """Look up a figure caption by ID.

    Accepts 'Figure 1', 'Fig. 1', or '1'.
    """
    num = _normalize_figure_id(figure_id)
    for cap in deps.paper.captions:
        if cap.label_type.lower() in ("figure", "fig", "fig."):
            if _caption_num(cap.figure_id) == num:
                return f"{cap.figure_id} (p.{cap.page_num}): {cap.caption}"
    return "Figure not found."


def get_table(deps: PaperDeps, table_id: str) -> str:
    """Return a parsed table as formatted text.

    Accepts 'Table 1' or '1'.
    """
    num = _normalize_table_id(table_id)
    for tbl in deps.paper.tables:
        if _caption_num(tbl.table_id) == num:
            header = " | ".join(c.header for c in tbl.columns)
            rows_text = []
            for row in tbl.rows:
                cells = [v.raw_text for v in row.cells.values()]
                rows_text.append(" | ".join(cells))
            return f"{tbl.table_id}: {tbl.caption}\n{header}\n" + "\n".join(rows_text)
    return "Table not found."


def list_figures(deps: PaperDeps) -> str:
    """List all figure IDs and first 100 chars of each caption."""
    figs = [
        c for c in deps.paper.captions
        if c.label_type.lower() in ("figure", "fig", "fig.")
    ]
    if not figs:
        return "No figures found."
    return "\n".join(
        f"{c.figure_id} (p.{c.page_num}): {c.caption[:100]}" for c in figs
    )


def list_tables(deps: PaperDeps) -> str:
    """List all table IDs and captions."""
    if not deps.paper.tables:
        # Fall back to table captions if tables weren't parsed.
        tbl_caps = [
            c for c in deps.paper.captions
            if c.label_type.lower() == "table"
        ]
        if not tbl_caps:
            return "No tables found."
        return "\n".join(
            f"{c.figure_id} (p.{c.page_num}): {c.caption[:100]}" for c in tbl_caps
        )
    return "\n".join(
        f"{t.table_id} (p.{t.page_num}): {t.caption}" for t in deps.paper.tables
    )


def list_sections(deps: PaperDeps) -> str:
    """Return all detected section names."""
    if not deps.sections:
        return "No sections detected."
    return "\n".join(
        f"{s.name} (pp.{s.page_start}-{s.page_end})" for s in deps.sections
    )
