"""Detect section boundaries in parsed paper text."""

from __future__ import annotations

import re

from models import PageText, SectionSpan

# Section name -> weight for evidence scoring
SECTION_WEIGHTS: dict[str, float] = {
    "methods": 1.0,
    "results": 0.85,
    "caption": 0.90,
    "abstract": 0.6,
    "introduction": 0.3,
    "conclusion": 0.4,
    "references": 0.0,
    "unknown": 0.5,
}

# Patterns map raw header text to a canonical section name.
# Order matters: first match wins, so more specific patterns come first.
_SECTION_HEADERS: list[tuple[str, re.Pattern]] = [
    ("abstract", re.compile(
        r'^\s*A\s*B\s*S\s*T\s*R\s*A\s*C\s*T\s*$|^\s*Abstract\s*$',
        re.IGNORECASE,
    )),
    ("methods", re.compile(
        r'^\s*(?:\d+\.?\s+)?'
        r'(?:Experimental\s+(?:methods?|setup|section|procedures?|details?)|'
        r'Experimental|'
        r'Methods?\s+(?:and\s+materials?)?|'
        r'Computational\s+(?:methods?|details?)|'
        r'Reactor\s+setup|'
        r'Materials?\s+and\s+methods?)\s*$',
        re.IGNORECASE,
    )),
    ("results", re.compile(
        r'^\s*(?:\d+\.?\s+)?'
        r'(?:Results?\s+(?:and\s+)?(?:discussion)?|'
        r'Discussion|'
        r'Results?|'
        r'Modeling\s+results?|'
        r'Numerical\s+results?|'
        r'Model\s+(?:validation|comparison|assessment))\s*$',
        re.IGNORECASE,
    )),
    ("introduction", re.compile(
        r'^\s*(?:\d+\.?\s+)?Introduction\s*$',
        re.IGNORECASE,
    )),
    ("conclusion", re.compile(
        r'^\s*(?:\d+\.?\s+)?'
        r'(?:Conclusions?|Summary(?:\s+and\s+conclusions?)?)\s*$',
        re.IGNORECASE,
    )),
    ("references", re.compile(
        r'^\s*(?:\d+\.?\s+)?'
        r'(?:References|Bibliography)\s*$',
        re.IGNORECASE,
    )),
]

# Subsection headers inside methods — REQUIRE a section number (e.g. "2.1. Facilities")
# to avoid matching bare keywords like "Shock tube" in the keywords list
_METHODS_SUBSECTIONS = re.compile(
    r'^\s*\d+\.\d+\.?\s+'
    r'(?:Facilities|Laser\s+absorption|Mixture\s+preparation|'
    r'Shock[\-\s]?tube|Reactor|Kinetic\s+model|Chemical\s+kinetic)',
    re.IGNORECASE,
)


def detect_sections(pages: list[PageText]) -> list[SectionSpan]:
    """Detect section boundaries and assign trust weights.

    Returns a list of SectionSpan objects covering the entire document.
    Pages not matched to any section get 'unknown' with weight 0.5.
    """
    # Collect (section_name, page_num, line_index) for each detected header
    boundaries: list[tuple[str, int, int]] = []
    past_references = False

    for page in pages:
        for line_idx, line in enumerate(page.text.splitlines()):
            section_name = _match_header(line, past_references=past_references)
            if section_name:
                boundaries.append((section_name, page.page_num, line_idx))
                if section_name == "references":
                    past_references = True

    # If no sections detected at all, use heuristic fallback
    if not boundaries:
        return _fallback_sections(pages)

    # Build spans from boundaries
    spans: list[SectionSpan] = []
    total_pages = len(pages)

    for i, (name, page_start, line_start) in enumerate(boundaries):
        if i + 1 < len(boundaries):
            page_end = boundaries[i + 1][1]
            # If next section starts on same page, this section ends on that page
            # If next section starts on a later page, this section ends on page before
            if boundaries[i + 1][2] > 0 or page_start == boundaries[i + 1][1]:
                page_end = boundaries[i + 1][1]
            else:
                page_end = boundaries[i + 1][1] - 1
        else:
            page_end = total_pages

        weight = SECTION_WEIGHTS.get(name, 0.5)
        spans.append(SectionSpan(
            name=name,
            page_start=page_start,
            page_end=page_end,
            line_start=line_start,
            weight=weight,
        ))

    return _merge_consecutive(spans)


def _merge_consecutive(spans: list[SectionSpan]) -> list[SectionSpan]:
    """Merge consecutive spans with the same section name."""
    if not spans:
        return spans
    merged: list[SectionSpan] = [spans[0]]
    for span in spans[1:]:
        prev = merged[-1]
        if span.name == prev.name and span.page_start <= prev.page_end + 1:
            prev.page_end = max(prev.page_end, span.page_end)
        else:
            merged.append(span)
    return merged


def get_section_for_page(sections: list[SectionSpan], page_num: int) -> SectionSpan | None:
    """Look up which section a page belongs to. Returns the last matching span."""
    result = None
    for span in sections:
        if span.page_start <= page_num <= span.page_end:
            result = span
    return result


def _match_header(line: str, *, past_references: bool = False) -> str | None:
    """Check if a line is a section header. Returns canonical section name or None."""
    stripped = line.strip()
    if not stripped or len(stripped) > 80:
        return None
    # Skip lines that end with a period (likely a sentence, not a header)
    if stripped.endswith('.') and len(stripped) > 30:
        return None

    for section_name, pattern in _SECTION_HEADERS:
        if pattern.match(stripped):
            return section_name

    # After references, don't match subsection headers (they're likely ref text)
    if past_references:
        return None

    # Check methods subsection headers — keep them labeled as methods
    if _METHODS_SUBSECTIONS.match(stripped):
        return "methods"

    return None


def _fallback_sections(pages: list[PageText]) -> list[SectionSpan]:
    """Heuristic fallback when no section headers are detected."""
    total = len(pages)
    if total == 0:
        return []

    spans = []
    if total >= 1:
        spans.append(SectionSpan("abstract", 1, 1, 0, SECTION_WEIGHTS["abstract"]))
    if total >= 3:
        spans.append(SectionSpan("introduction", 2, min(3, total), 0, SECTION_WEIGHTS["introduction"]))
    if total >= 4:
        spans.append(SectionSpan("methods", 3, total, 0, SECTION_WEIGHTS["methods"]))

    return spans
