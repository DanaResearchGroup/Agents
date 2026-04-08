"""Extract figure and table captions from parsed page text."""

from __future__ import annotations

import re

from models import PageText, FigureCaption


# Matches actual captions: "Fig. 1. Comparison...", "Figure 3 Species profiles..."
# Strategy: start of line + reject if followed by a reference verb.
# This avoids requiring specific punctuation styles (which vary across journals).
_FIG_PATTERN = re.compile(
    r'(?:^|\n)\s*'  # start of line
    r'((?:Figure|Fig\.?|FIGURE)\s+S?\d+[a-zA-Z]?)'
    r'(?:\s*[:.)\-\u2013\u2014]\s*|\s+)'  # punctuation OR space
    r'(.*?)(?=\n\s*\n|\n\s*(?:Figure|Fig\.?|Table|FIGURE|TABLE)\s+S?\d|\Z)',
    re.DOTALL | re.IGNORECASE,
)

# Same for tables
_TABLE_PATTERN = re.compile(
    r'(?:^|\n)\s*'  # start of line
    r'((?:Table|TABLE)\s+S?\d+[a-zA-Z]?)'
    r'(?:\s*[:.)\-\u2013\u2014]\s*|\s+)'
    r'(.*?)(?=\n\s*\n|\n\s*(?:Figure|Fig\.?|Table|FIGURE|TABLE)\s+S?\d|\Z)',
    re.DOTALL | re.IGNORECASE,
)

# Verbs that indicate a body-text reference, not a caption
# "Figure 6 shows...", "Figure 6 presents...", "Figure 6 compares..."
_REFERENCE_VERBS = re.compile(
    r'^(?:shows?|presents?|illustrates?|depicts?|displays?|demonstrates?|'
    r'plots?|compares?|provides?|contains?|summarizes?|reports?|gives?|'
    r'is\s|was\s|are\s|were\s|can\s|also\s|further\s)',
    re.IGNORECASE,
)

# Normalize figure IDs: "Fig. 1" -> "Figure 1", "FIGURE 1" -> "Figure 1"
_FIG_NORMALIZE = re.compile(r'^(?:Fig\.?|FIGURE)\s+', re.IGNORECASE)


def extract_captions(pages: list[PageText]) -> list[FigureCaption]:
    """Extract figure and table captions from page text."""
    captions: list[FigureCaption] = []
    seen: dict[str, FigureCaption] = {}  # dedup by (normalized_id, label_type)

    for page in pages:
        # Extract figure captions
        for match in _FIG_PATTERN.finditer(page.text):
            fig_id = _normalize_figure_id(match.group(1))
            caption_text = _clean_caption(match.group(2))
            if not caption_text:
                continue
            # Reject body-text references: "Figure 6 shows..."
            if _REFERENCE_VERBS.match(caption_text):
                continue
            key = f"figure:{fig_id}"
            existing = seen.get(key)
            if existing is None or len(caption_text) > len(existing.caption):
                cap = FigureCaption(
                    figure_id=fig_id,
                    label_type="figure",
                    page_num=page.page_num,
                    caption=caption_text,
                )
                seen[key] = cap

        # Extract table captions
        for match in _TABLE_PATTERN.finditer(page.text):
            table_id = match.group(1).strip()
            # Normalize: "TABLE 1" -> "Table 1"
            table_id = "Table " + table_id.split()[-1]
            caption_text = _clean_caption(match.group(2))
            if not caption_text:
                continue
            if _REFERENCE_VERBS.match(caption_text):
                continue
            key = f"table:{table_id}"
            existing = seen.get(key)
            if existing is None or len(caption_text) > len(existing.caption):
                cap = FigureCaption(
                    figure_id=table_id,
                    label_type="table",
                    page_num=page.page_num,
                    caption=caption_text,
                )
                seen[key] = cap

    captions = sorted(seen.values(), key=lambda c: (c.page_num, c.figure_id))
    return captions


def _normalize_figure_id(raw_id: str) -> str:
    """Normalize 'Fig. 1', 'FIGURE 1' etc. to 'Figure 1'."""
    normalized = _FIG_NORMALIZE.sub("Figure ", raw_id.strip())
    return normalized


def _clean_caption(text: str) -> str:
    """Collapse whitespace and strip trailing noise from caption text."""
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove trailing partial references like "[1" or "(see"
    text = re.sub(r'\s*[\[\(](?:see|cf\.?|from)?\s*$', '', text, flags=re.IGNORECASE)
    return text
