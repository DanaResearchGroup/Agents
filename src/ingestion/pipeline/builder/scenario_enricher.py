"""Enrich raw scenarios with context from methods, body text, and mixture definitions.

Core policy: FILL-ONLY by default.
- Caption-derived values are never overwritten by weaker sources
- Each field has a source level; only upgrade if new source is more specific
- Supporting context is added as provenance, not as replacement

Source precedence (highest to lowest):
  1. panel_text       — OCR from inside the figure
  2. figure_caption   — the caption's own text
  3. figure_paragraph — paragraph explicitly about this figure
  4. family_summary   — experiment-family summary in results
  5. methods_nominal  — methods section prepared/nominal values
  6. default          — family-level assumption (e.g., ~1 atm for shock tubes)
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.schemas.experimental import ExperimentalScenario, PaperDocument, ExperimentCandidate


def _collapse_spaces(text: str) -> str:
    """Collapse spaces within chemical formulas: 'NH 3' → 'NH3'."""
    text = re.sub(r'([A-Z]) (\d)', r'\1\2', text)
    text = re.sub(r'(\d) ([A-Z])', r'\1\2', text)
    return text


# Source levels — higher number = higher priority
SOURCE_PRIORITY = {
    "panel_text": 6,
    "table_body": 5,
    "figure_caption": 5,
    "figure_paragraph": 4,
    "family_summary": 3,
    "methods_nominal": 2,
    "default": 1,
}


@dataclass
class FieldSource:
    """Tracks where a field value came from."""
    value: str
    source_level: str
    source_detail: str  # e.g., "Figure 6 caption", "methods page 2"

    @property
    def priority(self) -> int:
        return SOURCE_PRIORITY.get(self.source_level, 0)


# ── Patterns ─────────────────────────────────────────────────────────────

_MIXTURE_DEF = re.compile(
    r'(?:mixture\s+(\d+)|mix\.?\s+(\d+))\s*[:\-]?\s*'
    r'((?:[\u223c~∼]?\s*\d+(?:\.\d+)?%?\s*[A-Z][A-Za-z0-9]*'
    r'(?:\s*/\s*\d+(?:\.\d+)?%?\s*[A-Z][A-Za-z0-9]*)*'
    r'(?:\s+in\s+[A-Z][a-z]?)?))',
    re.IGNORECASE,
)

_MIXTURE_REF = re.compile(r'mixture\s+(\d+)', re.IGNORECASE)

_DEFAULT_PRESSURE = re.compile(
    r'(?:(?:near|at)\s+)?atmospher(?:ic|e)\s+pressure|'
    r'ambient\s+pressure|'
    r'(?:at|near|approximately|around)\s+(\d+(?:\.\d+)?)\s*(?:atm|MPa)',
    re.IGNORECASE,
)

_TEMP_NEARBY = re.compile(
    r'(\d+(?:\.\d+)?)\s*(?:[-\u2013\u2014]\s*(\d+(?:\.\d+)?)\s*)?K\b',
)

_PRESS_NEARBY = re.compile(
    r'(\d+(?:\.\d+)?)\s*(?:[-\u2013\u2014]\s*(\d+(?:\.\d+)?)\s*)?atm\b',
)


def enrich_scenarios(
    scenarios: list[ExperimentalScenario],
    document: PaperDocument,
    candidate: ExperimentCandidate,
) -> list[ExperimentalScenario]:
    """Enrich scenarios using fill-only precedence policy."""
    mixture_defs = _extract_mixture_definitions(document)
    default_pressure = _find_default_pressure(document, candidate)

    for scenario in scenarios:
        # Step 1: validate — reject malformed values
        _validate_fields(scenario)

        # Step 2: track what source level existing fields came from
        if scenario.source == "table":
            base_level = "table_body"
        else:
            base_level = "figure_caption"
        comp_source = base_level if scenario.composition_text else None
        temp_source = base_level if scenario.temperature_text else None
        press_source = base_level if scenario.pressure_text else None

        # Step 3: fill MISSING composition from mixture references
        # Only if composition is None or was rejected by validation
        if not scenario.composition_text:
            resolved = _resolve_mixture_ref(scenario, mixture_defs, document)
            if resolved:
                scenario.composition_text = resolved.value
                comp_source = resolved.source_level
                scenario.notes.append(
                    f"composition filled from {resolved.source_detail}"
                )

        # Step 4: fill MISSING pressure from methods default
        if not scenario.pressure_text and default_pressure:
            scenario.pressure_text = default_pressure
            press_source = "default"
            scenario.notes.append(
                f"pressure filled from methods default: {default_pressure}"
            )

        # Step 5: fill MISSING temperature from nearby body text
        if not scenario.temperature_text:
            nearby_t = _find_nearby_temp(scenario, document)
            if nearby_t:
                scenario.temperature_text = nearby_t.value
                temp_source = nearby_t.source_level
                scenario.notes.append(
                    f"temperature filled from {nearby_t.source_detail}"
                )

        # Step 6: fill MISSING pressure from nearby body text
        if not scenario.pressure_text:
            nearby_p = _find_nearby_pressure(scenario, document)
            if nearby_p:
                scenario.pressure_text = nearby_p.value
                press_source = nearby_p.source_level
                scenario.notes.append(
                    f"pressure filled from {nearby_p.source_detail}"
                )

        # Step 7: re-evaluate simulatable
        scenario.simulatable = (
            scenario.temperature_text is not None
            and scenario.pressure_text is not None
            and scenario.composition_text is not None
        )

    return scenarios


# ── Mixture definition extraction ────────────────────────────────────────

def _extract_mixture_definitions(document: PaperDocument) -> dict[str, str]:
    """Find mixture definitions from methods pages."""
    defs: dict[str, str] = {}

    for page in document.pages[:5]:
        text = _collapse_spaces(page.text)
        for match in _MIXTURE_DEF.finditer(text):
            mix_num = match.group(1) or match.group(2)
            comp = match.group(3).strip()
            if mix_num and comp:
                defs[mix_num] = comp

    # Fallback: "prepared as X / Y and X / Y / Z"
    if not defs:
        _COMP_FRAG = (
            r'[\u223c~∼]?\s*\d+(?:\.\d+)?%?\s*[A-Z][A-Za-z0-9]*'
            r'(?:\s*/\s*\d+(?:\.\d+)?%?\s*[A-Z][A-Za-z0-9]*)*'
            r'(?:\s+in\s+[A-Z][a-z]?)?'
        )
        for page in document.pages[:5]:
            text = _collapse_spaces(page.text)
            block_match = re.search(
                r'prepared\s+as\s*(.+?)(?:\.\s+[A-Z]|\.\s*$)',
                text, re.IGNORECASE | re.DOTALL,
            )
            if not block_match:
                continue
            parts = re.split(r'\s+and\s+', block_match.group(1))
            for i, part in enumerate(parts, 1):
                comp_match = re.search(_COMP_FRAG, part)
                if comp_match:
                    defs[str(i)] = comp_match.group(0).strip()

    return defs


def _find_default_pressure(
    document: PaperDocument,
    candidate: ExperimentCandidate,
) -> str | None:
    """Find default experimental pressure from methods."""
    methods_ev = [e for e in candidate.evidence if e.section == "methods"]
    for ev in methods_ev:
        lower = ev.source_text.lower()
        if "atmospheric" in lower or "atmosphere pressure" in lower or "ambient pressure" in lower:
            return "1 atm"
        m = _DEFAULT_PRESSURE.search(ev.source_text)
        if m:
            val = m.group(1)
            if val:
                # "0.1 MPa" → 1 atm
                if "mpa" in (m.group(0) or "").lower():
                    return f"{float(val) * 9.869:.1f} atm"
                return f"{val} atm"
            return "1 atm"

    for page in document.pages[:5]:
        lower = page.text.lower()
        if any(kw in lower for kw in (
            "atmospheric pressure", "atmosphere pressure", "ambient pressure",
        )):
            return "1 atm"

    return None


# ── Field-level enrichment (fill-only) ───────────────────────────────────

def _validate_fields(scenario: ExperimentalScenario) -> None:
    """Reject obviously malformed field values."""
    if scenario.composition_text:
        comp = scenario.composition_text
        if re.match(r'^\s*\d+(?:\.\d+)?\s*(?:K|atm|Torr|bar|Pa)\s*$', comp, re.I):
            scenario.notes.append(f"rejected malformed composition: {comp}")
            scenario.composition_text = None
        elif re.match(r'^\s*\d+(?:\.\d+)?\s*$', comp):
            scenario.notes.append(f"rejected bare number as composition: {comp}")
            scenario.composition_text = None


def _resolve_mixture_ref(
    scenario: ExperimentalScenario,
    mixture_defs: dict[str, str],
    document: PaperDocument,
) -> FieldSource | None:
    """Try to resolve a 'mixture N' reference. Returns None if not applicable.

    ONLY called when composition is missing — never overwrites existing values.
    """
    if not mixture_defs:
        return None

    # Get caption text to look for "mixture N" reference
    caption_text = ""
    for cap in document.captions:
        if cap.figure_id == scenario.source_label:
            caption_text = _collapse_spaces(cap.caption)
            break

    # Check caption for "mixture N"
    for text_field in [caption_text, scenario.source_label]:
        m = _MIXTURE_REF.search(text_field)
        if m and m.group(1) in mixture_defs:
            resolved = mixture_defs[m.group(1)]
            return FieldSource(
                value=resolved,
                source_level="methods_nominal",
                source_detail=f"mixture {m.group(1)} definition: {resolved}",
            )

    return None


def _find_nearby_temp(
    scenario: ExperimentalScenario,
    document: PaperDocument,
) -> FieldSource | None:
    """Find temperature from nearby body text."""
    page_idx = scenario.source_page - 1
    if page_idx < 0 or page_idx >= len(document.pages):
        return None

    page_text = _collapse_spaces(document.pages[page_idx].text)

    # Look near the figure reference
    ref_pattern = re.compile(
        rf'{re.escape(scenario.source_label)}[^.]*\.',
        re.IGNORECASE,
    )
    ref_match = ref_pattern.search(page_text)
    search_text = ref_match.group(0) if ref_match else page_text[:2000]

    m = _TEMP_NEARBY.search(search_text)
    if m:
        return FieldSource(
            value=m.group(0),
            source_level="figure_paragraph",
            source_detail=f"nearby text on p{scenario.source_page}: {m.group(0)}",
        )
    return None


def _find_nearby_pressure(
    scenario: ExperimentalScenario,
    document: PaperDocument,
) -> FieldSource | None:
    """Find pressure from nearby body text."""
    page_idx = scenario.source_page - 1
    if page_idx < 0 or page_idx >= len(document.pages):
        return None

    page_text = _collapse_spaces(document.pages[page_idx].text)

    ref_pattern = re.compile(
        rf'{re.escape(scenario.source_label)}[^.]*\.',
        re.IGNORECASE,
    )
    ref_match = ref_pattern.search(page_text)
    search_text = ref_match.group(0) if ref_match else page_text[:2000]

    m = _PRESS_NEARBY.search(search_text)
    if m:
        return FieldSource(
            value=m.group(0),
            source_level="figure_paragraph",
            source_detail=f"nearby text on p{scenario.source_page}: {m.group(0)}",
        )
    return None
