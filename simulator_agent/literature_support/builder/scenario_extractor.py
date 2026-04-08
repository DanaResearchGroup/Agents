"""Extract experimental scenarios from paper evidence.

A scenario is one coherent set of conditions that can be simulated:
one mixture, one T (or range), one P, one observable.

Sources (in order of reliability):
1. Figure captions with explicit conditions
2. Methods-described mixture families
3. Table condition clusters
"""

from __future__ import annotations

import re

from models import (
    ExperimentCandidate, ExperimentalScenario, FigureCaption,
    PaperDocument, EvidenceSnippet, ParsedTable, TableRow,
)


# Collapse fitz spacing artifacts in chemical formulas
def _collapse_spaces(text: str) -> str:
    text = re.sub(r'([A-Za-z]) (\d)', r'\1\2', text)
    text = re.sub(r'(\d) ([A-Z])', r'\1\2', text)
    return text


# ── Pattern building blocks ──────────────────────────────────────────────

_NUM = r'(\d+(?:\.\d+)?)'
_DASH = r'[-\u2013\u2014\u2212]'

# Temperature: "2217 K", "2096-2487 K", "2096 to 2487 K"
_TEMP_RE = re.compile(
    rf'{_NUM}\s*(?:{_DASH}|to)\s*{_NUM}\s*K|{_NUM}\s*K',
)

# Pressure: "1.22 atm", "1 atm", "0.88-1.13 atm"
_PRESS_RE = re.compile(
    rf'{_NUM}\s*(?:{_DASH}|to)\s*{_NUM}\s*atm|{_NUM}\s*atm',
)

# Composition: "0.45% NH3 in Ar", "0.439% NH3 /Ar", "0.5% NH3/2% H2 in Ar"
# Requires % sign on first species to distinguish from bare numbers.
# Allows slash-separated species with or without % (e.g., "/Ar", "/N2").
_COMP_RE = re.compile(
    r'(?:(?:\u223c|~|∼)?\s*'                          # optional tilde
    r'\d+(?:\.\d+)?%\s*'                              # number + REQUIRED %
    r'[A-Z][A-Za-z0-9]*'                              # first species
    r'(?:\s*/\s*(?:\d+(?:\.\d+)?%?\s*)?[A-Z][A-Za-z0-9]*)*'  # /species pairs (% optional)
    r'(?:\s+in\s+[A-Z][a-z]?)?)',                     # optional "in Ar"
)

from literature_support.families import get_family, get_all_families, detect_family

# Caption keywords that indicate a literature review or reference table,
# not the current paper's experiments.
_REVIEW_TABLE_KW = re.compile(
    r'(?:recent|previous|prior|published|existing)\s+'
    r'(?:fundamental\s+)?(?:studies|investigations?|works?|data|results)|'
    r'literature\s+(?:review|survey|summary)|'
    r'comparison\s+(?:of|with)\s+(?:previous|existing|published)|'
    r'summary\s+of\s+(?:previous|existing|published)',
    re.IGNORECASE,
)


def _is_review_table(caption_text: str) -> bool:
    """Return True if the caption suggests a literature review / reference table."""
    return bool(_REVIEW_TABLE_KW.search(caption_text[:200]))


def extract_scenarios(
    document: PaperDocument,
    candidate: ExperimentCandidate,
) -> list[ExperimentalScenario]:
    """Extract experimental scenarios from figure captions, methods text,
    and structured condition tables.

    Focuses on the given candidate's experiment family.
    """
    scenarios: list[ExperimentalScenario] = []
    idx = 0

    # Source 1: figure/table captions with explicit conditions
    # Skip table captions that are literature reviews or already parsed
    # as condition tables by Source 3.
    parsed_table_ids = {t.table_id for t in document.tables}
    for caption in document.captions:
        if caption.label_type == "table":
            # Source 3 handles condition tables
            if caption.figure_id in parsed_table_ids:
                continue
            # Literature review / reference tables produce garbled body text
            if _is_review_table(caption.caption):
                continue
        # Limit to first ~500 chars of caption to avoid body text bleed
        raw = caption.caption[:500]
        text = _collapse_spaces(raw)
        caption_scenarios = _scenarios_from_caption(caption, text, candidate.experiment_family)
        for scenario in caption_scenarios:
            idx += 1
            scenario.scenario_id = f"scenario_{idx}"
            scenarios.append(scenario)

    # Source 2: methods mixtures (define scenario families)
    methods_scenarios = _scenarios_from_methods(document, candidate)
    for s in methods_scenarios:
        # Don't duplicate if a caption scenario already covers this mixture
        if not _is_duplicate(s, scenarios):
            idx += 1
            s.scenario_id = f"scenario_{idx}"
            scenarios.append(s)

    # Source 3: structured condition tables (one scenario per row)
    table_scenarios = _scenarios_from_table_rows(document, candidate)
    for s in table_scenarios:
        idx += 1
        s.scenario_id = f"scenario_{idx}"
        scenarios.append(s)

    # Deduplicate by (source_label, composition_text)
    seen: set[tuple] = set()
    deduped = []
    for s in scenarios:
        key = (s.source_label, s.composition_text)
        if key not in seen:
            seen.add(key)
            deduped.append(s)
    scenarios = deduped

    # Mark simulatable
    for s in scenarios:
        s.simulatable = (
            s.temperature_text is not None
            and s.pressure_text is not None
            and s.composition_text is not None
        )

    return scenarios


def _scenarios_from_caption(
    caption: FigureCaption,
    text: str,
    primary_family: str,
) -> list[ExperimentalScenario]:
    """Extract one or more scenarios from a figure/table caption.

    Emits multiple scenarios when the caption describes multiple compositions
    (e.g., "mixture of ~0.5% NH3 in Ar (red) and ~0.42% NH3/2% H2 in Ar (blue)").
    """
    temp = _extract_first(text, _TEMP_RE)
    pressure = _extract_pressure_range(text)
    compositions = _extract_all(text, _COMP_RE)

    # Need at least temperature or a composition to form a scenario
    if not temp and not compositions:
        return []

    # Detect observable using family registry
    family_rules = get_family(primary_family)
    observable = family_rules.detect_observable(text)
    # If primary family didn't match, try all families
    if observable is None:
        for rules in get_all_families().values():
            observable = rules.detect_observable(text)
            if observable:
                break

    # Family: default to parent candidate family.
    # Only override if the caption's FIRST sentence explicitly names a different family
    # (not from comparison/validation mentions later in the caption).
    family = primary_family
    first_sentence = text.split('.')[0] if '.' in text else text[:150]
    local_family = detect_family(first_sentence)
    if local_family and local_family != primary_family:
        family = local_family

    # Determine scenario type
    is_range = bool(re.search(rf'\d\s*(?:{_DASH}|to)\s*\d', temp or ""))
    scenario_type = "range" if is_range else "single_point"

    base_notes = []
    if caption.label_type == "table":
        base_notes.append("from table caption")
    if not pressure:
        base_notes.append("pressure not specified in caption")

    # If multiple compositions, emit one scenario per composition
    if not compositions:
        compositions = [None]

    scenarios = []
    for i, comp in enumerate(compositions):
        notes = list(base_notes)
        if len(compositions) > 1:
            notes.append(f"composition {i+1} of {len(compositions)} from caption")

        scenarios.append(ExperimentalScenario(
            scenario_id="",  # filled by caller
            experiment_family=family,
            scenario_type=scenario_type,
            source="figure_caption",
            source_label=caption.figure_id,
            source_page=caption.page_num,
            temperature_text=temp,
            pressure_text=pressure,
            composition_text=comp,
            residence_time_text=None,
            equivalence_ratio_text=None,
            observable=observable,
            simulatable=False,  # set later
            notes=notes,
        ))

    return scenarios


def _scenarios_from_methods(
    document: PaperDocument,
    candidate: ExperimentCandidate,
) -> list[ExperimentalScenario]:
    """Extract mixture-based scenarios from methods section evidence."""
    # Find methods-section evidence that mentions mixtures
    methods_ev = [
        e for e in candidate.evidence
        if e.section == "methods"
    ]
    if not methods_ev:
        return []

    # Scan methods pages for mixture definitions
    # Pattern: "mixture 1: ..." or "initially prepared as X% Species / Y% Species"
    scenarios = []
    seen_comps: set[str] = set()

    for ev in methods_ev:
        ctx = _collapse_spaces(ev.source_text)

        # Look for mixture definitions
        mixture_match = re.search(
            r'(?:mixture\s+\d|prepared\s+as|mixtures?\s+(?:of|containing))\s*'
            r'((?:\u223c|~|∼)?\s*\d+(?:\.\d+)?%?\s*[A-Z][A-Za-z0-9]*'
            r'(?:\s*/\s*\d+(?:\.\d+)?%?\s*[A-Z][A-Za-z0-9]*)*'
            r'(?:\s+in\s+[A-Z][a-z]?)?)',
            ctx, re.IGNORECASE,
        )
        if not mixture_match:
            continue

        comp_text = mixture_match.group(1).strip()
        if comp_text in seen_comps:
            continue
        seen_comps.add(comp_text)

        # Get temperature range from same evidence context
        temp = _extract_first(ctx, _TEMP_RE)
        pressure = _extract_first(ctx, _PRESS_RE)

        scenarios.append(ExperimentalScenario(
            scenario_id="",
            experiment_family=candidate.experiment_family,
            scenario_type="range" if temp and re.search(rf'\d\s*(?:{_DASH}|to)\s*\d', temp) else "single_point",
            source="methods_mixture",
            source_label="methods",
            source_page=ev.page_num,
            temperature_text=temp,
            pressure_text=pressure,
            composition_text=comp_text,
            residence_time_text=None,
            equivalence_ratio_text=None,
            observable=None,
            simulatable=False,
            notes=["from methods mixture definition"],
        ))

    return scenarios


def _extract_first(text: str, pattern: re.Pattern) -> str | None:
    """Extract the first match of a pattern from text."""
    m = pattern.search(text)
    return m.group(0).strip() if m else None


def _extract_all(text: str, pattern: re.Pattern) -> list[str]:
    """Extract all non-overlapping matches of a pattern from text."""
    return [m.group(0).strip() for m in pattern.finditer(text)]


def _extract_pressure_range(text: str) -> str | None:
    """Extract pressure, including 'between X and Y atm' ranges."""
    # Try range pattern first: "between 0.88 and 1.26 atm", "0.88-1.26 atm"
    range_match = re.search(
        rf'(?:between\s+)?{_NUM}\s*(?:{_DASH}|and|to)\s*{_NUM}\s*atm',
        text,
    )
    if range_match:
        return range_match.group(0).strip()
    # Fall back to single value
    return _extract_first(text, _PRESS_RE)


def _is_duplicate(
    new: ExperimentalScenario,
    existing: list[ExperimentalScenario],
) -> bool:
    """Check if a scenario's composition already exists in the list."""
    if not new.composition_text:
        return False
    for s in existing:
        if s.composition_text and s.composition_text == new.composition_text:
            return True
    return False


# ── Source 3: structured condition tables ────────────────────────────────

def _scenarios_from_table_rows(
    document: PaperDocument,
    candidate: ExperimentCandidate,
) -> list[ExperimentalScenario]:
    """Emit one scenario per row of each parsed condition table.

    Preserves multi-valued cells as structured table_conditions;
    text fields get a human-readable summary.
    """
    scenarios: list[ExperimentalScenario] = []
    family_rules = get_family(candidate.experiment_family)

    for table in document.tables:
        # Detect observable from table caption
        observable = family_rules.detect_observable(table.caption)
        if not observable and candidate.confirmed_observables:
            observable = candidate.confirmed_observables[0]

        for row in table.rows:
            scenario = _row_to_scenario(
                row, table, candidate.experiment_family, observable,
            )
            scenarios.append(scenario)

    return scenarios


def _row_to_scenario(
    row: TableRow,
    table: ParsedTable,
    family: str,
    observable: str | None,
) -> ExperimentalScenario:
    """Convert a single table row into an ExperimentalScenario."""
    cells = row.cells

    # Temperature
    temp_cell = cells.get('temperature')
    temp_text = _cell_to_text(temp_cell, suffix='K') if temp_cell else None

    # Pressure (may not be in table — enricher fills from methods)
    press_cell = cells.get('pressure')
    press_text = _cell_to_text(press_cell, suffix='atm') if press_cell else None

    # Residence time
    tau_cell = cells.get('residence_time')
    tau_text = _cell_to_text(tau_cell, suffix='s') if tau_cell else None

    # Equivalence ratio
    phi_cell = cells.get('equivalence_ratio')
    phi_text = _cell_to_text(phi_cell) if phi_cell else None

    # Composition: combine species columns
    comp_text = _build_composition_text(cells)

    # Determine scenario type from cell modes
    has_multi = any(
        c.mode in ('set', 'range') for c in cells.values()
    )
    scenario_type = "sweep" if has_multi else "single_point"

    # Structured conditions dict
    table_conditions = {k: v.to_dict() for k, v in cells.items()}

    row_id = f"case_{row.row_label}" if row.row_label else f"row_{row.row_index}"

    return ExperimentalScenario(
        scenario_id="",  # filled by caller
        experiment_family=family,
        scenario_type=scenario_type,
        source="table",
        source_label=f"{table.table_id} {row_id}",
        source_page=table.page_num,
        temperature_text=temp_text,
        pressure_text=press_text,
        composition_text=comp_text,
        residence_time_text=tau_text,
        equivalence_ratio_text=phi_text,
        observable=observable,
        simulatable=False,  # set later by caller
        notes=[f"from {table.table_id} row {row.row_label or row.row_index}"],
        table_row_id=row_id,
        table_conditions=table_conditions,
        expansion_state="unexpanded" if has_multi else None,
    )


def _cell_to_text(cell, suffix: str | None = None) -> str:
    """Convert a TableConditionValue to a human-readable text string."""
    unit = suffix or cell.unit or ''
    unit_sep = ' ' if unit else ''

    if cell.mode == 'formula':
        return cell.formula
    if cell.mode == 'set':
        vals = ', '.join(str(v) for v in cell.values)
        return f"{vals}{unit_sep}{unit}" if unit else vals
    if cell.mode == 'range':
        lo, hi = cell.values[0], cell.values[1]
        return f"{lo}-{hi}{unit_sep}{unit}" if unit else f"{lo}-{hi}"
    # single
    if cell.values:
        return f"{cell.values[0]}{unit_sep}{unit}" if unit else str(cell.values[0])
    return cell.raw_text


def _build_composition_text(cells: dict) -> str | None:
    """Build a composition string from species columns in the row."""
    parts: list[str] = []
    for key, cell in cells.items():
        if not key.startswith('species:'):
            continue
        species = key.split(':', 1)[1]
        unit = cell.unit or ''

        if cell.mode == 'set':
            vals = '/'.join(str(v) for v in cell.values)
            parts.append(f"{vals} {unit} {species}".strip())
        elif cell.mode == 'range':
            lo, hi = cell.values[0], cell.values[1]
            parts.append(f"{lo}-{hi} {unit} {species}".strip())
        elif cell.values:
            parts.append(f"{cell.values[0]} {unit} {species}".strip())

    return ', '.join(parts) if parts else None
