"""Detect and parse structured condition tables from PDF page text.

Identifies tables whose columns map to experimental conditions (T, P, tau, phi,
species concentrations) and extracts rows as structured data.

Handles common PyMuPDF text-extraction artifacts:
- Greek symbols rendered as ASCII: tau->t, phi->F
- En-dash rendered as 'e': 850e1350 -> 850-1350
- CJK ideographic commas (U+3001): 0.5、1、1.5 -> [0.5, 1.0, 1.5]
"""

from __future__ import annotations

import re

from src.schemas.experimental import (
    ParsedTable, TableColumn, TableRow, TableConditionValue, FigureCaption,
)


# -- Column header recognition ------------------------------------------------
#
# Each rule: (regex, role, unit_override)
# For species columns, group(1) captures the species name.

_HEADER_RULES: list[tuple[re.Pattern, str, str | None]] = [
    # Temperature
    (re.compile(r'^T\s*\(?K\)?$'), 'temperature', 'K'),
    (re.compile(r'^Temperature', re.I), 'temperature', 'K'),

    # Pressure
    (re.compile(r'^P\s*\((atm|MPa|bar|kPa)\)$', re.I), 'pressure', None),
    (re.compile(r'^Pressure', re.I), 'pressure', 'atm'),

    # Residence time -- tau often extracted as bare 't' by PyMuPDF
    (re.compile(r'^[\u03c4\u03c4]$'), 'residence_time', 's'),          # tau
    (re.compile(r'^t$'), 'residence_time', 's'),                        # ASCII fallback
    (re.compile(r'^(?:\u03c4|tau|residence)', re.I), 'residence_time', 's'),

    # Equivalence ratio -- phi often extracted as bare 'F' by PyMuPDF
    (re.compile(r'^[\u03a6\u03c6\u0278]$'), 'equivalence_ratio', None), # Phi, phi, latin phi
    (re.compile(r'^F$'), 'equivalence_ratio', None),                    # ASCII fallback
    (re.compile(r'^(?:\u03a6|\u03c6|phi|equivalence)', re.I), 'equivalence_ratio', None),

    # Species with explicit unit
    (re.compile(r'^(\w+)\s*\(ppm\)$'), 'species', 'ppm'),
    (re.compile(r'^(\w+)\s*\(vol%?\)$', re.I), 'species', 'vol%'),
    (re.compile(r'^(\w+)\s*\(mol%?\)$', re.I), 'species', 'mol%'),
    (re.compile(r'^X_?(\w+)$'), 'species', 'mole_fraction'),

    # Identifier columns (consumed but not stored as conditions)
    (re.compile(r'^(?:Case|No\.?|#)$', re.I), 'identifier', None),

    # Reference column -- signals literature-review table (skip entire table)
    (re.compile(r'^Reference', re.I), 'reference', None),
]


def _classify_header(text: str) -> tuple[str, str | None, str | None]:
    """Classify a column header text -> (role, species_name, unit)."""
    text = text.strip()
    for pattern, role, unit in _HEADER_RULES:
        m = pattern.match(text)
        if m:
            species = m.group(1) if role == 'species' and m.lastindex else None
            if role == 'pressure' and m.lastindex:
                unit = m.group(1)
            return role, species, unit
    return 'unknown', None, None


# -- Table detection -----------------------------------------------------------

_CONDITION_CAPTION_KW = re.compile(
    r'(?:experimental|operating|test)\s+conditions?|'
    r'considered\s+cases|'
    r'experimental\s+cases|'
    r'test\s+matrix',
    re.IGNORECASE,
)


def parse_tables(pages: list, captions: list[FigureCaption]) -> list[ParsedTable]:
    """Scan pages for condition tables and parse them into structured data.

    Only processes tables whose captions indicate experimental conditions.
    Returns one ParsedTable per successfully parsed condition table.
    """
    tables: list[ParsedTable] = []

    table_captions = [c for c in captions if c.label_type == "table"]

    for caption in table_captions:
        if not _CONDITION_CAPTION_KW.search(caption.caption):
            continue

        # Search caption's page and the next page (tables can span)
        for page in pages:
            if page.page_num not in (caption.page_num, caption.page_num + 1):
                continue
            parsed = _parse_table_from_text(page.text, caption)
            if parsed and parsed.rows:
                tables.append(parsed)
                break  # found it, don't check next page

    return tables


# -- Core parsing --------------------------------------------------------------

def _parse_table_from_text(text: str, caption: FigureCaption) -> ParsedTable | None:
    """Parse a condition table from page text given its caption."""
    lines = text.split('\n')

    header_start = _find_header_start(lines, caption.figure_id)
    if header_start is None:
        return None

    columns, header_end = _read_headers(lines, header_start)
    if len(columns) < 3:
        return None

    # Must have at least one real condition column
    condition_roles = {'temperature', 'pressure', 'residence_time',
                       'equivalence_ratio', 'species'}
    if not any(c.role in condition_roles for c in columns):
        return None

    # Reject literature-review tables (have a "Reference" column)
    if any(c.role == 'reference' for c in columns):
        return None

    rows = _read_rows(lines, header_end, columns)

    return ParsedTable(
        table_id=caption.figure_id,
        page_num=caption.page_num,
        caption=caption.caption,
        columns=columns,
        rows=rows,
    )


def _find_header_start(lines: list[str], table_id: str) -> int | None:
    """Find the line index where the table column headers begin.

    Searches for the last occurrence of the table ID or condition keywords,
    then scans forward for the first recognized column header.
    """
    anchor_idx = None
    table_id_compact = table_id.replace(" ", "").lower()

    for i, line in enumerate(lines):
        stripped = line.strip()
        if table_id_compact in stripped.replace(" ", "").lower():
            anchor_idx = i
        if _CONDITION_CAPTION_KW.search(stripped):
            anchor_idx = i

    if anchor_idx is None:
        return None

    # Scan forward from anchor to find first column header
    for i in range(anchor_idx, min(anchor_idx + 15, len(lines))):
        role, _, _ = _classify_header(lines[i].strip())
        if role not in ('unknown', 'reference'):
            return i

    return None


def _read_headers(
    lines: list[str], start: int,
) -> tuple[list[TableColumn], int]:
    """Read consecutive column headers starting at `start`.

    Returns (columns, index_of_first_data_line).
    """
    columns: list[TableColumn] = []
    i = start

    while i < len(lines):
        text = lines[i].strip()
        if not text:
            i += 1
            continue

        role, species, unit = _classify_header(text)
        if role == 'unknown':
            break

        columns.append(TableColumn(
            header=text, role=role, species=species, unit=unit,
        ))
        i += 1

    return columns, i


def _read_rows(
    lines: list[str],
    start: int,
    columns: list[TableColumn],
) -> list[TableRow]:
    """Read data rows: each row is len(columns) consecutive non-empty lines."""
    rows: list[TableRow] = []
    n_cols = len(columns)
    i = start
    row_idx = 0

    while i < len(lines):
        # Collect next n_cols non-empty values
        values: list[str] = []
        j = i
        while len(values) < n_cols and j < len(lines):
            text = lines[j].strip()
            if not text:
                j += 1
                continue
            if _is_stop_line(text):
                return rows
            values.append(text)
            j += 1

        if len(values) < n_cols:
            break

        # Map values to columns
        row_label = None
        cells: dict[str, TableConditionValue] = {}

        for col, val in zip(columns, values):
            if col.role == 'identifier':
                row_label = val
                continue
            if col.role == 'reference':
                continue

            cell = _parse_cell(val, col)
            key = f"species:{col.species}" if col.role == 'species' else col.role
            cells[key] = cell

        if cells:
            rows.append(TableRow(
                row_index=row_idx,
                row_label=row_label,
                cells=cells,
            ))
            row_idx += 1

        i = j

    return rows


def _is_stop_line(text: str) -> bool:
    """Return True if this line signals the end of table data."""
    # Spaced-out journal header/footer
    if re.match(r'^[a-z]\s[a-z]\s[a-z]\s[a-z]', text):
        return True
    # "Please cite this article..."
    if text.lower().startswith(('please cite', 'http', 'doi:')):
        return True
    # Start of next table or section
    if re.match(r'^Table\s+\d+', text, re.I):
        return True
    # Section headers like "Verification of ..."
    if re.match(r'^[A-Z][a-z]+\s+(?:of|and|the|in)\s', text) and not any(
        c.isdigit() for c in text
    ):
        return True
    return False


# -- Cell value parsing --------------------------------------------------------

_CJK_COMMA = '\u3001'


def _parse_cell(raw: str, column: TableColumn) -> TableConditionValue:
    """Parse a single table cell into a structured value with mode detection."""
    text = raw.replace(_CJK_COMMA, ', ')

    # Formula: "894/T", "V/Q"
    if re.match(r'^\d+(?:\.\d+)?\s*/\s*[A-Za-z]', text):
        return TableConditionValue(
            mode='formula', raw_text=text, unit=column.unit, formula=text,
        )

    # Set: comma-separated values "2, 1, 0.7"
    if ', ' in text:
        parts = [p.strip() for p in text.split(',')]
        try:
            values = [float(p) for p in parts]
            return TableConditionValue(
                mode='set', raw_text=text, values=values, unit=column.unit,
            )
        except ValueError:
            pass

    # Range via 'e' artifact: "850e1350", "0.5e2" (PyMuPDF en-dash -> 'e')
    # Distinguish from scientific notation: require both sides to be
    # plausible standalone values (exponent side is not tiny integer-only)
    e_match = re.match(r'^(\d+(?:\.\d+)?)e(\d+(?:\.\d+)?)$', text)
    if e_match:
        a, b = float(e_match.group(1)), float(e_match.group(2))
        # Scientific notation would typically have integer exponent >=2
        # and the result would be >> a.  Treat as range if both sides
        # are within plausible physical bounds for the column.
        if _looks_like_range(a, b, column):
            return TableConditionValue(
                mode='range', raw_text=text,
                values=[min(a, b), max(a, b)], unit=column.unit,
            )

    # Range via real dash: "850-1350", "0.5-2"
    dash_match = re.match(
        r'^(\d+(?:\.\d+)?)\s*[-\u2013\u2014\u2212]\s*(\d+(?:\.\d+)?)$', text,
    )
    if dash_match:
        a, b = float(dash_match.group(1)), float(dash_match.group(2))
        return TableConditionValue(
            mode='range', raw_text=text,
            values=[min(a, b), max(a, b)], unit=column.unit,
        )

    # Single numeric value
    try:
        val = float(text)
        return TableConditionValue(
            mode='single', raw_text=text, values=[val], unit=column.unit,
        )
    except ValueError:
        return TableConditionValue(
            mode='single', raw_text=text, unit=column.unit,
        )


def _looks_like_range(a: float, b: float, column: TableColumn) -> bool:
    """Heuristic: does 'aeb' look like a range rather than scientific notation?

    True when both sides are plausible standalone values for the column type.
    """
    # Per-role plausibility: both sides must be individually reasonable
    if column.role == 'temperature':
        return a < 5000 and b < 5000
    if column.role == 'species':
        return a <= 100 and b <= 100
    if column.role == 'equivalence_ratio':
        return a <= 10 and b <= 10
    if column.role == 'residence_time':
        return a <= 1000 and b <= 1000
    if column.role == 'pressure':
        return a <= 1000 and b <= 1000
    # Default: if b > 10, it's not a realistic exponent -> treat as range
    return b > 10 or (a > 1 and b > 1)
