"""Classify evidence snippets by semantic role and select best conditions.

This is the domain-specific intelligence layer between raw evidence
extraction and the simulation planner. It answers: "which of these
extracted values are the actual experimental conditions?"
"""

from __future__ import annotations

import re

from src.schemas.experimental import (
    EvidenceSnippet, ExperimentCandidate, ROLE_PRIORITY,
    NormalizedCondition, NormalizedComposition,
)


# ── Role classification patterns ─────────────────────────────────────────

# Temperature role patterns (applied to source_text)
_TEMP_SETUP_RANGE = re.compile(
    r'(?:over\s+(?:a\s+)?(?:temperature\s+)?range|'
    r'temperature\s+range|'
    r'(?:ranging|range)\s+(?:from|of)|'
    r'between\s+\d|'
    r'(?:experiments?|measurements?|pyrolysis)\s+(?:were|was)\s+(?:performed|conducted|carried)\s+.*?\d+\s*[-\u2013\u2014]\s*\d+)',
    re.IGNORECASE,
)

_TEMP_SETPOINT = re.compile(
    r'(?:T\s*=|temperature\s+(?:of|at|was|is)\s+|'
    r'(?:at|near)\s+\d+\s*(?:\u00b0\s*)?K\b(?!.*?(?:measured|observed|delay|conversion)))',
    re.IGNORECASE,
)

_TEMP_CONSTRAINT = re.compile(
    r'(?:below|above|less\s+than|greater\s+than|up\s+to|exceed)',
    re.IGNORECASE,
)

_TEMP_MEASUREMENT = re.compile(
    r'(?:(?:measured|observed|reported|found)\s+.*?\d+\s*K|'
    r'(?:ignition|delay|conversion|decomposition)\s+.*?\d+\s*K|'
    r'\d+\s*K\s*.*?(?:measured|observed|shows?))',
    re.IGNORECASE,
)

# Pressure role patterns
_PRESS_INSTRUMENT = re.compile(
    r'(?:gauge|transducer|sensor|full[\-\s]?scale|vacuum|vacuumed|'
    r'pressure\s+measurement\s+device|MKS|Setra|Baratron)',
    re.IGNORECASE,
)

_PRESS_SETUP = re.compile(
    r'(?:(?:at|near|approximately|around)\s+(?:\d|atmospheric)|'
    r'(?:atmospheric|ambient)\s+pressure|'
    r'P\s*=\s*\d|'
    r'pressure\s+(?:of|at|was|is)\s+\d)',
    re.IGNORECASE,
)

_PRESS_SETUP_RANGE = re.compile(
    r'(?:pressure\s+range|pressures?\s+(?:of|from|between)\s+\d+\s*[-\u2013])',
    re.IGNORECASE,
)

# Composition role patterns
_COMP_SETUP = re.compile(
    r'(?:mixture|prepared|mixtures?\s+(?:of|containing)|'
    r'(?:initially|nominally)\s+prepared|'
    r'mole\s+fraction|'
    r'X_?\w*\s*=)',
    re.IGNORECASE,
)

# Residence time role patterns
_RESTIME_SETUP = re.compile(
    r'(?:residence\s+time\s+(?:of|was|is|=)|tau\s*=|\u03c4\s*=)',
    re.IGNORECASE,
)

# Range detection: does a value_text contain a range (two numbers)?
_RANGE_DETECT = re.compile(
    r'\d+(?:\.\d+)?\s*[-\u2013\u2014]\s*\d+(?:\.\d+)?|'
    r'\d+(?:\.\d+)?\s+to\s+\d+(?:\.\d+)?',
)


def normalize_candidate_conditions(
    candidate: ExperimentCandidate,
) -> ExperimentCandidate:
    """Classify roles on all condition evidence within a candidate.

    Mutates the evidence snippets in-place (sets .role field).
    Returns the same candidate for chaining.
    """
    for snippet in candidate.evidence:
        if snippet.role != "unclassified":
            continue
        snippet.role = _classify_role(snippet)
    return candidate


def select_best_condition(
    snippets: list[EvidenceSnippet],
    kind: str,
) -> EvidenceSnippet | None:
    """Select the best evidence snippet for a given condition kind.

    Priority: role > is_range > range_width > section_weight > confidence.
    Wider ranges are preferred (they represent the full experimental scope).
    """
    matching = [s for s in snippets if s.kind == kind and s.role != "instrument"]
    if not matching:
        return None

    return max(matching, key=lambda s: (
        ROLE_PRIORITY.get(s.role, 1),
        _is_range(s.value_text),
        _range_width(s.value_text),
        s.section_weight,
        s.confidence,
    ))


def select_all_by_role(
    snippets: list[EvidenceSnippet],
    kind: str,
    roles: set[str] | None = None,
) -> list[EvidenceSnippet]:
    """Get all snippets of a kind filtered to specific roles."""
    if roles is None:
        roles = {"setup_range", "setpoint"}
    return [
        s for s in snippets
        if s.kind == kind and s.role in roles
    ]


# ── Role classification logic ────────────────────────────────────────────

def _classify_role(snippet: EvidenceSnippet) -> str:
    """Classify the semantic role of an evidence snippet."""
    # Background: anything from intro or abstract with low weight
    if snippet.section in ("introduction", "abstract") and snippet.section_weight <= 0.3:
        return "background"

    ctx = snippet.source_text
    kind = snippet.kind

    if kind == "temperature":
        return _classify_temperature(ctx)
    elif kind == "pressure":
        return _classify_pressure(ctx)
    elif kind == "composition":
        return _classify_composition(ctx)
    elif kind == "residence_time":
        return _classify_residence_time(ctx)
    elif kind in ("experiment_family", "observable_type"):
        return "unclassified"  # not a condition
    else:
        # flow_rate, reactor_volume, mechanism, equivalence_ratio
        return "setpoint" if snippet.section == "methods" else "measurement"


def _classify_temperature(ctx: str) -> str:
    if _TEMP_CONSTRAINT.search(ctx):
        return "constraint"
    if _TEMP_SETUP_RANGE.search(ctx):
        return "setup_range"
    if _TEMP_MEASUREMENT.search(ctx):
        return "measurement"
    if _TEMP_SETPOINT.search(ctx):
        return "setpoint"
    return "unclassified"


def _classify_pressure(ctx: str) -> str:
    if _PRESS_INSTRUMENT.search(ctx):
        return "instrument"
    if _PRESS_SETUP_RANGE.search(ctx):
        return "setup_range"
    if _PRESS_SETUP.search(ctx):
        return "setpoint"
    return "unclassified"


def _classify_composition(ctx: str) -> str:
    if _COMP_SETUP.search(ctx):
        return "setpoint"
    return "measurement"


def _classify_residence_time(ctx: str) -> str:
    if _RESTIME_SETUP.search(ctx):
        return "setpoint"
    return "measurement"


def _is_range(value_text: str) -> bool:
    """Check if a value_text contains a range (two numbers with dash/to)."""
    return bool(_RANGE_DETECT.search(value_text))


_NUM_RE = re.compile(r'(\d+(?:\.\d+)?)')


def _range_width(value_text: str) -> float:
    """Compute the width of a numeric range. Wider = more experimental scope."""
    nums = _NUM_RE.findall(value_text)
    if len(nums) >= 2:
        values = [float(n) for n in nums]
        return max(values) - min(values)
    return 0.0


# ── Structured condition builders ────────────────────────────────────────

_PRESSURE_UNITS = {
    "atm": "atm", "bar": "bar", "kpa": "kPa", "mpa": "MPa",
    "torr": "Torr", "psi": "psi", "pa": "Pa",
}


def build_normalized_condition(
    snippet: EvidenceSnippet,
    default_unit: str,
) -> NormalizedCondition:
    """Convert an evidence snippet into a structured NormalizedCondition."""
    nums = _NUM_RE.findall(snippet.value_text)
    values = [float(n) for n in nums] if nums else [0.0]

    # Detect unit from value text
    unit = _detect_unit(snippet.value_text, default_unit)

    min_val = min(values) if values else 0.0
    max_val = max(values) if len(values) >= 2 else min_val

    return NormalizedCondition(
        raw_text=snippet.value_text,
        kind=snippet.kind,
        role=snippet.role,
        is_range=len(values) >= 2 and min_val != max_val,
        min_value=min_val,
        max_value=max_val,
        unit=unit,
        source_section=snippet.section,
        source_page=snippet.page_num,
    )


def _detect_unit(text: str, default: str) -> str:
    """Detect the physical unit from value text."""
    # Temperature units — match after a digit to avoid "from", "for" etc.
    if re.search(r'\d\s*(?:\u00b0\s*)?C\b', text):
        return "C"
    if re.search(r'\d\s*(?:\u00b0\s*)?F\b', text):
        return "F"
    if re.search(r'\d\s*K\b', text):
        return "K"

    # Pressure units
    text_lower = text.lower()
    for key, canonical in _PRESSURE_UNITS.items():
        if key in text_lower:
            return canonical

    # Time units
    if "ms" in text_lower or "millisecond" in text_lower:
        return "ms"
    if re.search(r'\d\s*s\b', text) or "second" in text_lower:
        return "s"

    return default


# ── Composition parsing ──────────────────────────────────────────────────

# Common combustion species for validation
_KNOWN_SPECIES = {
    "NH3", "H2", "O2", "N2", "Ar", "He", "Ne", "Kr",
    "CH4", "C2H6", "C2H4", "C2H2", "CO", "CO2", "H2O",
    "NO", "NO2", "N2O", "NH2", "OH", "HCN", "CH3OH",
    "C3H8", "C4H10", "CH2O",
}

# Pattern: "0.5% NH3" or "NH3 0.5%" or "0.005 NH3"
_SPECIES_FRACTION = re.compile(
    r'(?:(\d+(?:\.\d+)?)\s*%?\s+([A-Z][A-Za-z0-9]*)\b|'
    r'([A-Z][A-Za-z0-9]*)\s*[=:]\s*(\d+(?:\.\d+)?)\s*%?|'
    r'(\d+(?:\.\d+)?)\s*%\s*([A-Z][A-Za-z0-9]*))',
)

# Pattern: "X_NH3 = 0.005" or "mole fraction of NH3 = 0.005"
_MOLE_FRACTION = re.compile(
    r'(?:X_?\s*(?:of\s+)?([A-Z][A-Za-z0-9]*)\s*=\s*(\d+\.?\d*)|'
    r'mole\s+fraction\s+(?:of\s+)?([A-Z][A-Za-z0-9]*)\s*(?:=|:|\s+)\s*(\d+\.?\d*))',
    re.IGNORECASE,
)

# Pattern: "CH4/O2/Ar" or "NH3/H2/Ar"
_MIXTURE_SLASH = re.compile(
    r'([A-Z][A-Za-z0-9]*(?:\s*/\s*[A-Z][A-Za-z0-9]*){1,})',
)

# Pattern: "diluted in Ar", "balanced by N2"
_DILUENT = re.compile(
    r'(?:dilut\w+\s+(?:in|with)|balance[d]?\s+(?:by|with)|bath\s+gas\s*[:=]?\s*|'
    r'(?:in|with)\s+(?:pure\s+)?)(Ar|N2|He|Ne|Kr)\b',
    re.IGNORECASE,
)


def _collapse_species_spaces(text: str) -> str:
    """Fix fitz artifact: 'NH 3' → 'NH3', 'C 2 H 6' → 'C2H6', 'H 2 O' → 'H2O'.

    Collapses spaces between uppercase letters, digits, and lowercase letters
    that form chemical formulas.
    """
    # Collapse: letter + space + digit → letter+digit (e.g., "NH 3" → "NH3")
    text = re.sub(r'([A-Za-z]) (\d)', r'\1\2', text)
    # Collapse: digit + space + letter → digit+letter (e.g., "C2 H" → "C2H")
    text = re.sub(r'(\d) ([A-Z])', r'\1\2', text)
    return text


def build_normalized_composition(
    evidence: list[EvidenceSnippet],
) -> NormalizedComposition | None:
    """Parse composition evidence into structured species fractions.

    Handles:
    - "0.5% NH3 in Ar" → {NH3: 0.005, Ar: balance}
    - "X_NH3 = 0.005" → {NH3: 0.005}
    - "diluted in Ar" → {Ar: balance}
    """
    comp_evidence = [e for e in evidence if e.kind == "composition"]
    # Also scan methods-section evidence for species fractions (they often
    # appear near temperature/pressure mentions, not just composition evidence)
    methods_evidence = [
        e for e in evidence
        if e.section == "methods" and e.kind != "composition"
    ]
    if not comp_evidence and not methods_evidence:
        return None

    species: dict[str, float] = {}
    balance_species: str | None = None
    raw_mentions: list[str] = []
    source_pages: list[int] = []

    for ev in comp_evidence:
        raw_mentions.append(ev.value_text)
        if ev.page_num not in source_pages:
            source_pages.append(ev.page_num)

        text = ev.value_text
        ctx = _collapse_species_spaces(ev.source_text)

        # Try mole fraction patterns first (most precise)
        for match in _MOLE_FRACTION.finditer(ctx):
            sp = match.group(1) or match.group(3)
            val = float(match.group(2) or match.group(4))
            if sp and sp in _KNOWN_SPECIES:
                # If value > 1, treat as percentage
                species[sp] = val / 100.0 if val > 1.0 else val

        # Try percentage + species patterns
        for match in _SPECIES_FRACTION.finditer(ctx):
            groups = match.groups()
            match_text = match.group(0)
            has_percent = "%" in match_text
            for i in range(0, len(groups), 2):
                num_str, sp_str = groups[i], groups[i + 1] if i + 1 < len(groups) else None
                if num_str and sp_str and sp_str in _KNOWN_SPECIES:
                    val = float(num_str)
                    # Always divide by 100 if % is present, or if value > 1
                    if has_percent or val > 1.0:
                        val /= 100.0
                    species[sp_str] = val
                    break

        # Detect diluent / balance gas
        diluent_match = _DILUENT.search(ctx)
        if diluent_match:
            balance_species = diluent_match.group(1)

    # Also scan methods evidence context for species fractions
    if not species:
        for ev in methods_evidence:
            ctx = _collapse_species_spaces(ev.source_text)
            for match in _SPECIES_FRACTION.finditer(ctx):
                groups = match.groups()
                has_percent = "%" in match.group(0)
                for i in range(0, len(groups), 2):
                    num_str = groups[i]
                    sp_str = groups[i + 1] if i + 1 < len(groups) else None
                    if num_str and sp_str and sp_str in _KNOWN_SPECIES:
                        val = float(num_str)
                        if has_percent or val > 1.0:
                            val /= 100.0
                        species[sp_str] = val
                        if ev.page_num not in source_pages:
                            source_pages.append(ev.page_num)
                        break
            diluent_m = _DILUENT.search(ctx)
            if diluent_m and not balance_species:
                balance_species = diluent_m.group(1)

    # If we found species but no balance, check if any known diluent is mentioned
    if species and not balance_species:
        for ev in comp_evidence:
            m = _DILUENT.search(_collapse_species_spaces(ev.source_text))
            if m:
                balance_species = m.group(1)
                break

    # Compute completeness
    total = sum(species.values())
    if balance_species and balance_species not in species and total < 1.0:
        species[balance_species] = round(1.0 - total, 6)
        composition_complete = True
    else:
        composition_complete = abs(total - 1.0) < 0.01

    if not species and not balance_species:
        # Couldn't parse anything structured, return partial
        return NormalizedComposition(
            species={},
            balance_species=balance_species,
            composition_complete=False,
            raw_mentions=raw_mentions,
            source_pages=source_pages,
        )

    return NormalizedComposition(
        species=species,
        balance_species=balance_species,
        composition_complete=composition_complete,
        raw_mentions=raw_mentions,
        source_pages=source_pages,
    )
