"""Extract simulation-relevant evidence from parsed paper documents.

Each extractor function scans text for a specific evidence kind using
regex + heuristics and returns EvidenceSnippet objects with page references.
"""

from __future__ import annotations

import re

from src.schemas.experimental import PaperDocument, EvidenceSnippet, EvidenceKind, PageText, SectionSpan
from src.ingestion.pdf_parser.section_detector import get_section_for_page
from src.ingestion.pipeline.families import get_all_families

# ---------------------------------------------------------------------------
# Shared numeric pattern building blocks
# ---------------------------------------------------------------------------
# Handles: 950, 1.5, 1.5e3, 1.5x10^3, 1.5×10³
_NUM = r'(?:\d+(?:\.\d+)?(?:\s*[xX\u00d7]\s*10\s*[\^]?\s*\d+)?)'
# Dash variants: hyphen, en-dash, em-dash, minus sign
_DASH = r'[-\u2013\u2014\u2212]'
_RANGE = rf'({_NUM})\s*(?:{_DASH}|to)\s*({_NUM})'


# ---------------------------------------------------------------------------
# 1a. Experiment family — patterns from family registry
# ---------------------------------------------------------------------------
def _extract_experiment_family(text: str, page_num: int) -> list[EvidenceSnippet]:
    snippets = []
    for name, rules in get_all_families().items():
        for pattern in rules.anchor_patterns:
            for match in pattern.finditer(text):
                matched = match.group(0)
                conf = 0.9 if matched.isupper() and len(matched) <= 4 else 0.85
                snippets.append(EvidenceSnippet(
                    kind="experiment_family",
                    value_text=name,
                    page_num=page_num,
                    source_text=_get_context(text, match),
                    confidence=conf,
                ))
    return snippets


# ---------------------------------------------------------------------------
# 1b. Observable type — patterns from family registry (merged across families)
# ---------------------------------------------------------------------------
def _build_observable_patterns() -> dict[str, re.Pattern]:
    """Collect all unique observable patterns across all families."""
    merged: dict[str, re.Pattern] = {}
    for rules in get_all_families().values():
        for obs_type, pattern in rules.observable_patterns.items():
            if obs_type not in merged:
                merged[obs_type] = pattern
    return merged


_OBSERVABLE_PATTERNS = _build_observable_patterns()


def _extract_observable_type(text: str, page_num: int) -> list[EvidenceSnippet]:
    snippets = []
    for obs_type, pattern in _OBSERVABLE_PATTERNS.items():
        for match in pattern.finditer(text):
            matched = match.group(0)
            conf = 0.9 if matched.isupper() and len(matched) <= 4 else 0.80
            snippets.append(EvidenceSnippet(
                kind="observable_type",
                value_text=obs_type,
                page_num=page_num,
                source_text=_get_context(text, match),
                confidence=conf,
            ))
    return snippets


# ---------------------------------------------------------------------------
# 2. Temperature
# ---------------------------------------------------------------------------
_TEMP_UNIT = r'(?:\s*(?:\u00b0\s*)?[KCF]\b)'
_TEMP_PATTERNS = [
    # "T = 950-1150 K" or "temperature range of 950 to 1150 K"
    re.compile(
        rf'(?:T\s*=\s*|[Tt]emperature[s]?\s+(?:of|at|from|between|ranging\s+from)\s+)?'
        rf'{_RANGE}{_TEMP_UNIT}',
    ),
    # Single value: "950 K", "T = 1000 K"
    re.compile(
        rf'(?:T\s*=\s*)?({_NUM}){_TEMP_UNIT}',
    ),
]


def _extract_temperature(text: str, page_num: int) -> list[EvidenceSnippet]:
    snippets = []
    for pattern in _TEMP_PATTERNS:
        for match in pattern.finditer(text):
            value = match.group(0).strip()
            if _is_likely_year(value):
                continue
            conf = 0.85 if re.search(r'(?:T\s*=|[Tt]emperature)', value) else 0.70
            snippets.append(EvidenceSnippet(
                kind="temperature",
                value_text=value,
                page_num=page_num,
                source_text=_get_context(text, match),
                confidence=conf,
            ))
    return snippets


# ---------------------------------------------------------------------------
# 3. Pressure
# ---------------------------------------------------------------------------
_PRESSURE_UNIT = r'(?:\s*(?:atm|bar|kPa|MPa|Torr|torr|psi|Pa)\b)'
_PRESSURE_PATTERNS = [
    # Range: "1-10 atm", "P = 1 to 5 bar"
    re.compile(
        rf'(?:P\s*=\s*|[Pp]ressure\s+(?:of|at)\s+)?{_RANGE}{_PRESSURE_UNIT}',
    ),
    # Single: "1 atm", "P = 101.325 kPa"
    re.compile(
        rf'(?:P\s*=\s*|[Pp]ressure\s+(?:of|at)\s+)?({_NUM}){_PRESSURE_UNIT}',
    ),
]


def _extract_pressure(text: str, page_num: int) -> list[EvidenceSnippet]:
    snippets = []
    for pattern in _PRESSURE_PATTERNS:
        for match in pattern.finditer(text):
            value = match.group(0).strip()
            conf = 0.90 if re.search(r'(?:P\s*=|[Pp]ressure)', value) else 0.70
            snippets.append(EvidenceSnippet(
                kind="pressure",
                value_text=value,
                page_num=page_num,
                source_text=_get_context(text, match),
                confidence=conf,
            ))
    return snippets


# ---------------------------------------------------------------------------
# 4. Residence time
# ---------------------------------------------------------------------------
_TIME_UNIT = r'(?:\s*(?:m?s|seconds?|milliseconds?)\b)'
_RESIDENCE_PATTERNS = [
    # "residence time of 0.07 s", "tau = 70 ms", "τ = 2.0 s"
    re.compile(
        rf'(?:residence\s+time[s]?\s+(?:of\s+)?|(?:tau|\u03c4|t_res)\s*=\s*)'
        rf'({_NUM}){_TIME_UNIT}',
        re.IGNORECASE,
    ),
    # Range: "residence times of 0.5-2.0 s"
    re.compile(
        rf'(?:residence\s+time[s]?\s+(?:of\s+)?|(?:tau|\u03c4)\s*=\s*)'
        rf'{_RANGE}{_TIME_UNIT}',
        re.IGNORECASE,
    ),
]


def _extract_residence_time(text: str, page_num: int) -> list[EvidenceSnippet]:
    snippets = []
    for pattern in _RESIDENCE_PATTERNS:
        for match in pattern.finditer(text):
            snippets.append(EvidenceSnippet(
                kind="residence_time",
                value_text=match.group(0).strip(),
                page_num=page_num,
                source_text=_get_context(text, match),
                confidence=0.90,
            ))
    return snippets


# ---------------------------------------------------------------------------
# 5. Equivalence ratio
# ---------------------------------------------------------------------------
_EQUIV_PATTERNS = [
    # "phi = 0.5", "φ = 0.5-2.0", "equivalence ratio of 0.5, 1.0, and 2.0"
    re.compile(
        rf'(?:equivalence\s+ratio[s]?\s+(?:of\s+)?|(?:phi|\u03c6|\u0278)\s*=\s*)'
        rf'((?:{_NUM}\s*[,]\s*)*{_NUM}(?:\s+and\s+{_NUM})?)',
        re.IGNORECASE,
    ),
    # Range: "phi = 0.5-2.0"
    re.compile(
        rf'(?:equivalence\s+ratio[s]?\s+(?:of\s+)?|(?:phi|\u03c6|\u0278)\s*=\s*)'
        rf'{_RANGE}',
        re.IGNORECASE,
    ),
]


def _extract_equivalence_ratio(text: str, page_num: int) -> list[EvidenceSnippet]:
    snippets = []
    for pattern in _EQUIV_PATTERNS:
        for match in pattern.finditer(text):
            snippets.append(EvidenceSnippet(
                kind="equivalence_ratio",
                value_text=match.group(0).strip(),
                page_num=page_num,
                source_text=_get_context(text, match),
                confidence=0.90,
            ))
    return snippets


# ---------------------------------------------------------------------------
# 6. Composition
# ---------------------------------------------------------------------------
_COMPOSITION_PATTERNS = [
    # "X_CH4 = 0.01", "mole fraction of CH4 = 0.01"
    re.compile(
        r'(?:mole\s+fraction|mol\s*%|X_?\s*)\s*(?:of\s+)?'
        r'([A-Z][A-Za-z0-9]*)\s*(?:=|:)\s*(\d+\.?\d*)',
        re.IGNORECASE,
    ),
    # "CH4/O2/Ar = 1/0.5/3.76" or "50% CH4 / 50% O2"
    re.compile(
        r'(?:\d+(?:\.\d+)?%?\s*)?(?:[A-Z][a-z]?\d*)+(?:\s*/\s*(?:\d+(?:\.\d+)?%?\s*)?(?:[A-Z][a-z]?\d*)+){1,}',
    ),
    # "diluted in Ar", "bath gas: N2", "balanced by N2"
    re.compile(
        r'(?:dilut\w+\s+(?:in|with)|bath\s+gas|balance[d]?\s+(?:by|with)|inert[\s:]+)'
        r'\s*(Ar|N2|He|Ne|Kr)',
        re.IGNORECASE,
    ),
]


def _extract_composition(text: str, page_num: int) -> list[EvidenceSnippet]:
    snippets = []
    for pattern in _COMPOSITION_PATTERNS:
        for match in pattern.finditer(text):
            value = match.group(0).strip()
            # Boost if near composition context words
            context = _get_context(text, match)
            conf = 0.75
            if re.search(r'(?i)(?:mixture|composition|mole\s+fraction|dilut)', context):
                conf = 0.85
            snippets.append(EvidenceSnippet(
                kind="composition",
                value_text=value,
                page_num=page_num,
                source_text=context,
                confidence=conf,
            ))
    return snippets


# ---------------------------------------------------------------------------
# 7. Mechanism
# ---------------------------------------------------------------------------
_MECHANISM_PATTERNS = [
    # Named mechanisms
    re.compile(
        r'(?:GRI[\-\s]?Mech\s*\d[\.\d]*|AramcoMech\s*[\d.]+|USC\s+Mech\s*\w*|'
        r'San\s+Diego\s+mechanism|CRECK|Jet[Ss]urf\s*[\d.]*|'
        r'NUI\s+Galway|NUIG|Caltech\s+mechanism)',
        re.IGNORECASE,
    ),
    # "mechanism of Smith et al." or "kinetic model developed by ..."
    # Requires a capitalized proper name (not adjectives like "selected", "several")
    re.compile(
        r'(?:mechanism|kinetic\s+model|chemical\s+(?:kinetic\s+)?mechanism)\s+'
        r'(?:of|by|from|developed\s+by|proposed\s+by)\s+'
        r'(?!(?:selected|several|various|different|the|this|that|each|some|many|most|all|their|our|a)\b)'
        r'([A-Z][a-z]+(?:\s+et\s+al\.?)?)',
        re.IGNORECASE,
    ),
    # "mechanism contains 53 species and 325 reactions"
    re.compile(
        r'(?:mechanism|model)\s+(?:contain(?:s|ing)|consist(?:s|ing)\s+of|with)\s+'
        r'(\d+)\s+species\s+and\s+(\d+)\s+reactions',
        re.IGNORECASE,
    ),
]


def _extract_mechanism(text: str, page_num: int) -> list[EvidenceSnippet]:
    snippets = []
    for pattern in _MECHANISM_PATTERNS:
        for match in pattern.finditer(text):
            value = match.group(0).strip()
            # Named mechanisms get high confidence
            conf = 0.90 if re.search(
                r'(?i)(?:GRI|Aramco|USC|CRECK|JetSurf|San\s+Diego|NUIG)', value
            ) else 0.65
            snippets.append(EvidenceSnippet(
                kind="mechanism",
                value_text=value,
                page_num=page_num,
                source_text=_get_context(text, match),
                confidence=conf,
            ))
    return snippets


# ---------------------------------------------------------------------------
# 8. Flow rate
# ---------------------------------------------------------------------------
_FLOW_PATTERNS = [
    # "flow rate of 30 L/min", "total flow = 500 sccm"
    re.compile(
        rf'(?:(?:total\s+|gas\s+)?flow\s*(?:rate)?\s*(?:of|=|:)?\s*)'
        rf'({_NUM})\s*'
        r'(?:L|mL|cm3|cc|slpm|sccm|NL|Nm3)(?:\s*/\s*(?:min|s|h))?',
        re.IGNORECASE,
    ),
]


def _extract_flow_rate(text: str, page_num: int) -> list[EvidenceSnippet]:
    snippets = []
    for pattern in _FLOW_PATTERNS:
        for match in pattern.finditer(text):
            snippets.append(EvidenceSnippet(
                kind="flow_rate",
                value_text=match.group(0).strip(),
                page_num=page_num,
                source_text=_get_context(text, match),
                confidence=0.85,
            ))
    return snippets


# ---------------------------------------------------------------------------
# 9. Reactor volume
# ---------------------------------------------------------------------------
_VOLUME_PATTERNS = [
    # "reactor volume of 90 cm3", "V = 29.5 mL"
    re.compile(
        rf'(?:(?:reactor\s+)?volume\s*(?:of|=|:)?\s*|V\s*=\s*)'
        rf'({_NUM})\s*(?:L|mL|cm[\^]?3|m3|cc)\b',
        re.IGNORECASE,
    ),
    # "90 cm3 reactor"
    re.compile(
        rf'({_NUM})\s*(?:L|mL|cm[\^]?3|cc)\s+(?:reactor|vessel|chamber)',
        re.IGNORECASE,
    ),
]


def _extract_reactor_volume(text: str, page_num: int) -> list[EvidenceSnippet]:
    snippets = []
    for pattern in _VOLUME_PATTERNS:
        for match in pattern.finditer(text):
            snippets.append(EvidenceSnippet(
                kind="reactor_volume",
                value_text=match.group(0).strip(),
                page_num=page_num,
                source_text=_get_context(text, match),
                confidence=0.85,
            ))
    return snippets


# ---------------------------------------------------------------------------
# Extractor registry
# ---------------------------------------------------------------------------
_EXTRACTORS = [
    _extract_experiment_family,
    _extract_observable_type,
    _extract_temperature,
    _extract_pressure,
    _extract_residence_time,
    _extract_equivalence_ratio,
    _extract_composition,
    _extract_mechanism,
    _extract_flow_rate,
    _extract_reactor_volume,
]


# ---------------------------------------------------------------------------
# Setup-phrase detection (boosts confidence for the paper's own experiments)
# ---------------------------------------------------------------------------
_SETUP_PHRASES = re.compile(
    r'(?:experiments?\s+(?:were|was)\s+(?:performed|conducted|carried\s+out)|'
    r'(?:the\s+)?reactor\s+(?:was|is)\s+operated|'
    r'(?:ignition\s+delay|IDT)\s+(?:times?\s+)?(?:were|was)\s+measured|'
    r'mixtures?\s+(?:were|was)\s+prepared|'
    r'measurements?\s+(?:were|was)\s+(?:performed|conducted|taken|obtained)|'
    r'(?:the\s+)?(?:temperature|pressure)\s+(?:was|were)\s+(?:set|fixed|maintained|controlled)|'
    r'residence\s+time\s+(?:was|were)\s+(?:set|fixed))',
    re.IGNORECASE,
)

_SETUP_PHRASE_BOOST = 0.15


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def extract_evidence(
    document: PaperDocument,
    sections: list[SectionSpan] | None = None,
) -> list[EvidenceSnippet]:
    """Extract all simulation-relevant evidence from a parsed document.

    If sections are provided, each snippet is tagged with its section name
    and weight. Pages in the 'references' section are skipped entirely.
    """
    snippets: list[EvidenceSnippet] = []

    # Scan page text
    for page in document.pages:
        section = get_section_for_page(sections, page.page_num) if sections else None

        # Skip references entirely
        if section and section.name == "references":
            continue

        page_snippets = []
        for extractor_fn in _EXTRACTORS:
            page_snippets.extend(extractor_fn(page.text, page.page_num))

        # Tag and weight each snippet
        for snip in page_snippets:
            _apply_section_and_weight(snip, section, page.text)

        snippets.extend(page_snippets)

    # Scan captions (high-value targets — caption weight = 0.90)
    for caption in document.captions:
        for extractor_fn in _EXTRACTORS:
            for snip in extractor_fn(caption.caption, caption.page_num):
                snip.section = "caption"
                snip.section_weight = 0.90
                snip.confidence = min(snip.confidence * 0.90 + 0.10, 1.0)
                snippets.append(snip)

    return _deduplicate(snippets)


def _apply_section_and_weight(
    snippet: EvidenceSnippet,
    section: SectionSpan | None,
    page_text: str,
) -> None:
    """Tag snippet with section info and adjust confidence."""
    if section:
        snippet.section = section.name
        snippet.section_weight = section.weight
        snippet.confidence = snippet.confidence * section.weight
    else:
        snippet.section = "unknown"
        snippet.section_weight = 0.5

    # Boost if near a setup-style phrase (paper's own experiments)
    if _SETUP_PHRASES.search(snippet.source_text):
        snippet.confidence = min(snippet.confidence + _SETUP_PHRASE_BOOST, 1.0)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def _get_context(text: str, match: re.Match, window: int = 200) -> str:
    """Extract a context window around a regex match, snapping to word boundaries."""
    start = max(0, match.start() - window // 2)
    end = min(len(text), match.end() + window // 2)
    # Snap start forward to the nearest word boundary (avoid clipping mid-word)
    if start > 0:
        space = text.rfind(' ', max(0, start - 20), start)
        if space != -1:
            start = space + 1
    # Snap end to nearest word boundary
    if end < len(text):
        space = text.find(' ', end, min(len(text), end + 20))
        if space != -1:
            end = space
    return text[start:end].strip()


def _deduplicate(snippets: list[EvidenceSnippet]) -> list[EvidenceSnippet]:
    """Remove duplicates by (kind, value_text, page_num).

    Prefer higher section_weight first (methods > caption > intro),
    then higher confidence as tiebreaker. This ensures a methods-sourced
    snippet isn't lost to a caption duplicate with a marginal confidence edge.
    """
    seen: dict[tuple, EvidenceSnippet] = {}
    for s in snippets:
        key = (s.kind, s.value_text.strip(), s.page_num)
        if key not in seen:
            seen[key] = s
        else:
            existing = seen[key]
            # Prefer higher section_weight, then confidence
            if (s.section_weight, s.confidence) > (existing.section_weight, existing.confidence):
                seen[key] = s
    return sorted(seen.values(), key=lambda s: (s.page_num, s.kind))


def _is_likely_year(value: str) -> bool:
    """Reject temperature matches that are actually year numbers."""
    nums = re.findall(r'\d{4}', value)
    return any(1900 <= int(n) <= 2030 for n in nums)
