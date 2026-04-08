"""Experiment family registry.

Central lookup for family-specific rules. All family knowledge
lives here, not scattered across extractors and builders.

Usage:
    from src.simulation.reactors import get_family, get_all_families, detect_family

    rules = get_family("shock_tube")
    rules.temperature_plausible(2500.0)  # True
    rules.detect_observable("species profile at 2400 K")  # "species_profile"
"""

from __future__ import annotations

from .base import FamilyRules
from .shock_tube import shock_tube
from .jsr import jsr
from .pfr import flow_reactor
from .rcm import rcm
from .flame import flame


_REGISTRY: dict[str, FamilyRules] = {
    "shock_tube": shock_tube,
    "jsr": jsr,
    "flow_reactor": flow_reactor,
    "rcm": rcm,
    "flame": flame,
}

# Fallback rules for unknown families
_UNKNOWN = FamilyRules(
    name="unknown",
    anchor_patterns=[],
    observable_patterns={},
    temperature_bounds=(200.0, 5000.0),
    required_conditions=["temperature", "pressure", "composition"],
    template_family="unsupported",
    default_pressure=None,
)


def get_family(name: str) -> FamilyRules:
    """Look up rules for a family by name. Returns unknown fallback if not found."""
    return _REGISTRY.get(name, _UNKNOWN)


def get_all_families() -> dict[str, FamilyRules]:
    """Return all registered families."""
    return dict(_REGISTRY)


def detect_family(text: str) -> str | None:
    """Detect experiment family from text. Returns family name or None."""
    for name, rules in _REGISTRY.items():
        if rules.matches_text(text):
            return name
    return None
