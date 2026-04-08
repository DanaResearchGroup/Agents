"""Base class for experiment family rules.

Each experiment family (shock tube, JSR, etc.) defines its own:
- Anchor patterns (how to detect this family in text)
- Observable patterns (what measurements are typical)
- Condition requirements (what's needed to simulate)
- Temperature bounds (plausible range for validation)
- Template mapping (which simulation template to use)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class FamilyRules:
    """Domain knowledge for one experiment family."""

    name: str

    # -- Detection --------------------------------------------------------
    # Patterns that identify this family in text
    anchor_patterns: list[re.Pattern]
    # Patterns that identify observables typical of this family
    observable_patterns: dict[str, re.Pattern]

    # -- Validation -------------------------------------------------------
    # Plausible temperature range (K) for this family
    temperature_bounds: tuple[float, float]
    # Conditions required for simulation
    required_conditions: list[str]

    # -- Planning ---------------------------------------------------------
    # Which simulation template to use
    template_family: str
    # Default pressure assumption when not stated (e.g., "1 atm" for shock tubes)
    default_pressure: str | None = None

    def matches_text(self, text: str) -> bool:
        """Check if any anchor pattern matches the text."""
        return any(p.search(text) for p in self.anchor_patterns)

    def detect_observable(self, text: str) -> str | None:
        """Detect the most likely observable from text."""
        for obs_type, pattern in self.observable_patterns.items():
            if pattern.search(text):
                return obs_type
        return None

    def temperature_plausible(self, temp_k: float) -> bool:
        """Check if a temperature value is plausible for this family."""
        lo, hi = self.temperature_bounds
        return lo <= temp_k <= hi
