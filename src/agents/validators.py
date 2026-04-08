"""Model isolation and branch validation. No LLM calls — pure deterministic logic."""

from __future__ import annotations

import re
from pathlib import Path

try:
    import cantera as ct
except ImportError:
    ct = None  # type: ignore[assignment]


class ModelIsolationViolation(Exception):
    """Raised when Path 1 literature model shares reactions with the original."""

    def __init__(self, overlapping: set[str]) -> None:
        self.overlapping = overlapping
        joined = "\n  ".join(sorted(overlapping))
        super().__init__(
            f"Model isolation violated — {len(overlapping)} shared reaction(s):\n  {joined}"
        )


_ARROW_RE = re.compile(r"\s*(<?=>?|=)\s*")


def _normalize_equation(eq: str) -> str:
    """Normalize a reaction equation string for comparison.

    - Lowercase
    - Split on arrow (<=>, =>, =)
    - Sort species alphabetically on each side
    - Rejoin with ' <=> ' (all arrows normalized to reversible form)
    """
    parts = _ARROW_RE.split(eq, maxsplit=1)
    if len(parts) < 3:
        return eq.strip().lower()

    reactants_str, _arrow, products_str = parts[0], parts[1], parts[2]

    reactants = sorted(s.strip().lower() for s in reactants_str.split("+") if s.strip())
    products = sorted(s.strip().lower() for s in products_str.split("+") if s.strip())

    return " + ".join(reactants) + " <=> " + " + ".join(products)


def _load_reactions(model_path: Path) -> list[str]:
    """Load a Cantera YAML mechanism and return its raw equation strings."""
    # Resolve cantera at call time so test-injected sys.modules fakes work
    import sys
    _ct = sys.modules.get("cantera", ct)
    if _ct is None:
        raise ImportError("Cantera is required for model validation")
    try:
        sol = _ct.Solution(str(model_path))
    except Exception as exc:
        raise ValueError(f"Failed to load mechanism: {model_path}") from exc
    return [r.equation for r in sol.reactions()]


class ModelIsolationValidator:
    """Validates model isolation (Path 1) and branch superset (Path 2) constraints."""

    def reaction_ids(self, model_path: Path) -> set[str]:
        """Return the set of normalized reaction strings from a mechanism file."""
        raw_eqs = _load_reactions(model_path)
        return {_normalize_equation(eq) for eq in raw_eqs}

    def validate_path1(
        self,
        literature_model: Path,
        original_model: Path,
    ) -> None:
        """Verify that literature and original models share NO reactions.

        Raises ModelIsolationViolation if any overlap is found.
        Raises ValueError if either file cannot be loaded.
        """
        lit_ids = self.reaction_ids(literature_model)
        orig_ids = self.reaction_ids(original_model)

        overlap = lit_ids & orig_ids
        if overlap:
            raise ModelIsolationViolation(overlap)

    def validate_path2_branch(
        self,
        branch_model: Path,
        original_model: Path,
    ) -> None:
        """Verify that branch_model is a superset of original_model.

        Every reaction in the original must be present in the branch.
        Raises ValueError if any original reaction is missing.
        """
        branch_ids = self.reaction_ids(branch_model)
        orig_ids = self.reaction_ids(original_model)

        missing = orig_ids - branch_ids
        if missing:
            joined = "\n  ".join(sorted(missing))
            raise ValueError(
                f"Branch model is missing {len(missing)} original reaction(s):\n  {joined}"
            )
