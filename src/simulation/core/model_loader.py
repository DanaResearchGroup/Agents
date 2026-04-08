"""Load and validate Cantera mechanism YAML files.

This module handles loading one or more mechanism files and
attaching them to SimulationSpec objects for evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

try:
    import cantera as ct
    HAS_CANTERA = True
except ImportError:
    ct = None
    HAS_CANTERA = False


@dataclass
class MechanismInfo:
    """Metadata about a loaded mechanism."""
    path: str
    name: str           # user-facing label (filename stem or explicit name)
    n_species: int
    n_reactions: int


def load_mechanism(path: str | Path, name: str | None = None) -> MechanismInfo:
    """Load a Cantera mechanism YAML and return its metadata.

    Accepts both absolute file paths and Cantera built-in mechanism names
    (e.g. 'gri30.yaml', 'h2o2.yaml') which Cantera resolves via its data dirs.

    Raises ImportError if Cantera is not installed.
    """
    if not HAS_CANTERA:
        raise ImportError("Cantera is required for mechanism loading. Install with: conda install -c cantera cantera")

    path_str = str(path)
    gas = ct.Solution(path_str)
    label = name or Path(path_str).stem

    return MechanismInfo(
        path=path_str,
        name=label,
        n_species=gas.n_species,
        n_reactions=gas.n_reactions,
    )
