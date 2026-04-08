"""Normalized simulation specification.

A SimulationSpec is the canonical internal representation of
"what to simulate". It is produced by exp_loader (from YAML)
or optionally from literature_support (from PDF-derived plans).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


ReactorType = Literal[
    "shock_tube",
    "jsr",
    "flow_reactor",
    "rcm",
    "flame",
]

Observable = Literal[
    "ignition_delay",
    "species_profile",
    "conversion",
    "temperature_profile",
    "flame_speed",
    "half_life",
]


@dataclass
class SimulationSpec:
    """Fully normalized specification for one simulation run.

    All values are in SI units:
      temperature: K
      pressure:    Pa
      time:        s
      composition: mole fractions (must sum to 1.0)
    """

    experiment_id: str
    reactor_type: ReactorType
    observable: Observable
    observable_species: str | None = None  # e.g. "NH3" for species_profile

    # Conditions
    temperature: float = 0.0               # K
    pressure: float = 0.0                  # Pa
    composition: dict[str, float] = field(default_factory=dict)  # species -> mole fraction
    equivalence_ratio: float | None = None
    residence_time: float | None = None    # s (JSR, flow reactor)

    # Sweep support: if set, override the scalar above
    temperature_list: list[float] | None = None  # K
    pressure_list: list[float] | None = None     # Pa

    # Simulation control
    end_time: float = 1.0                  # s (max simulation time)
    mechanism_file: str | None = None      # path to Cantera YAML

    def validate(self) -> list[str]:
        """Return list of error strings. Empty means valid."""
        errors = []
        if not self.experiment_id:
            errors.append("experiment_id is required")
        if self.temperature <= 0 and not self.temperature_list:
            errors.append("temperature must be positive (or provide temperature_list)")
        if self.pressure <= 0 and not self.pressure_list:
            errors.append("pressure must be positive (or provide pressure_list)")
        if not self.composition:
            errors.append("composition is required")
        return errors
