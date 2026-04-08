"""Load experimental YAML into SimulationSpec objects.

Expected YAML schema:

    experiments:
      - id: exp_001
        reactor_type: shock_tube
        observable: ignition_delay
        temperature: 1200          # K
        pressure: 101325           # Pa  (or use pressure_atm)
        composition:
          NH3: 0.01
          O2: 0.01
          Ar: 0.98
        end_time: 0.1              # s  (optional)
        data:                      # experimental measurements
          - temperature: 1200
            value: 0.00045
          - temperature: 1400
            value: 0.00012
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from core.simulation_spec import SimulationSpec


# Pressure unit conversions to Pa
_PRESSURE_CONVERSIONS = {
    "pa": 1.0,
    "kpa": 1e3,
    "mpa": 1e6,
    "bar": 1e5,
    "atm": 101325.0,
    "torr": 133.322,
    "psi": 6894.76,
}


@dataclass
class ExperimentalDataPoint:
    """One experimental measurement."""
    conditions: dict[str, float]   # e.g. {"temperature": 1200}
    value: float
    uncertainty: float | None = None


@dataclass
class ExperimentalDataset:
    """An experiment definition with its measured data."""
    spec: SimulationSpec
    data: list[ExperimentalDataPoint]
    metadata: dict[str, Any]


def load_experiment_yaml(path: str | Path) -> list[ExperimentalDataset]:
    """Load experimental YAML and return a list of ExperimentalDataset."""
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict) or "experiments" not in raw:
        raise ValueError(f"YAML must contain top-level 'experiments' key: {path}")

    global_meta = {k: v for k, v in raw.items() if k != "experiments"}
    datasets = []

    for entry in raw["experiments"]:
        spec = _build_spec(entry)
        data = _build_data(entry.get("data", []))
        meta = {**global_meta, **{k: v for k, v in entry.items()
                if k not in ("id", "reactor_type", "observable", "observable_species",
                             "temperature", "pressure", "pressure_atm", "pressure_bar",
                             "composition", "equivalence_ratio", "residence_time",
                             "end_time", "data", "temperature_list", "pressure_list")}}
        datasets.append(ExperimentalDataset(spec=spec, data=data, metadata=meta))

    return datasets


def _build_spec(entry: dict) -> SimulationSpec:
    """Build a SimulationSpec from a single experiment entry."""
    pressure = _resolve_pressure(entry)

    return SimulationSpec(
        experiment_id=str(entry["id"]),
        reactor_type=entry["reactor_type"],
        observable=entry["observable"],
        observable_species=entry.get("observable_species"),
        temperature=float(entry.get("temperature", 0)),
        pressure=pressure,
        composition=entry.get("composition", {}),
        equivalence_ratio=entry.get("equivalence_ratio"),
        residence_time=entry.get("residence_time"),
        temperature_list=entry.get("temperature_list"),
        pressure_list=entry.get("pressure_list"),
        end_time=float(entry.get("end_time", 1.0)),
    )


def _resolve_pressure(entry: dict) -> float:
    """Resolve pressure from various unit fields to Pa."""
    if "pressure" in entry:
        return float(entry["pressure"])

    # Check for unit-suffixed pressure fields: pressure_atm, pressure_bar, etc.
    for suffix, factor in _PRESSURE_CONVERSIONS.items():
        key = f"pressure_{suffix}"
        if key in entry:
            return float(entry[key]) * factor

    return 0.0


def _build_data(raw_data: list[dict]) -> list[ExperimentalDataPoint]:
    """Parse the data array into ExperimentalDataPoint objects."""
    points = []
    for item in raw_data:
        value = item["value"]
        uncertainty = item.get("uncertainty")
        conditions = {k: float(v) for k, v in item.items()
                      if k not in ("value", "uncertainty")}
        points.append(ExperimentalDataPoint(
            conditions=conditions,
            value=float(value),
            uncertainty=float(uncertainty) if uncertainty is not None else None,
        ))
    return points
