"""Export a validated simulation plan as normalized JSON.

Only accepts executable plans. Returns a clean, flat JSON structure
ready for downstream tools (RMG, Cantera, etc.).
"""

from __future__ import annotations

import json

from models import SimulationPlan


def export_plan_json(plan: SimulationPlan) -> dict:
    """Export an executable plan as a normalized JSON-ready dict.

    Raises ValueError if plan is not executable.
    """
    if plan.plan_status != "executable":
        raise ValueError(
            f"Cannot export non-executable plan (status={plan.plan_status}, "
            f"scenario={plan.scenario_id})"
        )

    result: dict = {
        "scenario_id": plan.scenario_id,
        "experiment_family": plan.experiment_family,
        "template_family": plan.template_family,
    }

    # Temperature
    if plan.temperature:
        result["temperature"] = {
            "min_K": plan.temperature.min_value,
            "max_K": plan.temperature.max_value,
            "mode": plan.temperature.mode,
        }

    # Pressure — normalize to Pa
    if plan.pressure:
        factor = _pressure_to_pa(plan.pressure.unit)
        result["pressure"] = {
            "min_Pa": plan.pressure.min_value * factor,
            "max_Pa": plan.pressure.max_value * factor,
            "mode": plan.pressure.mode,
        }

    # Composition
    if plan.composition and plan.composition.species:
        result["composition"] = plan.composition.species
        result["composition_complete"] = plan.composition.composition_complete

    # Observables
    result["target_observables"] = plan.target_observables

    # Mechanism
    if plan.mechanism:
        result["mechanism"] = plan.mechanism

    return result


def export_plan_json_string(plan: SimulationPlan) -> str:
    """Export as a formatted JSON string."""
    return json.dumps(export_plan_json(plan), indent=2)


_PRESSURE_FACTORS = {
    "atm": 101325.0,
    "bar": 1e5,
    "kpa": 1e3,
    "mpa": 1e6,
    "torr": 133.322,
    "psi": 6894.76,
    "pa": 1.0,
}


def _pressure_to_pa(unit: str) -> float:
    return _PRESSURE_FACTORS.get(unit.lower(), 101325.0)
