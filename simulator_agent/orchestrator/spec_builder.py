"""Convert between pipeline representations and SimulationSpec.

Two conversion paths:
  ExperimentalDataset → SimulationSpec  (trivial — spec already exists)
  SimulationPlan      → SimulationSpec  (unit conversion + field mapping)
"""

from __future__ import annotations

from models import SimulationPlan
from core.simulation_spec import SimulationSpec
from core.exp_loader import ExperimentalDataset


# ── Template family → reactor type mapping ──────────────────────────────

_TEMPLATE_TO_REACTOR: dict[str, str] = {
    "idt_const_uv":  "shock_tube",
    "idt_const_hp":  "rcm",
    "jsr_const_tp":  "jsr",
    "pfr_const_tp":  "flow_reactor",
    "flame_speed":   "flame",
    "batch_const_tp": "flow_reactor",
}

# ── Pressure unit → Pa conversion ───────────────────────────────────────

_PRESSURE_TO_PA: dict[str, float] = {
    "pa":   1.0,
    "kpa":  1e3,
    "mpa":  1e6,
    "bar":  1e5,
    "atm":  101325.0,
    "torr": 133.322,
    "psi":  6894.76,
}

# ── Time unit → seconds conversion ──────────────────────────────────────

_TIME_TO_S: dict[str, float] = {
    "s":  1.0,
    "ms": 1e-3,
    "us": 1e-6,
    "min": 60.0,
}


def spec_from_dataset(dataset: ExperimentalDataset) -> SimulationSpec:
    """Extract the SimulationSpec from an ExperimentalDataset.

    Trivial pass-through — the spec is already built by exp_loader.
    """
    return dataset.spec


def spec_from_plan(plan: SimulationPlan, mechanism_file: str | None = None) -> SimulationSpec:
    """Convert an executable SimulationPlan into a SimulationSpec.

    Handles unit normalization (pressure → Pa, time → s, temperature stays K).
    Only accepts executable plans. Raises ValueError otherwise.
    """
    if plan.plan_status != "executable":
        raise ValueError(
            f"Cannot build SimulationSpec from non-executable plan "
            f"(status={plan.plan_status}, scenario={plan.scenario_id})"
        )

    reactor_type = _TEMPLATE_TO_REACTOR.get(plan.template_family)
    if reactor_type is None:
        raise ValueError(f"Unknown template_family: {plan.template_family}")

    # Temperature (K — no conversion needed for K, handle C if present)
    temperature = 0.0
    temperature_list = None
    if plan.temperature:
        temperature = _to_kelvin(plan.temperature.min_value, plan.temperature.unit)
        if plan.temperature.is_range:
            t_max = _to_kelvin(plan.temperature.max_value, plan.temperature.unit)
            temperature_list = [temperature, t_max]

    # Pressure → Pa
    pressure = 0.0
    if plan.pressure:
        factor = _PRESSURE_TO_PA.get(plan.pressure.unit.lower(), 101325.0)
        pressure = plan.pressure.min_value * factor

    # Composition (already mole fractions in NormalizedComposition)
    composition: dict[str, float] = {}
    if plan.composition:
        composition = dict(plan.composition.species)

    # Residence time → seconds
    residence_time = None
    if plan.residence_time:
        factor = _TIME_TO_S.get(plan.residence_time.unit.lower(), 1.0)
        residence_time = plan.residence_time.min_value * factor

    # Observable
    observable = plan.target_observables[0] if plan.target_observables else "species_profile"

    # End time heuristic based on reactor type
    end_time = _default_end_time(reactor_type, temperature)

    return SimulationSpec(
        experiment_id=plan.scenario_id,
        reactor_type=reactor_type,
        observable=observable,
        temperature=temperature,
        pressure=pressure,
        composition=composition,
        residence_time=residence_time,
        temperature_list=temperature_list,
        end_time=end_time,
        mechanism_file=mechanism_file or plan.mechanism,
    )


def _to_kelvin(value: float, unit: str) -> float:
    """Convert temperature to Kelvin."""
    u = unit.lower().strip()
    if u in ("k", ""):
        return value
    if u in ("c", "°c", "celsius"):
        return value + 273.15
    if u in ("f", "°f", "fahrenheit"):
        return (value - 32) * 5 / 9 + 273.15
    return value  # assume K


def _default_end_time(reactor_type: str, temperature: float) -> float:
    """Reasonable default end_time based on reactor type and temperature."""
    if reactor_type == "flame":
        return 1.0  # not used for FreeFlame but needs a value
    if reactor_type == "jsr":
        return 1.0  # JSR uses residence_time, not end_time
    if reactor_type in ("shock_tube", "rcm"):
        # Higher T → shorter IDT → shorter needed simulation window
        if temperature > 1500:
            return 0.01   # 10 ms
        if temperature > 1000:
            return 0.1    # 100 ms
        return 1.0        # 1 s
    return 1.0
