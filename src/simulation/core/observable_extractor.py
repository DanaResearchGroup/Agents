"""Extract observables from raw simulation results.

Maps each observable type to an extraction function that
processes SimulationResult into comparable values.
"""

from __future__ import annotations

from src.simulation.core.simulation_spec import SimulationSpec
from src.simulation.core.runner import SimulationResult


def extract_observable(spec: SimulationSpec, result: SimulationResult) -> list[float]:
    """Extract the requested observable from simulation results.

    Returns a list of values corresponding to the experimental data points.
    For single-point experiments, returns a one-element list.
    """
    if not result.success:
        return []

    dispatch = {
        "ignition_delay": _extract_ignition_delay,
        "species_profile": _extract_species_profile,
        "flame_speed": _extract_flame_speed,
        "half_life": _extract_half_life,
        "conversion": _extract_conversion,
        "temperature_profile": _extract_temperature_profile,
    }

    fn = dispatch.get(spec.observable)
    if fn is None:
        return []

    return fn(spec, result)


def _extract_ignition_delay(spec: SimulationSpec, result: SimulationResult) -> list[float]:
    """Extract ignition delay time from temperature history.

    Defined as time of maximum dT/dt (matching T3/RMG convention).
    Returns time at max dT/dt if temperature rise exceeds 2% of T0.
    Returns end_time if no meaningful temperature rise is detected
    (consistent with "IDT > end_time" convention in literature).
    """
    if len(result.times) < 3:
        return []

    times = result.times
    temps = result.temperature_history

    # Compute dT/dt via forward differences
    max_dtdt = 0.0
    idt_idx = 0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        if dt <= 0:
            continue
        dtdt = (temps[i] - temps[i - 1]) / dt
        if dtdt > max_dtdt:
            max_dtdt = dtdt
            idt_idx = i

    # Sanity check: temperature must rise by at least 2% of T0 to indicate
    # that meaningful chemistry occurred. For dilute mixtures this can be
    # as little as 20-30 K. If not met, report end_time as the IDT
    # (consistent with "IDT > simulation window" convention).
    T0 = temps[0] if temps[0] > 0 else 1.0
    T_rise = max(temps) - T0
    if T_rise < 0.02 * T0:
        return [times[-1]]

    return [times[idt_idx]]


def _extract_species_profile(spec: SimulationSpec, result: SimulationResult) -> list[float]:
    """Extract species mole fraction time series."""
    sp = spec.observable_species
    if not sp or sp not in result.species_histories:
        return []
    return list(result.species_histories[sp])


def _extract_flame_speed(spec: SimulationSpec, result: SimulationResult) -> list[float]:
    """Extract laminar flame speed from flame simulation."""
    fs = result.extra.get("flame_speed")
    if fs is None:
        return []
    return [fs]


def _extract_half_life(spec: SimulationSpec, result: SimulationResult) -> list[float]:
    """Extract half-life: time for target species to reach 50% of initial.

    Uses linear interpolation between the last point above and first
    point below the half-value for sub-timestep accuracy.
    """
    sp = spec.observable_species
    if not sp or sp not in result.species_histories:
        return []

    profile = result.species_histories[sp]
    if len(profile) < 2:
        return []

    initial = profile[0]
    if initial <= 0:
        return []

    half = initial / 2.0

    for i in range(1, len(profile)):
        if profile[i] <= half:
            # Linear interpolation between points i-1 and i
            x_above = profile[i - 1]
            x_below = profile[i]
            t_above = result.times[i - 1]
            t_below = result.times[i]

            if x_above == x_below:
                return [t_below]

            frac = (x_above - half) / (x_above - x_below)
            t_half = t_above + frac * (t_below - t_above)
            return [t_half]

    # Species never reached half -- return end time as sentinel
    return [result.times[-1]]


def _extract_conversion(spec: SimulationSpec, result: SimulationResult) -> list[float]:
    """Extract fuel conversion: (X_initial - X_final) / X_initial."""
    sp = spec.observable_species
    if not sp or sp not in result.species_histories:
        return []

    profile = result.species_histories[sp]
    if not profile or profile[0] <= 0:
        return []

    conversion = (profile[0] - profile[-1]) / profile[0]
    return [conversion]


def _extract_temperature_profile(spec: SimulationSpec, result: SimulationResult) -> list[float]:
    """Extract temperature time series."""
    return list(result.temperature_history)
