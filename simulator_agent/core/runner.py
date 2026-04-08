"""Run simulations by dispatching on reactor type.

Each reactor type has a dedicated setup function that configures
the Cantera reactor network and integrates it.

Returns raw time-series data; observable extraction is separate.

Reactor type → Cantera class mapping:
    shock_tube    → IdealGasReactor          (const-UV, adiabatic)
    rcm           → IdealGasReactor          (const-UV, adiabatic, post-compression)
    jsr           → IdealGasReactor + inlet/outlet (const-TP via energy='off', steady state)
    flow_reactor  → IdealGasConstPressureReactor  (const-P, isothermal, time ≈ residence)
    flame         → FreeFlame                (1-D premixed, laminar flame speed)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.simulation_spec import SimulationSpec

try:
    import cantera as ct
    import numpy as np
    HAS_CANTERA = True
except ImportError:
    ct = None
    np = None
    HAS_CANTERA = False

# Solver tolerances (matching T3 defaults)
_ATOL = 1e-16
_RTOL = 1e-8


@dataclass
class SimulationResult:
    """Raw output from a single simulation run."""
    experiment_id: str
    mechanism: str
    times: list[float] = field(default_factory=list)
    temperature_history: list[float] = field(default_factory=list)
    pressure_history: list[float] = field(default_factory=list)
    species_histories: dict[str, list[float]] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: str | None = None


def run_simulation(spec: SimulationSpec, mechanism_file: str) -> SimulationResult:
    """Run a simulation for the given spec and mechanism.

    Dispatches to the appropriate reactor setup based on spec.reactor_type.
    """
    if not HAS_CANTERA:
        raise ImportError("Cantera is required. Install with: conda install -c cantera cantera")

    errors = spec.validate()
    if errors:
        return SimulationResult(
            experiment_id=spec.experiment_id,
            mechanism=mechanism_file,
            success=False,
            error=f"Invalid spec: {'; '.join(errors)}",
        )

    dispatch = {
        "shock_tube": _run_shock_tube,
        "jsr": _run_jsr,
        "flow_reactor": _run_flow_reactor,
        "rcm": _run_rcm,
        "flame": _run_flame,
    }

    runner_fn = dispatch.get(spec.reactor_type)
    if runner_fn is None:
        return SimulationResult(
            experiment_id=spec.experiment_id,
            mechanism=mechanism_file,
            success=False,
            error=f"Unsupported reactor type: {spec.reactor_type}",
        )

    try:
        return runner_fn(spec, mechanism_file)
    except Exception as e:
        return SimulationResult(
            experiment_id=spec.experiment_id,
            mechanism=mechanism_file,
            success=False,
            error=f"{type(e).__name__}: {e}",
        )


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_gas(spec: SimulationSpec, mechanism_file: str):
    """Create and configure a Cantera gas phase object."""
    gas = ct.Solution(mechanism_file)
    comp_str = ", ".join(f"{sp}: {frac}" for sp, frac in spec.composition.items())
    gas.TPX = spec.temperature, spec.pressure, comp_str
    return gas


def _tracked_species(spec: SimulationSpec) -> set[str]:
    """Species to record: composition keys + observable_species if set."""
    tracked = set(spec.composition.keys())
    if spec.observable_species:
        tracked.add(spec.observable_species)
    return tracked


def _record_state(reactor, gas, tracked_species: set[str],
                  times, temps, pressures, species_hists, t: float):
    """Append current reactor state to history lists."""
    times.append(t)
    temps.append(reactor.T)
    pressures.append(reactor.phase.P)
    for sp in tracked_species:
        idx = gas.species_index(sp)
        species_hists[sp].append(reactor.phase.X[idx])


# ── Reactor implementations ─────────────────────────────────────────────


def _run_shock_tube(spec: SimulationSpec, mechanism_file: str) -> SimulationResult:
    """Constant-volume adiabatic reactor (reflected shock).

    Uses IdealGasReactor (const-UV): volume fixed, energy equation ON.
    Pressure and temperature evolve due to chemistry.
    """
    gas = _make_gas(spec, mechanism_file)
    reactor = ct.IdealGasReactor(gas, clone=True)
    sim = ct.ReactorNet([reactor])
    sim.atol = _ATOL
    sim.rtol = _RTOL

    tracked = _tracked_species(spec)
    times, temps, pressures = [], [], []
    species = {sp: [] for sp in tracked}

    # Record initial state (t=0)
    _record_state(reactor, gas, tracked, times, temps, pressures, species, 0.0)

    while sim.time < spec.end_time:
        sim.step()
        _record_state(reactor, gas, tracked, times, temps, pressures, species, sim.time)

    return SimulationResult(
        experiment_id=spec.experiment_id,
        mechanism=mechanism_file,
        times=times,
        temperature_history=temps,
        pressure_history=pressures,
        species_histories=species,
    )


def _run_rcm(spec: SimulationSpec, mechanism_file: str) -> SimulationResult:
    """Rapid compression machine: constant-volume adiabatic (post-compression).

    Same physics as shock tube (const-UV, piston locked after compression).
    Initial T and P represent post-compression conditions.
    """
    gas = _make_gas(spec, mechanism_file)
    reactor = ct.IdealGasReactor(gas, clone=True)
    sim = ct.ReactorNet([reactor])
    sim.atol = _ATOL
    sim.rtol = _RTOL

    tracked = _tracked_species(spec)
    times, temps, pressures = [], [], []
    species = {sp: [] for sp in tracked}

    _record_state(reactor, gas, tracked, times, temps, pressures, species, 0.0)

    while sim.time < spec.end_time:
        sim.step()
        _record_state(reactor, gas, tracked, times, temps, pressures, species, sim.time)

    return SimulationResult(
        experiment_id=spec.experiment_id,
        mechanism=mechanism_file,
        times=times,
        temperature_history=temps,
        pressure_history=pressures,
        species_histories=species,
    )


def _run_jsr(spec: SimulationSpec, mechanism_file: str) -> SimulationResult:
    """Jet-stirred reactor: isothermal, constant-P, steady-state.

    Models a CSTR with continuous inlet/outlet. The reactor is integrated
    for many residence times until steady state is reached.
    """
    gas = _make_gas(spec, mechanism_file)

    # Upstream reservoir with FRESH inlet composition (separate Solution)
    inlet_gas = ct.Solution(mechanism_file)
    comp_str = ", ".join(f"{sp}: {frac}" for sp, frac in spec.composition.items())
    inlet_gas.TPX = spec.temperature, spec.pressure, comp_str
    upstream = ct.Reservoir(inlet_gas, clone=True)

    # Downstream exhaust reservoir
    exhaust_gas = ct.Solution(mechanism_file)
    exhaust_gas.TPX = spec.temperature, spec.pressure, comp_str
    downstream = ct.Reservoir(exhaust_gas, clone=True)

    # Reactor: isothermal (energy='off'), constant-P via outlet controller
    reactor = ct.IdealGasReactor(gas, energy="off", clone=True)

    tau = spec.residence_time or 1.0
    volume = reactor.volume
    mass_flow = inlet_gas.density * volume / tau

    mfc = ct.MassFlowController(upstream, reactor, mdot=mass_flow)
    ct.PressureController(reactor, downstream, primary=mfc, K=1e-3)

    sim = ct.ReactorNet([reactor])
    sim.atol = _ATOL
    sim.rtol = _RTOL

    # Integrate to steady state (~50 residence times)
    sim.advance(tau * 50)

    tracked = _tracked_species(spec)
    species = {}
    for sp in tracked:
        idx = gas.species_index(sp)
        species[sp] = [reactor.phase.X[idx]]

    return SimulationResult(
        experiment_id=spec.experiment_id,
        mechanism=mechanism_file,
        times=[sim.time],
        temperature_history=[reactor.T],
        pressure_history=[reactor.phase.P],
        species_histories=species,
        extra={"residence_time": tau, "steady_state": True},
    )


def _run_flow_reactor(spec: SimulationSpec, mechanism_file: str) -> SimulationResult:
    """Plug flow reactor: constant-P, isothermal, time axis = residence time.

    Uses IdealGasConstPressureReactor(energy='off'): pressure held constant,
    temperature held constant. Time integration maps to axial position via
    the velocity profile.
    """
    gas = _make_gas(spec, mechanism_file)
    reactor = ct.IdealGasConstPressureReactor(gas, energy="off", clone=True)
    sim = ct.ReactorNet([reactor])
    sim.atol = _ATOL
    sim.rtol = _RTOL

    tau = spec.residence_time or spec.end_time
    tracked = _tracked_species(spec)
    times, temps, pressures = [], [], []
    species = {sp: [] for sp in tracked}

    # Record initial state
    _record_state(reactor, gas, tracked, times, temps, pressures, species, 0.0)

    n_steps = 200
    dt = tau / n_steps
    for i in range(1, n_steps + 1):
        t = i * dt
        sim.advance(t)
        _record_state(reactor, gas, tracked, times, temps, pressures, species, sim.time)

    return SimulationResult(
        experiment_id=spec.experiment_id,
        mechanism=mechanism_file,
        times=times,
        temperature_history=temps,
        pressure_history=pressures,
        species_histories=species,
    )


def _run_flame(spec: SimulationSpec, mechanism_file: str) -> SimulationResult:
    """Freely propagating premixed flame → laminar flame speed.

    Uses FreeFlame: 1-D simulation with adaptive grid refinement.
    Returns flame speed in m/s via extra['flame_speed'].
    """
    gas = _make_gas(spec, mechanism_file)

    width = 0.03  # m, domain width
    flame = ct.FreeFlame(gas, width=width)
    flame.set_refine_criteria(ratio=3, slope=0.07, curve=0.14)
    flame.solve(loglevel=0, auto=True)

    flame_speed = flame.velocity[0]

    return SimulationResult(
        experiment_id=spec.experiment_id,
        mechanism=mechanism_file,
        times=[0.0],
        temperature_history=[flame.T[-1]],
        pressure_history=[spec.pressure],
        species_histories={},
        extra={"flame_speed": flame_speed},
    )
