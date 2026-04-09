"""Tests for migrated core modules (no Cantera dependency)."""

from src.schemas.experimental import SimConditions, ReactorType, ObservableType
from src.simulation.core.runner import _apply_aliases

import pytest

from src.simulation.core.simulation_spec import SimulationSpec
from src.simulation.core.mae import compute_mae, compute_relative_mae
from src.simulation.core.observable_extractor import extract_observable
from src.simulation.core.runner import SimulationResult
from src.simulation.core.model_loader import MechanismInfo


# -- SimulationSpec -----------------------------------------------------------

def test_spec_valid():
    spec = SimulationSpec(
        experiment_id="test1",
        reactor_type="shock_tube",
        observable="ignition_delay",
        temperature=1200.0,
        pressure=101325.0,
        composition={"NH3": 0.01, "Ar": 0.99},
    )
    assert spec.validate() == []


def test_spec_missing_temperature():
    spec = SimulationSpec(
        experiment_id="test1",
        reactor_type="shock_tube",
        observable="ignition_delay",
        pressure=101325.0,
        composition={"NH3": 0.01},
    )
    errors = spec.validate()
    assert any("temperature" in e for e in errors)


def test_spec_missing_composition():
    spec = SimulationSpec(
        experiment_id="test1",
        reactor_type="shock_tube",
        observable="ignition_delay",
        temperature=1200.0,
        pressure=101325.0,
    )
    errors = spec.validate()
    assert any("composition" in e for e in errors)


def test_spec_with_temperature_list():
    spec = SimulationSpec(
        experiment_id="test1",
        reactor_type="shock_tube",
        observable="ignition_delay",
        pressure=101325.0,
        composition={"NH3": 0.01, "Ar": 0.99},
        temperature_list=[1000, 1200, 1400],
    )
    assert spec.validate() == []


# -- MAE ----------------------------------------------------------------------

def test_mae_exact():
    assert compute_mae([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0


def test_mae_offset():
    assert compute_mae([2.0, 3.0], [1.0, 2.0]) == 1.0


def test_mae_length_mismatch():
    with pytest.raises(ValueError, match="Length"):
        compute_mae([1.0], [1.0, 2.0])


def test_mae_empty():
    with pytest.raises(ValueError, match="empty"):
        compute_mae([], [])


def test_relative_mae():
    result = compute_relative_mae([2.0, 4.0], [1.0, 2.0])
    assert abs(result - 1.0) < 1e-9  # |2-1|/|1| + |4-2|/|2| = 1 + 1 = 2/2 = 1


def test_relative_mae_skips_zero():
    result = compute_relative_mae([1.0, 2.0], [0.0, 2.0])
    assert result == 0.0  # only the second point counts, |2-2|/|2| = 0


# -- Observable extractor -----------------------------------------------------

def test_extract_ignition_delay():
    spec = SimulationSpec(
        experiment_id="t1", reactor_type="shock_tube",
        observable="ignition_delay",
        temperature=1200, pressure=101325,
        composition={"NH3": 0.01},
    )
    # Simulate clear ignition: 1200 K -> 2800 K (T_rise=1600 K > 400 K threshold)
    result = SimulationResult(
        experiment_id="t1", mechanism="test.yaml",
        times=[0.0, 0.001, 0.002, 0.003, 0.004, 0.005],
        temperature_history=[1200, 1200, 1250, 1800, 2600, 2800],
    )
    values = extract_observable(spec, result)
    assert len(values) == 1
    assert values[0] == 0.004  # steepest rise: (2600-1800)/dt at t=0.004


def test_extract_ignition_delay_no_ignition():
    spec = SimulationSpec(
        experiment_id="t1", reactor_type="shock_tube",
        observable="ignition_delay",
        temperature=1200, pressure=101325,
        composition={"NH3": 0.01},
    )
    # No ignition: T rise < 2% of T0 (24 K). Only rises 3 K.
    result = SimulationResult(
        experiment_id="t1", mechanism="test.yaml",
        times=[0.0, 0.001, 0.002, 0.003],
        temperature_history=[1200, 1201, 1202, 1203],
    )
    values = extract_observable(spec, result)
    assert len(values) == 1
    assert values[0] == 0.003  # returns end_time as sentinel


def test_extract_species_profile():
    spec = SimulationSpec(
        experiment_id="t1", reactor_type="shock_tube",
        observable="species_profile", observable_species="NH3",
        temperature=1200, pressure=101325,
        composition={"NH3": 0.01},
    )
    result = SimulationResult(
        experiment_id="t1", mechanism="test.yaml",
        times=[0.0, 0.001],
        temperature_history=[1200, 1300],
        species_histories={"NH3": [0.01, 0.005]},
    )
    values = extract_observable(spec, result)
    assert values == [0.01, 0.005]


def test_extract_flame_speed():
    spec = SimulationSpec(
        experiment_id="t1", reactor_type="flame",
        observable="flame_speed",
        temperature=300, pressure=101325,
        composition={"CH4": 0.095},
    )
    result = SimulationResult(
        experiment_id="t1", mechanism="test.yaml",
        extra={"flame_speed": 0.37},
    )
    values = extract_observable(spec, result)
    assert values == [0.37]


def test_extract_half_life_interpolated():
    spec = SimulationSpec(
        experiment_id="t1", reactor_type="shock_tube",
        observable="half_life", observable_species="NH3",
        temperature=1200, pressure=101325,
        composition={"NH3": 0.01},
    )
    # NH3 goes from 0.01 -> 0.006 -> 0.004 (half = 0.005)
    # Interpolation: 0.005 is midway between 0.006 and 0.004,
    # so t_half = 0.001 + 0.5 * (0.002 - 0.001) = 0.0015
    result = SimulationResult(
        experiment_id="t1", mechanism="test.yaml",
        times=[0.0, 0.001, 0.002],
        temperature_history=[1200, 1400, 1600],
        species_histories={"NH3": [0.01, 0.006, 0.004]},
    )
    values = extract_observable(spec, result)
    assert len(values) == 1
    assert abs(values[0] - 0.0015) < 1e-10


def test_extract_failed_result():
    spec = SimulationSpec(
        experiment_id="t1", reactor_type="shock_tube",
        observable="ignition_delay",
        temperature=1200, pressure=101325,
        composition={"NH3": 0.01},
    )
    result = SimulationResult(
        experiment_id="t1", mechanism="test.yaml",
        success=False, error="boom",
    )
    assert extract_observable(spec, result) == []


# -- MechanismInfo (dataclass smoke test) -------------------------------------

def test_mechanism_info():
    info = MechanismInfo(path="/tmp/mech.yaml", name="test_mech", n_species=53, n_reactions=325)
    assert info.name == "test_mech"
    assert info.n_species == 53


# -- Species name preservation ------------------------------------------------

def test_species_names_with_rmg_indices_preserved():
    """SimConditions with RMG-style names like NH3(1) must pass through to
    SimulationSpec.composition without stripping the index suffix."""
    cond = SimConditions(
        reactor_type=ReactorType.SHOCK_TUBE,
        T=1200.0,
        P=1.0,
        X={"NH3(1)": 0.005, "Ar": 0.995},
        observable_type=ObservableType.IDT,
    )
    # Verify dict keys are exactly as provided
    assert "NH3(1)" in cond.X
    assert "Ar" in cond.X
    assert "NH3" not in cond.X

    # Verify flow through _to_spec
    from src.pipelines.path1 import _to_spec
    spec = _to_spec(cond, 0, "/tmp/mech.yaml")
    assert "NH3(1)" in spec.composition
    assert "Ar" in spec.composition
    assert "NH3" not in spec.composition


# -- Species alias mapping ----------------------------------------------------

def test_apply_aliases_remaps_species():
    """_apply_aliases remaps keys present in the alias dict."""
    composition = {"NH3": 0.005, "O2": 0.21, "Ar": 0.785}
    aliases = {"NH3": "NH3(1)", "O2": "O2(6)"}
    result = _apply_aliases(composition, aliases)
    assert result == {"NH3(1)": 0.005, "O2(6)": 0.21, "Ar": 0.785}


def test_apply_aliases_unknown_species_pass_through():
    """Species not in aliases pass through unchanged."""
    composition = {"NH3": 0.005, "Kr": 0.995}
    aliases = {"NH3": "NH3(1)"}
    result = _apply_aliases(composition, aliases)
    assert result == {"NH3(1)": 0.005, "Kr": 0.995}


def test_apply_aliases_empty_aliases_is_identity():
    """Empty aliases dict returns composition unchanged."""
    composition = {"NH3": 0.005, "Ar": 0.995}
    result = _apply_aliases(composition, {})
    assert result == composition


def test_observable_species_remapped_by_run_simulation():
    """run_simulation applies aliases to observable_species when present."""
    spec = SimulationSpec(
        experiment_id="test_alias",
        reactor_type="shock_tube",
        observable="species_profile",
        observable_species="NH3",
        temperature=1200.0,
        pressure=101325.0,
        composition={"NH3": 0.005, "Ar": 0.995},
    )
    aliases = {"NH3": "NH3(1)"}

    # We can't call run_simulation without Cantera, so test the alias logic
    # by importing and applying the same mapping run_simulation does
    updates: dict = {"composition": _apply_aliases(spec.composition, aliases)}
    if spec.observable_species and spec.observable_species in aliases:
        updates["observable_species"] = aliases[spec.observable_species]
    mapped = spec.model_copy(update=updates)

    assert mapped.composition == {"NH3(1)": 0.005, "Ar": 0.995}
    assert mapped.observable_species == "NH3(1)"
