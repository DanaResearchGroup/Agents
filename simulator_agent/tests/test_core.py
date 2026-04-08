"""Tests for core evaluation modules (no Cantera dependency)."""

import json
import tempfile
from pathlib import Path

import pytest
import yaml

from core.simulation_spec import SimulationSpec
from core.exp_loader import load_experiment_yaml, ExperimentalDataset
from core.mae import compute_mae, compute_relative_mae
from core.report_writer import (
    ModelScore, build_experiment_report, build_evaluation_report,
    write_report_json,
)
from core.observable_extractor import extract_observable
from core.runner import SimulationResult


# ── SimulationSpec ──────────────────────────────────────────────────────

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


# ── exp_loader ──────────────────────────────────────────────────────────

def test_load_experiment_yaml():
    data = {
        "source": "test",
        "experiments": [
            {
                "id": "exp_001",
                "reactor_type": "shock_tube",
                "observable": "ignition_delay",
                "temperature": 1200,
                "pressure_atm": 1.0,
                "composition": {"NH3": 0.01, "O2": 0.01, "Ar": 0.98},
                "data": [
                    {"temperature": 1200, "value": 0.00045},
                    {"temperature": 1400, "value": 0.00012},
                ],
            }
        ],
    }

    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        yaml.dump(data, f)
        f.flush()
        datasets = load_experiment_yaml(f.name)

    assert len(datasets) == 1
    ds = datasets[0]
    assert ds.spec.experiment_id == "exp_001"
    assert ds.spec.reactor_type == "shock_tube"
    assert abs(ds.spec.pressure - 101325.0) < 1.0
    assert len(ds.data) == 2
    assert ds.data[0].value == 0.00045
    assert ds.metadata.get("source") == "test"


def test_load_experiment_yaml_missing_key():
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        yaml.dump({"wrong_key": []}, f)
        f.flush()
        with pytest.raises(ValueError, match="experiments"):
            load_experiment_yaml(f.name)


# ── MAE ─────────────────────────────────────────────────────────────────

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


# ── Observable extractor ───────────────────────────────────────────────

def test_extract_ignition_delay():
    spec = SimulationSpec(
        experiment_id="t1", reactor_type="shock_tube",
        observable="ignition_delay",
        temperature=1200, pressure=101325,
        composition={"NH3": 0.01},
    )
    # Simulate clear ignition: 1200 K → 2800 K (T_rise=1600 K > 400 K threshold)
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
    # NH3 goes from 0.01 → 0.006 → 0.004 (half = 0.005)
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


# ── Report writer ──────────────────────────────────────────────────────

def test_experiment_report_ranking():
    scores = [
        ModelScore(model_name="model_A", mechanism_file="a.yaml", mae=0.5, n_points=3),
        ModelScore(model_name="model_B", mechanism_file="b.yaml", mae=0.2, n_points=3),
        ModelScore(model_name="model_C", mechanism_file="c.yaml", mae=None, error="failed"),
    ]
    report = build_experiment_report(
        experiment_id="e1", reactor_type="shock_tube",
        observable="ignition_delay", scores=scores,
    )
    assert report.winner == "model_B"
    assert report.ranking == ["model_B", "model_A"]


def test_experiment_report_threshold():
    scores = [
        ModelScore(model_name="good", mechanism_file="g.yaml", mae=0.1, n_points=5),
        ModelScore(model_name="bad", mechanism_file="b.yaml", mae=0.5, n_points=5),
    ]
    report = build_experiment_report(
        experiment_id="e1", reactor_type="shock_tube",
        observable="ignition_delay", scores=scores, threshold=0.3,
    )
    assert report.models[0].passed_threshold is True
    assert report.models[1].passed_threshold is False


def test_write_report_json():
    scores = [ModelScore(model_name="m1", mechanism_file="m1.yaml", mae=0.1, n_points=2)]
    er = build_experiment_report(
        experiment_id="e1", reactor_type="shock_tube",
        observable="ignition_delay", scores=scores,
    )
    report = build_evaluation_report([er])

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "report.json"
        write_report_json(report, path)

        data = json.loads(path.read_text())
        assert len(data["experiments"]) == 1
        assert data["experiments"][0]["winner"] == "m1"
        assert data["summary"]["total_experiments"] == 1
