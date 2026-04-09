"""Tests for src/report.py — ReportGenerator."""

from pathlib import Path

import pytest
import yaml

from src.report import ReportGenerator
from src.schemas.experimental import (
    BranchResult,
    ExperimentalCondition,
    ExperimentalDataset,
    MAEResult,
    ObservableType,
    Path1Results,
    Path2Results,
)


@pytest.fixture()
def generator() -> ReportGenerator:
    return ReportGenerator()


@pytest.fixture()
def original_model(tmp_path: Path) -> Path:
    p = tmp_path / "model.yaml"
    p.write_text("phases: []")
    return p


@pytest.fixture()
def experimental_data() -> ExperimentalDataset:
    return ExperimentalDataset(
        name="NH3_JSR_Dagaut2000",
        source="10.1016/test",
        conditions=[
            ExperimentalCondition(
                reactor_type="jsr",
                temperature_K=[1200],
                pressure_atm=1.0,
                mixture={"NH3": 0.01, "O2": 0.015, "N2": 0.975},
                observable_type=ObservableType.SPECIES_PROFILE,
                measured_value=0.001,
                error_threshold=0.05,
            ),
        ],
    )


@pytest.fixture()
def path1_results() -> Path1Results:
    return Path1Results(
        literature_model_path=Path("/tmp/wang2023.yaml"),
        source_doi="10.1016/j.combustflame.2023.xxxxx",
        conditions_tested=2,
        mae_results=[
            MAEResult(model_name="original", conditions_id="cond_0", mae=0.04312345, threshold=0.05, passed=True),
            MAEResult(model_name="literature", conditions_id="cond_0", mae=0.02156789, threshold=0.05, passed=True),
            MAEResult(model_name="original", conditions_id="cond_1", mae=0.03887766, threshold=0.05, passed=True),
            MAEResult(model_name="literature", conditions_id="cond_1", mae=0.01923456, threshold=0.05, passed=True),
        ],
        literature_better_count=2,
        overall_literature_better=True,
    )


@pytest.fixture()
def path2_results() -> Path2Results:
    return Path2Results(
        baseline_mae=0.04312345,
        branches=[
            BranchResult(
                branch_id="branch_001",
                reactions_added=["NH2 + O <=> HNO + H"],
                mae=0.03100000,
                delta_mae=-0.01212345,
                improved=True,
                threshold=0.04,
                passed=True,
            ),
            BranchResult(
                branch_id="branch_002",
                reactions_added=["H + O2 <=> OH + O"],
                mae=0.05000000,
                delta_mae=0.00687655,
                improved=False,
                threshold=0.04,
                passed=False,
            ),
        ],
        best_branch_id="branch_001",
    )


def _load_report(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


# ── Tests ────────────────────────────────────────────────────────────────────


def test_both_paths_present(
    generator, original_model, experimental_data, path1_results, path2_results, tmp_path,
):
    """Full report with both paths — all keys present."""
    path = generator.generate(
        original_model, experimental_data, tmp_path,
        path1_results=path1_results,
        path2_results=path2_results,
    )
    report = _load_report(path)
    assert "run_id" in report
    assert "path1" in report
    assert "path2" in report
    assert report["path1"]["conditions_tested"] == 2
    assert len(report["path2"]["branches"]) == 2


def test_path1_only(
    generator, original_model, experimental_data, path1_results, tmp_path,
):
    """path1 only — path2 key absent."""
    path = generator.generate(
        original_model, experimental_data, tmp_path,
        path1_results=path1_results,
    )
    report = _load_report(path)
    assert "path1" in report
    assert "path2" not in report


def test_path2_only(
    generator, original_model, experimental_data, path2_results, tmp_path,
):
    """path2 only — path1 key absent."""
    path = generator.generate(
        original_model, experimental_data, tmp_path,
        path2_results=path2_results,
    )
    report = _load_report(path)
    assert "path1" not in report
    assert "path2" in report


def test_neither_path(
    generator, original_model, experimental_data, tmp_path,
):
    """Neither path — only run metadata."""
    path = generator.generate(original_model, experimental_data, tmp_path)
    report = _load_report(path)
    assert "run_id" in report
    assert "original_model" in report
    assert "experimental_data" in report
    assert "path1" not in report
    assert "path2" not in report


def test_mae_formatted_4_decimal(
    generator, original_model, experimental_data, path1_results, path2_results, tmp_path,
):
    """All MAE floats formatted to 4 decimal places."""
    path = generator.generate(
        original_model, experimental_data, tmp_path,
        path1_results=path1_results,
        path2_results=path2_results,
    )
    text = path.read_text()
    # Check the raw YAML text for 4-decimal formatting
    # original_mae: 0.0431 (not 0.04312345)
    assert "0.0431" in text
    assert "0.0216" in text
    # path2 baseline
    assert "baseline_mae" in text


def test_output_file_exists(
    generator, original_model, experimental_data, tmp_path,
):
    """Output file exists at returned path."""
    path = generator.generate(original_model, experimental_data, tmp_path)
    assert path.exists()
    assert path.is_file()
    assert path.suffix == ".yaml"


def test_run_id_format(
    generator, original_model, experimental_data, tmp_path,
):
    """run_id matches run_YYYYMMDD_HHMMSS pattern."""
    import re
    path = generator.generate(original_model, experimental_data, tmp_path)
    report = _load_report(path)
    assert re.match(r"run_\d{8}_\d{6}", report["run_id"])
