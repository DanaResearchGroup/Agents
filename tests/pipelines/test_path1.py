"""Tests for Path 1 pipeline — all external dependencies mocked."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.llm_client import LLMClient, LLMConfig
from src.agents.validators import ModelIsolationViolation
from src.schemas.experimental import (
    ExperimentalCondition,
    ExperimentalDataset,
    MAEResult,
    ObservableType,
    Path1Results,
    ReactorType,
    SimConditions,
)
from src.schemas.ingestion import PaperRecord
from src.simulation.core.runner import SimulationResult


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture()
def llm_client() -> LLMClient:
    return LLMClient(config=LLMConfig())


@pytest.fixture()
def original_model(tmp_path: Path) -> Path:
    model = tmp_path / "original.yaml"
    model.write_text("phases:\n- name: gas\n")
    return model


@pytest.fixture()
def si_dir(tmp_path: Path) -> Path:
    """SI directory with a Chemkin mechanism file."""
    d = tmp_path / "si"
    d.mkdir()
    (d / "mech.inp").write_text("ELEMENTS H O END")
    return d


@pytest.fixture()
def paper(si_dir: Path, tmp_path: Path) -> PaperRecord:
    pdf = tmp_path / "paper.pdf"
    pdf.write_text("fake pdf")
    return PaperRecord(
        id="test-paper",
        title="Test Paper",
        doi="10.1234/test",
        si_path=str(si_dir),
        pdf_path=str(pdf),
    )


@pytest.fixture()
def experimental_data() -> ExperimentalDataset:
    return ExperimentalDataset(
        name="test_dataset",
        conditions=[
            ExperimentalCondition(
                reactor_type=ReactorType.SHOCK_TUBE,
                temperature_K=1200.0,
                pressure_atm=1.0,
                mixture={"H2": 0.5, "O2": 0.5},
                observable_type=ObservableType.IDT,
                measured_value=0.001,
                error_threshold=0.01,
            ),
            ExperimentalCondition(
                reactor_type=ReactorType.SHOCK_TUBE,
                temperature_K=1400.0,
                pressure_atm=1.0,
                mixture={"H2": 0.5, "O2": 0.5},
                observable_type=ObservableType.IDT,
                measured_value=0.0005,
                error_threshold=0.01,
            ),
        ],
    )


TWO_CONDITIONS = [
    SimConditions(
        reactor_type=ReactorType.SHOCK_TUBE,
        T=1200.0,
        P=1.0,
        X={"H2": 0.5, "O2": 0.5},
        observable_type=ObservableType.IDT,
    ),
    SimConditions(
        reactor_type=ReactorType.SHOCK_TUBE,
        T=1400.0,
        P=1.0,
        X={"H2": 0.5, "O2": 0.5},
        observable_type=ObservableType.IDT,
    ),
]


def _make_sim_result(experiment_id: str, value: float) -> SimulationResult:
    """Helper: build a SimulationResult with an IDT-style observable."""
    return SimulationResult(
        experiment_id=experiment_id,
        mechanism="mock",
        times=[0.0, value, value * 2],
        temperature_history=[300.0, 2000.0, 2100.0],  # big jump at index 1
        success=True,
    )


def _make_failed_result(experiment_id: str, error: str) -> SimulationResult:
    return SimulationResult(
        experiment_id=experiment_id,
        mechanism="mock",
        success=False,
        error=error,
    )


# ── Shared mock setup ───────────────────────────────────────────────────────


def _patch_all():
    """Return a dict of patch contexts for all external dependencies."""
    from src.schemas.experimental import ConversionResult, PaperDocument, PageText

    converter_mock = AsyncMock()
    converter_mock.return_value.convert = AsyncMock(return_value=ConversionResult(
        success=True,
        output_path=Path("/tmp/mech.yaml"),
        attempts=1,
    ))

    converter_instance = MagicMock(
        convert=AsyncMock(return_value=ConversionResult(
            success=True,
            output_path=Path("/tmp/mech.yaml"),
            attempts=1,
        )),
        extract_rates=MagicMock(return_value={
            "h + o2 <=> o + oh": {"A": 1.04e14, "n": 0.0, "Ea": 15310.0},
        }),
    )

    return {
        "converter_cls": patch(
            "src.pipelines.path1.ChemkinConverter",
            return_value=converter_instance,
        ),
        "validator_cls": patch(
            "src.pipelines.path1.ModelIsolationValidator",
            return_value=MagicMock(validate_path1=MagicMock(return_value=None)),
        ),
        "parse_pdf": patch(
            "src.pipelines.path1.parse_pdf",
            return_value=PaperDocument(
                pdf_path="/tmp/paper.pdf",
                title="Test",
                pages=[PageText(page_num=1, text="conditions: T=1200K")],
                captions=[],
            ),
        ),
        "extract": patch(
            "src.pipelines.path1.ConditionExtractionAgent",
            return_value=MagicMock(extract=AsyncMock(return_value=TWO_CONDITIONS)),
        ),
    }


# ── Test: happy path ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_happy_path_two_conditions(
    paper: PaperRecord,
    original_model: Path,
    experimental_data: ExperimentalDataset,
    llm_client: LLMClient,
):
    """Two conditions, both succeed. Literature model has lower MAE on both."""
    patches = _patch_all()

    call_count = 0

    def mock_run_sim(spec, mechanism_file):
        nonlocal call_count
        call_count += 1
        # Literature model: simulated IDT closer to measured
        if "mech.yaml" in mechanism_file:
            return _make_sim_result(spec.experiment_id, 0.0009)
        # Original model: simulated IDT farther from measured
        return _make_sim_result(spec.experiment_id, 0.005)

    with (
        patches["converter_cls"],
        patches["validator_cls"],
        patches["parse_pdf"],
        patches["extract"],
        patch("src.pipelines.path1.run_simulation", side_effect=mock_run_sim),
    ):
        from src.pipelines.path1 import run_path1

        result = await run_path1(paper, original_model, experimental_data, llm_client)

    assert isinstance(result, Path1Results)
    assert result.conditions_tested == 2
    assert len(result.mae_results) == 4  # 2 per condition
    assert result.source_doi == "10.1234/test"
    assert result.literature_better_count == 2
    assert result.overall_literature_better is True
    assert result.extracted_rates is not None
    assert "h + o2 <=> o + oh" in result.extracted_rates


# ── Test: no SI mechanism ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_no_si_mechanism(
    original_model: Path,
    experimental_data: ExperimentalDataset,
    llm_client: LLMClient,
):
    """Paper has no si_path → ValueError."""
    paper = PaperRecord(id="no-si", title="No SI")

    from src.pipelines.path1 import run_path1

    with pytest.raises(ValueError, match="No mechanism found in SI"):
        await run_path1(paper, original_model, experimental_data, llm_client)


# ── Test: conversion failure ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_conversion_failure(
    paper: PaperRecord,
    original_model: Path,
    experimental_data: ExperimentalDataset,
    llm_client: LLMClient,
):
    """ChemkinConverter fails → ValueError."""
    from src.schemas.experimental import ConversionResult

    with patch(
        "src.pipelines.path1.ChemkinConverter",
        return_value=MagicMock(convert=AsyncMock(return_value=ConversionResult(
            success=False,
            output_path=None,
            errors=["syntax error in line 42"],
            attempts=2,
        ))),
    ):
        from src.pipelines.path1 import run_path1

        with pytest.raises(ValueError, match="Mechanism conversion failed"):
            await run_path1(paper, original_model, experimental_data, llm_client)


# ── Test: isolation violation ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_isolation_violation_propagates(
    paper: PaperRecord,
    original_model: Path,
    experimental_data: ExperimentalDataset,
    llm_client: LLMClient,
):
    """ModelIsolationViolation must propagate — not caught."""
    from src.schemas.experimental import ConversionResult

    violation = ModelIsolationViolation({"H + O2 <=> OH + O"})

    with (
        patch(
            "src.pipelines.path1.ChemkinConverter",
            return_value=MagicMock(convert=AsyncMock(return_value=ConversionResult(
                success=True,
                output_path=Path("/tmp/mech.yaml"),
                attempts=1,
            ))),
        ),
        patch(
            "src.pipelines.path1.ModelIsolationValidator",
            return_value=MagicMock(validate_path1=MagicMock(side_effect=violation)),
        ),
    ):
        from src.pipelines.path1 import run_path1

        with pytest.raises(ModelIsolationViolation):
            await run_path1(paper, original_model, experimental_data, llm_client)


# ── Test: no conditions extracted ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_no_conditions_extracted(
    paper: PaperRecord,
    original_model: Path,
    experimental_data: ExperimentalDataset,
    llm_client: LLMClient,
):
    """Extraction returns empty list → ValueError."""
    from src.schemas.experimental import ConversionResult, PaperDocument, PageText

    with (
        patch(
            "src.pipelines.path1.ChemkinConverter",
            return_value=MagicMock(convert=AsyncMock(return_value=ConversionResult(
                success=True, output_path=Path("/tmp/mech.yaml"), attempts=1,
            ))),
        ),
        patch(
            "src.pipelines.path1.ModelIsolationValidator",
            return_value=MagicMock(validate_path1=MagicMock(return_value=None)),
        ),
        patch(
            "src.pipelines.path1.parse_pdf",
            return_value=PaperDocument(
                pdf_path="/tmp/paper.pdf", title="Test",
                pages=[PageText(page_num=1, text="no conditions")],
                captions=[],
            ),
        ),
        patch(
            "src.pipelines.path1.ConditionExtractionAgent",
            return_value=MagicMock(extract=AsyncMock(return_value=[])),
        ),
    ):
        from src.pipelines.path1 import run_path1

        with pytest.raises(ValueError, match="No conditions extracted"):
            await run_path1(paper, original_model, experimental_data, llm_client)


# ── Test: partial simulation failure ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_partial_simulation_failure(
    paper: PaperRecord,
    original_model: Path,
    experimental_data: ExperimentalDataset,
    llm_client: LLMClient,
):
    """One condition's simulation fails — other succeeds. Results are partial."""
    patches = _patch_all()

    sim_call = 0

    def mock_run_sim(spec, mechanism_file):
        nonlocal sim_call
        sim_call += 1
        # Fail the first condition's literature run
        if "cond_0" in spec.experiment_id and "mech.yaml" in mechanism_file:
            return _make_failed_result(spec.experiment_id, "diverged")
        return _make_sim_result(spec.experiment_id, 0.0009)

    with (
        patches["converter_cls"],
        patches["validator_cls"],
        patches["parse_pdf"],
        patches["extract"],
        patch("src.pipelines.path1.run_simulation", side_effect=mock_run_sim),
    ):
        from src.pipelines.path1 import run_path1

        result = await run_path1(paper, original_model, experimental_data, llm_client)

    # Only one condition should have results (cond_1)
    assert result.conditions_tested == 1
    assert len(result.mae_results) == 2  # lit + orig for one condition


# ── Test: overall_literature_better majority logic ───────────────────────────


@pytest.mark.asyncio
async def test_majority_logic_original_better(
    paper: PaperRecord,
    original_model: Path,
    experimental_data: ExperimentalDataset,
    llm_client: LLMClient,
):
    """Original model is better on both conditions → overall_literature_better is False."""
    patches = _patch_all()

    def mock_run_sim(spec, mechanism_file):
        # Original model closer to measured values
        if "mech.yaml" in mechanism_file:
            return _make_sim_result(spec.experiment_id, 0.01)  # literature: far
        return _make_sim_result(spec.experiment_id, 0.0009)    # original: close

    with (
        patches["converter_cls"],
        patches["validator_cls"],
        patches["parse_pdf"],
        patches["extract"],
        patch("src.pipelines.path1.run_simulation", side_effect=mock_run_sim),
    ):
        from src.pipelines.path1 import run_path1

        result = await run_path1(paper, original_model, experimental_data, llm_client)

    assert result.literature_better_count == 0
    assert result.overall_literature_better is False
