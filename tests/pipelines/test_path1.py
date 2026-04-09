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
    NormalizedComposition,
    NormalizedCondition,
    ObservableType,
    Path1Results,
    ReactorType,
    SimConditions,
    SimulationPlan,
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
            "src.pipelines.path1.extract_conditions",
            new=AsyncMock(return_value=TWO_CONDITIONS),
        ),
        "model_species": patch(
            "src.pipelines.path1._get_model_species",
            return_value=["H2", "O2", "H2O"],
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

    def mock_run_sim(spec, mechanism_file, **kwargs):
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
        patches["model_species"],
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
    """Paper has no si_path -> ValueError."""
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
    """ChemkinConverter fails -> ValueError."""
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
    """Both agent and fallback return empty -> ValueError."""
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
            "src.pipelines.path1.extract_conditions",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "src.pipelines.path1._get_model_species",
            return_value=[],
        ),
        patch(
            "src.pipelines.path1.PaperExtractionPipeline",
            return_value=MagicMock(extract=MagicMock(return_value=[])),
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

    def mock_run_sim(spec, mechanism_file, **kwargs):
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
        patches["model_species"],
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
    """Original model is better on both conditions -> overall_literature_better is False."""
    patches = _patch_all()

    def mock_run_sim(spec, mechanism_file, **kwargs):
        # Original model closer to measured values
        if "mech.yaml" in mechanism_file:
            return _make_sim_result(spec.experiment_id, 0.01)  # literature: far
        return _make_sim_result(spec.experiment_id, 0.0009)    # original: close

    with (
        patches["converter_cls"],
        patches["validator_cls"],
        patches["parse_pdf"],
        patches["extract"],
        patches["model_species"],
        patch("src.pipelines.path1.run_simulation", side_effect=mock_run_sim),
    ):
        from src.pipelines.path1 import run_path1

        result = await run_path1(paper, original_model, experimental_data, llm_client)

    assert result.literature_better_count == 0
    assert result.overall_literature_better is False


# ── Helpers for deterministic pipeline tests ─────────────────────────────────


def _make_executable_plan(
    scenario_id: str = "scenario_1",
    family: str = "shock_tube",
    T_min: float = 1200.0,
    T_max: float = 1200.0,
    P: float = 1.0,
    species: dict | None = None,
    observable: str = "ignition_delay",
) -> SimulationPlan:
    return SimulationPlan(
        scenario_id=scenario_id,
        experiment_family=family,
        template_family="idt_const_uv",
        plan_status="executable",
        temperature=NormalizedCondition(
            raw_text=f"{T_min}-{T_max} K",
            kind="temperature",
            role="setup_range",
            is_range=T_min != T_max,
            min_value=T_min,
            max_value=T_max,
            unit="K",
            source_page=1,
        ),
        pressure=NormalizedCondition(
            raw_text=f"{P} atm",
            kind="pressure",
            role="setpoint",
            is_range=False,
            min_value=P,
            max_value=P,
            unit="atm",
            source_page=1,
        ),
        composition=NormalizedComposition(
            species=species or {"H2": 0.5, "O2": 0.5},
            balance_species=None,
            composition_complete=True,
            raw_mentions=["50% H2 / 50% O2"],
            source_pages=[1],
        ),
        target_observables=[observable],
        auto_generation_allowed=True,
    )


def _pipeline_patches(plans: list[SimulationPlan]):
    """Shared patches for deterministic pipeline tests."""
    from src.schemas.experimental import ConversionResult, PaperDocument, PageText

    return {
        "converter_cls": patch(
            "src.pipelines.path1.ChemkinConverter",
            return_value=MagicMock(
                convert=AsyncMock(return_value=ConversionResult(
                    success=True, output_path=Path("/tmp/mech.yaml"), attempts=1,
                )),
                extract_rates=MagicMock(return_value={}),
            ),
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
                pages=[PageText(page_num=1, text="shock tube T=1200K 1 atm")],
                captions=[],
            ),
        ),
        "model_species": patch(
            "src.pipelines.path1._get_model_species",
            return_value=["H2", "O2"],
        ),
    }


# ── Test: agent returns conditions -> deterministic fallback NOT called ──────


@pytest.mark.asyncio
async def test_agent_returns_conditions_skips_fallback(
    paper: PaperRecord,
    original_model: Path,
    experimental_data: ExperimentalDataset,
    llm_client: LLMClient,
):
    """When ConditionReasoningAgent returns conditions, deterministic fallback is skipped."""
    pp = _pipeline_patches([])

    pipeline_cls = MagicMock()

    with (
        pp["converter_cls"],
        pp["validator_cls"],
        pp["parse_pdf"],
        pp["model_species"],
        patch(
            "src.pipelines.path1.extract_conditions",
            new=AsyncMock(return_value=TWO_CONDITIONS),
        ),
        patch("src.pipelines.path1.PaperExtractionPipeline", pipeline_cls),
        patch("src.pipelines.path1.run_simulation", return_value=_make_sim_result("x", 0.001)),
    ):
        from src.pipelines.path1 import run_path1

        result = await run_path1(paper, original_model, experimental_data, llm_client)

    # Deterministic pipeline should NOT be instantiated (agent succeeded)
    pipeline_cls.assert_not_called()
    assert result.conditions_tested >= 1


# ── Test: agent returns empty -> deterministic fallback called ───────────────


@pytest.mark.asyncio
async def test_agent_empty_triggers_deterministic_fallback(
    paper: PaperRecord,
    original_model: Path,
    experimental_data: ExperimentalDataset,
    llm_client: LLMClient,
):
    """When agent returns [], deterministic pipeline runs as fallback."""
    plans = [_make_executable_plan()]

    pp = _pipeline_patches(plans)
    pipeline_mock = MagicMock(
        extract=MagicMock(return_value=plans),
    )

    with (
        pp["converter_cls"],
        pp["validator_cls"],
        pp["parse_pdf"],
        pp["model_species"],
        patch(
            "src.pipelines.path1.extract_conditions",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "src.pipelines.path1.PaperExtractionPipeline",
            return_value=pipeline_mock,
        ),
        patch("src.pipelines.path1.run_simulation", return_value=_make_sim_result("x", 0.001)),
    ):
        from src.pipelines.path1 import run_path1

        result = await run_path1(paper, original_model, experimental_data, llm_client)

    # Pipeline was used as fallback
    pipeline_mock.extract.assert_called_once()
    assert result.conditions_tested >= 1


# ── Test: model_species passed to agent ──────────────────────────────────────


@pytest.mark.asyncio
async def test_model_species_passed_to_agent(
    paper: PaperRecord,
    original_model: Path,
    experimental_data: ExperimentalDataset,
    llm_client: LLMClient,
):
    """model_species from _get_model_species is passed to extract_conditions."""
    pp = _pipeline_patches([])

    extract_mock = AsyncMock(return_value=TWO_CONDITIONS)

    with (
        pp["converter_cls"],
        pp["validator_cls"],
        pp["parse_pdf"],
        patch("src.pipelines.path1._get_model_species", return_value=["H2", "O2", "N2"]),
        patch("src.pipelines.path1.extract_conditions", new=extract_mock),
        patch("src.pipelines.path1.run_simulation", return_value=_make_sim_result("x", 0.001)),
    ):
        from src.pipelines.path1 import run_path1

        await run_path1(paper, original_model, experimental_data, llm_client)

    # Verify model_species was passed
    call_kwargs = extract_mock.call_args
    assert call_kwargs.kwargs["model_species"] == ["H2", "O2", "N2"]


# ── Test: paper_summary passed when available ────────────────────────────────


@pytest.mark.asyncio
async def test_paper_summary_passed_to_agent(
    paper: PaperRecord,
    original_model: Path,
    experimental_data: ExperimentalDataset,
    llm_client: LLMClient,
):
    """paper_summary is forwarded to extract_conditions."""
    from src.schemas.experimental import PaperSummary

    pp = _pipeline_patches([])
    summary = PaperSummary(reactor_types=["shock_tube"])

    extract_mock = AsyncMock(return_value=TWO_CONDITIONS)

    with (
        pp["converter_cls"],
        pp["validator_cls"],
        pp["parse_pdf"],
        pp["model_species"],
        patch("src.pipelines.path1.extract_conditions", new=extract_mock),
        patch("src.pipelines.path1.run_simulation", return_value=_make_sim_result("x", 0.001)),
    ):
        from src.pipelines.path1 import run_path1

        await run_path1(
            paper, original_model, experimental_data, llm_client,
            paper_summary=summary,
        )

    call_kwargs = extract_mock.call_args
    assert call_kwargs.kwargs["paper_summary"] is not None
    assert call_kwargs.kwargs["paper_summary"].reactor_types == ["shock_tube"]


# ── Test: deduplication removes identical conditions ─────────────────────────


def test_deduplication_removes_identical_conditions():
    """Identical (reactor_type, T, P, observable_type) are deduplicated."""
    from src.pipelines.path1 import _deduplicate_conditions

    conditions = [
        SimConditions(
            reactor_type=ReactorType.SHOCK_TUBE,
            T=1200.0, P=1.0,
            X={"H2": 0.5, "O2": 0.5},
            observable_type=ObservableType.IDT,
        ),
        SimConditions(
            reactor_type=ReactorType.SHOCK_TUBE,
            T=1200.0, P=1.0,
            X={"H2": 0.3, "O2": 0.7},  # different X, same T/P/reactor/obs
            observable_type=ObservableType.IDT,
        ),
        SimConditions(
            reactor_type=ReactorType.SHOCK_TUBE,
            T=1400.0, P=1.0,
            X={"H2": 0.5, "O2": 0.5},
            observable_type=ObservableType.IDT,
        ),
    ]
    result = _deduplicate_conditions(conditions)
    # First two share (SHOCK_TUBE, 1200.0, 1.0, IDT) -> one kept
    assert len(result) == 2
    assert result[0].T == 1200.0
    assert result[1].T == 1400.0


# ── Test: provided literature model skips SI discovery ───────────────────────


@pytest.mark.asyncio
async def test_literature_model_skips_si_discovery(
    paper: PaperRecord,
    original_model: Path,
    experimental_data: ExperimentalDataset,
    llm_client: LLMClient,
    tmp_path: Path,
):
    """When literature_model is provided, _find_mechanism and converter are not called."""
    from src.schemas.experimental import PaperDocument, PageText

    lit_model = tmp_path / "lit.yaml"
    lit_model.write_text("phases: []")

    with (
        patch("src.pipelines.path1._find_mechanism") as mock_find,
        patch("src.pipelines.path1.ChemkinConverter") as mock_converter_cls,
        patch(
            "src.pipelines.path1.ModelIsolationValidator",
            return_value=MagicMock(validate_path1=MagicMock(return_value=None)),
        ),
        patch(
            "src.pipelines.path1.parse_pdf",
            return_value=PaperDocument(
                pdf_path="/tmp/paper.pdf",
                title="Test",
                pages=[PageText(page_num=1, text="T=1200K")],
                captions=[],
            ),
        ),
        patch(
            "src.pipelines.path1.extract_conditions",
            new=AsyncMock(return_value=TWO_CONDITIONS),
        ),
        patch("src.pipelines.path1._get_model_species", return_value=[]),
        patch("src.pipelines.path1.run_simulation", return_value=_make_sim_result("x", 0.001)),
    ):
        from src.pipelines.path1 import run_path1

        result = await run_path1(
            paper, original_model, experimental_data, llm_client,
            literature_model=lit_model,
        )

    mock_find.assert_not_called()
    mock_converter_cls.assert_not_called()
    assert result.literature_model_path == lit_model
    assert result.extracted_rates is None


# ── Tests: tolerance matching ────────────────────────────────────────────────


def _make_dataset(*conditions):
    """Helper: build an ExperimentalDataset from (T, P) tuples."""
    return ExperimentalDataset(
        name="test",
        conditions=[
            ExperimentalCondition(
                reactor_type=ReactorType.SHOCK_TUBE,
                temperature_K=t,
                pressure_atm=p,
                mixture={"H2": 0.5, "O2": 0.5},
                observable_type=ObservableType.IDT,
                measured_value=0.001,
                error_threshold=0.01,
            )
            for t, p in conditions
        ],
    )


def test_match_experimental_exact():
    """Exact T/P still matches."""
    from src.pipelines.path1 import _match_experimental

    cond = SimConditions(
        reactor_type=ReactorType.SHOCK_TUBE, T=1200.0, P=1.0,
        X={"H2": 0.5}, observable_type=ObservableType.IDT,
    )
    ds = _make_dataset((1200.0, 1.0))
    result = _match_experimental(cond, ds)
    assert result is not None
    assert result.temperature_K == 1200.0


def test_match_experimental_within_tolerance():
    """T within 100K and P within 0.15atm matches."""
    from src.pipelines.path1 import _match_experimental

    cond = SimConditions(
        reactor_type=ReactorType.SHOCK_TUBE, T=1280.0, P=1.10,
        X={"H2": 0.5}, observable_type=ObservableType.IDT,
    )
    ds = _make_dataset((1200.0, 1.0))
    result = _match_experimental(cond, ds)
    assert result is not None


def test_match_experimental_outside_tolerance():
    """T outside 100K does not match."""
    from src.pipelines.path1 import _match_experimental

    cond = SimConditions(
        reactor_type=ReactorType.SHOCK_TUBE, T=1350.0, P=1.0,
        X={"H2": 0.5}, observable_type=ObservableType.IDT,
    )
    ds = _make_dataset((1200.0, 1.0))
    result = _match_experimental(cond, ds)
    assert result is None


def test_match_experimental_p_outside_tolerance():
    """P outside 0.15atm does not match even if T is close."""
    from src.pipelines.path1 import _match_experimental

    cond = SimConditions(
        reactor_type=ReactorType.SHOCK_TUBE, T=1200.0, P=1.20,
        X={"H2": 0.5}, observable_type=ObservableType.IDT,
    )
    ds = _make_dataset((1200.0, 1.0))
    result = _match_experimental(cond, ds)
    assert result is None


def test_match_experimental_closest_selected():
    """When multiple candidates are within tolerance, closest wins."""
    from src.pipelines.path1 import _match_experimental

    cond = SimConditions(
        reactor_type=ReactorType.SHOCK_TUBE, T=1210.0, P=1.0,
        X={"H2": 0.5}, observable_type=ObservableType.IDT,
    )
    ds = _make_dataset((1200.0, 1.0), (1250.0, 1.0))
    result = _match_experimental(cond, ds)
    assert result is not None
    assert result.temperature_K == 1200.0  # closer to 1210


# ── Tests: reactor family filtering ─────────────────────────────────────────


def test_filter_by_reactor_family_keeps_matching():
    """Conditions matching experimental reactor types are kept."""
    from src.pipelines.path1 import _filter_by_reactor_family

    conditions = [
        SimConditions(
            reactor_type=ReactorType.SHOCK_TUBE, T=1200.0, P=1.0,
            X={"H2": 0.5}, observable_type=ObservableType.IDT,
        ),
    ]
    ds = _make_dataset((1200.0, 1.0))
    result = _filter_by_reactor_family(conditions, ds)
    assert len(result) == 1


def test_filter_by_reactor_family_drops_wrong_family():
    """JSR conditions dropped when experimental data is only shock_tube."""
    from src.pipelines.path1 import _filter_by_reactor_family

    conditions = [
        SimConditions(
            reactor_type=ReactorType.SHOCK_TUBE, T=1200.0, P=1.0,
            X={"H2": 0.5}, observable_type=ObservableType.IDT,
        ),
        SimConditions(
            reactor_type=ReactorType.JSR, T=800.0, P=1.0,
            X={"H2": 0.5}, observable_type=ObservableType.SPECIES_PROFILE,
        ),
    ]
    ds = _make_dataset((1200.0, 1.0))  # only shock_tube
    result = _filter_by_reactor_family(conditions, ds)
    assert len(result) == 1
    assert result[0].reactor_type == ReactorType.SHOCK_TUBE


def test_filter_by_reactor_family_all_dropped():
    """All conditions dropped when none match experimental reactor types."""
    from src.pipelines.path1 import _filter_by_reactor_family

    conditions = [
        SimConditions(
            reactor_type=ReactorType.JSR, T=800.0, P=1.0,
            X={"H2": 0.5}, observable_type=ObservableType.SPECIES_PROFILE,
        ),
    ]
    ds = _make_dataset((1200.0, 1.0))  # only shock_tube
    result = _filter_by_reactor_family(conditions, ds)
    assert len(result) == 0
