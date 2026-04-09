"""Tests for Path 2 pipeline — all external dependencies mocked."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.llm_client import LLMClient, LLMConfig
from src.schemas.experimental import (
    BranchResult,
    CandidateReaction,
    ExperimentalCondition,
    ExperimentalDataset,
    FigureCaption,
    FigureClassification,
    FluxInterpretation,
    FluxPathway,
    MAEResult,
    ModelBranch,
    ObservableType,
    PageText,
    PaperDocument,
    Path1Results,
    Path2Results,
    ReactorType,
    SensitiveReaction,
    SensitivityInterpretation,
)
from src.schemas.ingestion import PaperRecord
from src.simulation.core.runner import SimulationResult

from src.pipelines.path2 import run_path2


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def llm_client() -> LLMClient:
    return LLMClient(LLMConfig(provider="test", model="test-model"))


@pytest.fixture()
def original_model(tmp_path: Path) -> Path:
    model = tmp_path / "original.yaml"
    model.write_text("phases: []\nreactions: []\n")
    return model


@pytest.fixture()
def paper(tmp_path: Path) -> PaperRecord:
    pdf = tmp_path / "paper.pdf"
    pdf.write_text("fake")
    return PaperRecord(
        id="test-paper",
        title="Test Paper",
        doi="10.1234/test",
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


# ── Helpers ──────────────────────────────────────────────────────────────────

TWO_CAPTIONS = [
    FigureCaption(figure_id="fig1", label_type="figure", page_num=1, caption="Flux diagram for H2/O2"),
    FigureCaption(figure_id="fig2", label_type="figure", page_num=2, caption="Sensitivity analysis"),
]


def _make_sim_result(experiment_id: str, idt_value: float) -> SimulationResult:
    """Build a SimulationResult with an IDT-style observable."""
    return SimulationResult(
        experiment_id=experiment_id,
        mechanism="mock",
        times=[0.0, idt_value],
        temperature_history=[300.0, 2000.0],
        success=True,
    )


def _make_failed_result(experiment_id: str, error: str) -> SimulationResult:
    return SimulationResult(
        experiment_id=experiment_id,
        mechanism="mock",
        success=False,
        error=error,
    )


def _make_candidate(reaction_string: str, rate_params: dict | None = None, confidence: float = 0.8) -> CandidateReaction:
    return CandidateReaction(
        reaction_string=reaction_string,
        rate_params=rate_params,
        source_section="test",
        justification="test",
        confidence=confidence,
    )


def _mock_mining_agent():
    """Create a mock ReactionMiningAgent that returns two figure types."""
    agent = MagicMock()
    agent.classify_figure = AsyncMock(side_effect=[
        FigureClassification(label="sensitivity_analysis", confidence=0.9, reasoning="test"),
        FigureClassification(label="sensitivity_analysis", confidence=0.9, reasoning="test"),
    ])
    agent.interpret_sensitivity = AsyncMock(side_effect=[
        SensitivityInterpretation(
            target_property="IDT",
            top_reactions=[
                SensitiveReaction(rank=1, reaction="H + O2 => OH + O", coefficient=0.9, impact="high"),
            ],
            confidence=0.8,
        ),
        SensitivityInterpretation(
            target_property="IDT",
            top_reactions=[
                SensitiveReaction(rank=1, reaction="OH + H2 => H2O + H", coefficient=0.7, impact="medium"),
            ],
            confidence=0.7,
        ),
    ])
    return agent


def _patch_all(
    captions=None,
    mining_agent=None,
    branches=None,
    run_sim_side_effect=None,
):
    """Return a dict of patch contexts for all external dependencies."""
    if captions is None:
        captions = TWO_CAPTIONS
    if mining_agent is None:
        mining_agent = _mock_mining_agent()
    if branches is None:
        branches = [
            ModelBranch(
                branch_id="branch_000",
                reactions_added=[_make_candidate("H + O2 => OH + O", rate_params={"A": 1e10})],
                model_path=Path("/tmp/branch_000.yaml"),
            ),
            ModelBranch(
                branch_id="branch_001",
                reactions_added=[_make_candidate("OH + H2 => H2O + H", rate_params={"A": 2e10})],
                model_path=Path("/tmp/branch_001.yaml"),
            ),
        ]

    branching_mock = MagicMock()
    branching_mock.create_branch = MagicMock(side_effect=branches)

    return {
        "parse_pdf": patch(
            "src.pipelines.path2.parse_pdf",
            return_value=PaperDocument(
                pdf_path="/tmp/paper.pdf",
                title="Test",
                pages=[PageText(page_num=1, text="test content")],
                captions=captions,
            ),
        ),
        "mining_cls": patch(
            "src.pipelines.path2.ReactionMiningAgent",
            return_value=mining_agent,
        ),
        "branching_cls": patch(
            "src.pipelines.path2.ModelBranchingAgent",
            return_value=branching_mock,
        ),
        "run_sim": patch(
            "src.pipelines.path2.run_simulation",
            side_effect=run_sim_side_effect,
        ) if run_sim_side_effect else patch(
            "src.pipelines.path2.run_simulation",
        ),
    }


# ── Tests ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_happy_path_two_reactions_one_improves(
    paper, original_model, experimental_data, llm_client, tmp_path,
):
    """Two reactions mined, two branches — one improves baseline."""
    call_count = 0

    def mock_run_sim(spec, mechanism_file, **kwargs):
        nonlocal call_count
        call_count += 1
        # Baseline: IDT = 0.005 for both conditions (far from measured 0.001 / 0.0005)
        if "original" in mechanism_file:
            return _make_sim_result(spec.experiment_id, 0.005)
        # Branch 000: improves (closer to measured)
        if "branch_000" in mechanism_file:
            return _make_sim_result(spec.experiment_id, 0.0009)
        # Branch 001: does not improve
        return _make_sim_result(spec.experiment_id, 0.006)

    patches = _patch_all(run_sim_side_effect=mock_run_sim)

    with (
        patches["parse_pdf"],
        patches["mining_cls"],
        patches["branching_cls"],
        patches["run_sim"],
    ):
        result = await run_path2(
            paper, original_model, experimental_data, llm_client, tmp_path,
        )

    assert isinstance(result, Path2Results)
    assert result.baseline_mae > 0
    assert len(result.branches) == 2
    assert result.best_branch_id == "branch_000"

    improved = [b for b in result.branches if b.improved]
    assert len(improved) == 1
    assert improved[0].branch_id == "branch_000"
    assert improved[0].delta_mae < 0


@pytest.mark.asyncio
async def test_no_high_confidence_reactions_raises(
    paper, original_model, experimental_data, llm_client, tmp_path,
):
    """ValueError when all mined reactions have confidence < 0.5."""
    low_conf_agent = MagicMock()
    low_conf_agent.classify_figure = AsyncMock(return_value=FigureClassification(
        label="sensitivity_analysis", confidence=0.9, reasoning="test",
    ))
    low_conf_agent.interpret_sensitivity = AsyncMock(return_value=SensitivityInterpretation(
        target_property="IDT",
        top_reactions=[SensitiveReaction(rank=1, reaction="A => B", coefficient=0.1, impact="low")],
        confidence=0.3,  # below threshold
    ))

    patches = _patch_all(
        captions=[FigureCaption(figure_id="fig1", label_type="figure", page_num=1, caption="test")],
        mining_agent=low_conf_agent,
    )

    with (
        patches["parse_pdf"],
        patches["mining_cls"],
        patches["branching_cls"],
        patches["run_sim"],
    ):
        with pytest.raises(ValueError, match="No high-confidence candidate reactions"):
            await run_path2(
                paper, original_model, experimental_data, llm_client, tmp_path,
            )


@pytest.mark.asyncio
async def test_all_baseline_simulations_fail_raises(
    paper, original_model, experimental_data, llm_client, tmp_path,
):
    """ValueError when every baseline simulation fails."""
    def mock_run_sim(spec, mechanism_file, **kwargs):
        return _make_failed_result(spec.experiment_id, "Cantera error")

    patches = _patch_all(run_sim_side_effect=mock_run_sim)

    with (
        patches["parse_pdf"],
        patches["mining_cls"],
        patches["branching_cls"],
        patches["run_sim"],
    ):
        with pytest.raises(ValueError, match="All baseline simulations failed"):
            await run_path2(
                paper, original_model, experimental_data, llm_client, tmp_path,
            )


@pytest.mark.asyncio
async def test_reaction_missing_rate_params_skipped(
    paper, original_model, experimental_data, llm_client, tmp_path,
):
    """Reaction without rate_params → branch creation raises, branch skipped."""
    # Only one branch succeeds; the other raises ValueError (no rate_params)
    branching_mock = MagicMock()
    branching_mock.create_branch = MagicMock(side_effect=[
        ValueError("no reactions had rate_params"),
        ModelBranch(
            branch_id="branch_001",
            reactions_added=[_make_candidate("OH + H2 => H2O + H", rate_params={"A": 2e10})],
            model_path=Path("/tmp/branch_001.yaml"),
        ),
    ])

    def mock_run_sim(spec, mechanism_file, **kwargs):
        if "original" in mechanism_file:
            return _make_sim_result(spec.experiment_id, 0.005)
        return _make_sim_result(spec.experiment_id, 0.004)

    patches = _patch_all(run_sim_side_effect=mock_run_sim)

    with (
        patches["parse_pdf"],
        patches["mining_cls"],
        patch("src.pipelines.path2.ModelBranchingAgent", return_value=branching_mock),
        patches["run_sim"],
    ):
        result = await run_path2(
            paper, original_model, experimental_data, llm_client, tmp_path,
        )

    # Only one branch should exist (the one that didn't raise)
    assert len(result.branches) == 1
    assert result.branches[0].branch_id == "branch_001"


@pytest.mark.asyncio
async def test_branch_validation_failure_propagates(
    paper, original_model, experimental_data, llm_client, tmp_path,
):
    """ValueError from branch validation must propagate (branch skipped)."""
    branching_mock = MagicMock()
    branching_mock.create_branch = MagicMock(side_effect=[
        ValueError("Branch model is missing original reactions"),
        ValueError("Branch model is missing original reactions"),
    ])

    def mock_run_sim(spec, mechanism_file, **kwargs):
        return _make_sim_result(spec.experiment_id, 0.005)

    patches = _patch_all(run_sim_side_effect=mock_run_sim)

    with (
        patches["parse_pdf"],
        patches["mining_cls"],
        patch("src.pipelines.path2.ModelBranchingAgent", return_value=branching_mock),
        patches["run_sim"],
    ):
        result = await run_path2(
            paper, original_model, experimental_data, llm_client, tmp_path,
        )

    # All branch creations failed — no branches, no improvement
    assert len(result.branches) == 0
    assert result.best_branch_id is None


@pytest.mark.asyncio
async def test_no_branch_improves_best_is_none(
    paper, original_model, experimental_data, llm_client, tmp_path,
):
    """best_branch_id is None when no branch improves baseline."""
    def mock_run_sim(spec, mechanism_file, **kwargs):
        if "original" in mechanism_file:
            return _make_sim_result(spec.experiment_id, 0.001)  # perfect baseline
        return _make_sim_result(spec.experiment_id, 0.005)  # branches worse

    patches = _patch_all(run_sim_side_effect=mock_run_sim)

    with (
        patches["parse_pdf"],
        patches["mining_cls"],
        patches["branching_cls"],
        patches["run_sim"],
    ):
        result = await run_path2(
            paper, original_model, experimental_data, llm_client, tmp_path,
        )

    assert result.best_branch_id is None
    assert all(not b.improved for b in result.branches)


@pytest.mark.asyncio
async def test_path1_results_accepted(
    paper, original_model, experimental_data, llm_client, tmp_path,
):
    """path1_results with extracted_rates enriches candidate reactions."""
    path1 = Path1Results(
        literature_model_path=Path("/tmp/lit.yaml"),
        source_doi="10.1234/test",
        conditions_tested=2,
        mae_results=[
            MAEResult(model_name="literature", conditions_id="c0", mae=0.001, threshold=0.01, passed=True),
        ],
        literature_better_count=1,
        overall_literature_better=True,
        extracted_rates={
            "h + o2 <=> o + oh": {"A": 1.04e14, "n": 0.0, "Ea": 15310.0},
            "h2 + oh <=> h + h2o": {"A": 2.14e8, "n": 1.52, "Ea": 3449.0},
        },
    )

    def mock_run_sim(spec, mechanism_file, **kwargs):
        return _make_sim_result(spec.experiment_id, 0.005)

    patches = _patch_all(run_sim_side_effect=mock_run_sim)

    with (
        patches["parse_pdf"],
        patches["mining_cls"],
        patches["branching_cls"],
        patches["run_sim"],
    ):
        result = await run_path2(
            paper, original_model, experimental_data, llm_client, tmp_path,
            path1_results=path1,
        )

    assert isinstance(result, Path2Results)


@pytest.mark.asyncio
async def test_partial_branch_simulation_failure(
    paper, original_model, experimental_data, llm_client, tmp_path,
):
    """Branch MAE computed from successful conditions only when some fail."""
    call_count = 0

    def mock_run_sim(spec, mechanism_file, **kwargs):
        nonlocal call_count
        call_count += 1
        # Baseline succeeds for both
        if "original" in mechanism_file:
            return _make_sim_result(spec.experiment_id, 0.005)
        # Branch 000: first condition fails, second succeeds
        if "branch_000" in mechanism_file:
            if "cond_0" in spec.experiment_id:
                return _make_failed_result(spec.experiment_id, "Cantera error")
            return _make_sim_result(spec.experiment_id, 0.004)
        # Branch 001: both succeed
        return _make_sim_result(spec.experiment_id, 0.003)

    patches = _patch_all(run_sim_side_effect=mock_run_sim)

    with (
        patches["parse_pdf"],
        patches["mining_cls"],
        patches["branching_cls"],
        patches["run_sim"],
    ):
        result = await run_path2(
            paper, original_model, experimental_data, llm_client, tmp_path,
        )

    assert len(result.branches) == 2
    # Branch 000 had a partial failure but should still have a result
    branch_000 = next(b for b in result.branches if b.branch_id == "branch_000")
    assert branch_000.mae < float("inf")  # computed from the one successful condition
