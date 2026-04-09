"""End-to-end integration tests with real Cantera, mocked LLM calls.

These tests exercise the full orchestrator pipeline against real mechanism
files and real simulation, but mock all LLM calls and external I/O
(paper ingestion, PDF parsing, mechanism conversion).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from src.agents.validators import ModelIsolationViolation
from src.schemas.experimental import (
    BranchResult,
    CandidateReaction,
    ConversionResult,
    ExperimentalCondition,
    ExperimentalDataset,
    ExtractionResult,
    FigureCaption,
    FigureClassification,
    FluxConditions,
    FluxInterpretation,
    FluxPathway,
    MAEResult,
    ObservableType,
    PaperDocument,
    PaperSource,
    Path1Results,
    Path2Results,
    ReactorType,
    RunConfig,
    SimConditions,
)
from src.schemas.ingestion import PaperRecord

# ── Fixture paths ────────────────────────────────────────────────────────────

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
SIMPLE_MODEL = FIXTURES / "models" / "h2o2_simple.yaml"
LIT_MODEL = FIXTURES / "models" / "h2o2_literature.yaml"
EXP_DATA = FIXTURES / "experimental" / "h2o2_idt.yaml"
PAPER_JSON = FIXTURES / "papers" / "h2o2_paper.json"


def _load_paper() -> PaperRecord:
    """Load the h2o2 paper fixture."""
    with open(PAPER_JSON) as f:
        return PaperRecord.model_validate(json.load(f))


def _load_dataset() -> ExperimentalDataset:
    """Load the experimental dataset fixture."""
    with open(EXP_DATA) as f:
        return ExperimentalDataset.model_validate(yaml.safe_load(f))


# ── Shared mock helpers ──────────────────────────────────────────────────────


def _mock_paper_document() -> PaperDocument:
    """A minimal PaperDocument with one figure caption for reaction mining."""
    return PaperDocument(
        pdf_path="tests/fixtures/papers/h2o2_paper.pdf",
        title="H2/O2 shock tube study",
        pages=[],
        captions=[
            FigureCaption(
                figure_id="fig1",
                label_type="figure",
                page_num=1,
                caption="Flux diagram for H2/O2 at 1200K showing dominant pathways",
            ),
        ],
        tables=[],
    )


def _mock_llm_client() -> MagicMock:
    """Create a mock LLMClient that returns appropriate Pydantic models."""
    client = MagicMock()
    client.complete = AsyncMock()
    return client


def _make_config(
    tmp_path: Path,
    *,
    run_path1: bool = True,
    run_path2: bool = True,
    paper_mode: str = "upload",
    paper_value: str = "dummy.pdf",
) -> RunConfig:
    """Build a RunConfig pointing at our fixture files."""
    return RunConfig(
        original_model=SIMPLE_MODEL,
        experimental_data=EXP_DATA,
        paper_source=PaperSource(mode=paper_mode, value=paper_value),
        run_path1=run_path1,
        run_path2=run_path2,
        output_dir=tmp_path / "reports",
        llm_config=Path("config/llm_config.yaml"),
    )


# ── Shared mock context manager ─────────────────────────────────────────────


def _patch_ingestion_and_conversion(
    paper: PaperRecord,
    document: PaperDocument,
    lit_model_path: Path = LIT_MODEL,
):
    """Return a stack of patches for paper ingestion, PDF parsing, and conversion.

    Mocks:
      - Orchestrator.ingest_paper -> returns paper
      - parse_pdf (in path1 and path2) -> returns document
      - ChemkinConverter.convert -> success pointing to lit_model_path
      - ChemkinConverter.extract_rates -> empty dict
      - PaperExtractionPipeline.extract -> empty list (force LLM fallback)
    """
    patches = [
        patch(
            "src.orchestrator.Orchestrator.ingest_paper",
            new_callable=AsyncMock,
            return_value=paper,
        ),
        patch(
            "src.pipelines.path1.parse_pdf",
            return_value=document,
        ),
        patch(
            "src.pipelines.path2.parse_pdf",
            return_value=document,
        ),
        patch(
            "src.agents.conversion.ChemkinConverter.convert",
            new_callable=AsyncMock,
            return_value=ConversionResult(
                success=True,
                output_path=lit_model_path,
                errors=[],
                warnings=[],
                attempts=1,
            ),
        ),
        patch(
            "src.agents.conversion.ChemkinConverter.extract_rates",
            return_value={},
        ),
        patch(
            "src.pipelines.path1.PaperExtractionPipeline.extract",
            return_value=[],  # Force LLM fallback path
        ),
        # Mock _find_mechanism: SI dir has .yaml not .inp/.dat Chemkin files
        patch(
            "src.pipelines.path1._find_mechanism",
            return_value=lit_model_path,
        ),
    ]
    return patches


# ═════════════════════════════════════════════════════════════════════════════
# Tests
# ═══��═════════════════════════════════════════════════════════════════════════


class TestPath1EndToEnd:
    """Path 1 integration: literature model evaluation with real Cantera."""

    @pytest.mark.asyncio
    async def test_path1_end_to_end(self, tmp_path: Path):
        """Full Path 1 run: mock LLM, real simulation, real MAE."""
        paper = _load_paper()
        paper.si_path = str(FIXTURES / "models")  # So _find_mechanism can find it
        document = _mock_paper_document()

        # The condition extraction agent returns one condition matching our fixture
        mock_conditions = [
            SimConditions(
                reactor_type=ReactorType.SHOCK_TUBE,
                T=1200.0,
                P=1.0,
                X={"H2": 0.1, "O2": 0.05, "AR": 0.85},
                observable_type=ObservableType.IDT,
            ),
        ]

        config = _make_config(tmp_path, run_path1=True, run_path2=False)

        patches = _patch_ingestion_and_conversion(paper, document)

        # Mock extract_conditions (ConditionReasoningAgent)
        patches.append(
            patch(
                "src.pipelines.path1.extract_conditions",
                new_callable=AsyncMock,
                return_value=mock_conditions,
            ),
        )
        patches.append(
            patch("src.pipelines.path1._get_model_species", return_value=[]),
        )

        # Mock LLMClient constructor (orchestrator creates one)
        patches.append(
            patch("src.orchestrator.LLMClient"),
        )

        for p in patches:
            p.start()

        try:
            from src.orchestrator import Orchestrator

            orch = Orchestrator(config)
            report_path = await orch.run()

            # ── Assertions ───────────────────────────────────────────────
            assert report_path.exists(), "Report file should exist"
            report = yaml.safe_load(report_path.read_text())

            assert "path1" in report, "Report must have path1 key"
            assert "path2" not in report, "Report must not have path2 key"

            p1 = report["path1"]
            assert p1["conditions_tested"] >= 1
            assert isinstance(p1["overall_literature_better"], bool)

            # Check individual results
            assert len(p1["results"]) >= 1
            for row in p1["results"]:
                assert "original_mae" in row
                assert "literature_mae" in row
                # MAE values should be formatted as strings with 4 decimal places
                assert "." in row["original_mae"]
                assert "." in row["literature_mae"]
                orig_mae = float(row["original_mae"])
                lit_mae = float(row["literature_mae"])
                assert orig_mae >= 0, "MAE must be non-negative"
                assert lit_mae >= 0, "MAE must be non-negative"
        finally:
            for p in patches:
                p.stop()

    @pytest.mark.asyncio
    async def test_path1_isolation_enforced(self, tmp_path: Path):
        """Using the same model for original and literature must raise."""
        paper = _load_paper()
        paper.si_path = str(FIXTURES / "models")
        document = _mock_paper_document()

        config = _make_config(tmp_path, run_path1=True, run_path2=False)

        # Mock converter to return the SAME model as original
        patches = _patch_ingestion_and_conversion(
            paper, document, lit_model_path=SIMPLE_MODEL
        )

        # Mock condition extraction (won't be reached, but needed for setup)
        patches.append(
            patch(
                "src.pipelines.path1.extract_conditions",
                new_callable=AsyncMock,
                return_value=[],
            ),
        )
        patches.append(
            patch("src.pipelines.path1._get_model_species", return_value=[]),
        )
        patches.append(
            patch("src.orchestrator.LLMClient"),
        )

        for p in patches:
            p.start()

        try:
            from src.orchestrator import Orchestrator

            orch = Orchestrator(config)
            with pytest.raises(ModelIsolationViolation):
                await orch.run()

            # No report file should exist
            report_dir = tmp_path / "reports"
            if report_dir.exists():
                yamls = list(report_dir.glob("report_*.yaml"))
                assert len(yamls) == 0, "No report should be written on isolation failure"
        finally:
            for p in patches:
                p.stop()


class TestPath2EndToEnd:
    """Path 2 integration: targeted model improvements with real Cantera."""

    @pytest.mark.asyncio
    async def test_path2_end_to_end(self, tmp_path: Path):
        """Full Path 2 run: mock reaction mining, real simulation."""
        paper = _load_paper()
        document = _mock_paper_document()

        config = _make_config(tmp_path, run_path1=False, run_path2=True)

        patches = _patch_ingestion_and_conversion(paper, document)

        # Mock LLMClient
        patches.append(patch("src.orchestrator.LLMClient"))

        # Mock ReactionMiningAgent methods
        mock_classification = FigureClassification(
            label="flux_diagram",
            confidence=0.9,
        )
        mock_flux = FluxInterpretation(
            figure_id="fig1",
            dominant_pathways=[
                FluxPathway(
                    from_species="H",
                    to_species="OH",
                    rank=1,
                    percentage=45.0,
                    evidence="Primary chain branching pathway",
                ),
            ],
            conditions=FluxConditions(temperature="1200 K", pressure="1 atm"),
            confidence=0.8,
        )

        patches.append(
            patch(
                "src.pipelines.path2.ReactionMiningAgent.classify_figure",
                new_callable=AsyncMock,
                return_value=mock_classification,
            ),
        )
        patches.append(
            patch(
                "src.pipelines.path2.ReactionMiningAgent.interpret_flux",
                new_callable=AsyncMock,
                return_value=mock_flux,
            ),
        )

        # Mock ModelBranchingAgent to create a real branch model
        # We need rate_params on the candidate for branching to work,
        # so we also mock the candidate enrichment step
        def _mock_create_branch(reactions, branch_id):
            """Create a branch that is a copy of the original model with a dummy reaction."""
            from src.schemas.experimental import ModelBranch

            branch_path = tmp_path / "reports" / f"{branch_id}.yaml"
            branch_path.parent.mkdir(parents=True, exist_ok=True)

            # Copy original model as the branch (superset guaranteed)
            import shutil
            shutil.copy2(SIMPLE_MODEL, branch_path)

            return ModelBranch(
                branch_id=branch_id,
                reactions_added=reactions,
                model_path=branch_path,
            )

        patches.append(
            patch(
                "src.pipelines.path2.ModelBranchingAgent.create_branch",
                side_effect=_mock_create_branch,
            ),
        )

        for p in patches:
            p.start()

        try:
            from src.orchestrator import Orchestrator

            orch = Orchestrator(config)
            report_path = await orch.run()

            # ── Assertions ───────────────────────────────────────────────
            assert report_path.exists(), "Report file should exist"
            report = yaml.safe_load(report_path.read_text())

            assert "path2" in report, "Report must have path2 key"
            assert "path1" not in report, "Report must not have path1 key"

            p2 = report["path2"]
            baseline = float(p2["baseline_mae"])
            assert baseline > 0, "Baseline MAE must be positive"

            assert len(p2["branches"]) >= 1, "Must have at least one branch"
            for branch in p2["branches"]:
                assert "branch_id" in branch
                assert "mae" in branch
                assert "improvement" in branch
                assert "delta_mae" in branch
                branch_mae = float(branch["mae"])
                assert branch_mae >= 0 or branch_mae == float("inf")
        finally:
            for p in patches:
                p.stop()


class TestReportStructure:
    """Validate the output report matches ARCHITECTURE.md section 3.7."""

    @pytest.mark.asyncio
    async def test_report_structure(self, tmp_path: Path):
        """Run both paths, verify all required report keys present."""
        paper = _load_paper()
        paper.si_path = str(FIXTURES / "models")
        document = _mock_paper_document()

        config = _make_config(tmp_path, run_path1=True, run_path2=True)

        patches = _patch_ingestion_and_conversion(paper, document)

        # Path 1 mocks
        mock_conditions = [
            SimConditions(
                reactor_type=ReactorType.SHOCK_TUBE,
                T=1200.0,
                P=1.0,
                X={"H2": 0.1, "O2": 0.05, "AR": 0.85},
                observable_type=ObservableType.IDT,
            ),
        ]
        patches.append(
            patch(
                "src.pipelines.path1.extract_conditions",
                new_callable=AsyncMock,
                return_value=mock_conditions,
            ),
        )
        patches.append(
            patch("src.pipelines.path1._get_model_species", return_value=[]),
        )

        # Path 2 mocks
        mock_classification = FigureClassification(label="flux_diagram", confidence=0.9)
        mock_flux = FluxInterpretation(
            figure_id="fig1",
            dominant_pathways=[
                FluxPathway(
                    from_species="H", to_species="OH", rank=1,
                    percentage=45.0, evidence="chain branching",
                ),
            ],
            conditions=FluxConditions(temperature="1200 K", pressure="1 atm"),
            confidence=0.8,
        )
        patches.append(
            patch(
                "src.pipelines.path2.ReactionMiningAgent.classify_figure",
                new_callable=AsyncMock,
                return_value=mock_classification,
            ),
        )
        patches.append(
            patch(
                "src.pipelines.path2.ReactionMiningAgent.interpret_flux",
                new_callable=AsyncMock,
                return_value=mock_flux,
            ),
        )

        def _mock_create_branch(reactions, branch_id):
            from src.schemas.experimental import ModelBranch
            import shutil
            branch_path = tmp_path / "reports" / f"{branch_id}.yaml"
            branch_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(SIMPLE_MODEL, branch_path)
            return ModelBranch(
                branch_id=branch_id,
                reactions_added=reactions,
                model_path=branch_path,
            )

        patches.append(
            patch(
                "src.pipelines.path2.ModelBranchingAgent.create_branch",
                side_effect=_mock_create_branch,
            ),
        )
        patches.append(patch("src.orchestrator.LLMClient"))

        for p in patches:
            p.start()

        try:
            from src.orchestrator import Orchestrator

            orch = Orchestrator(config)
            report_path = await orch.run()

            assert report_path.exists()
            report = yaml.safe_load(report_path.read_text())

            # ── Top-level keys (ARCHITECTURE.md 3.7) ─────────────────────
            assert "run_id" in report
            assert "original_model" in report
            assert "experimental_data" in report

            # ── Path 1 structure ─────────────────────────────────────────
            assert "path1" in report
            p1 = report["path1"]
            assert "literature_model" in p1
            assert "source_doi" in p1
            assert "conditions_tested" in p1
            assert "results" in p1
            assert "overall_literature_better" in p1

            for row in p1["results"]:
                assert "condition_id" in row
                # MAE formatted to 4 decimal places
                if "original_mae" in row:
                    parts = row["original_mae"].split(".")
                    assert len(parts[1]) == 4, f"original_mae not 4dp: {row['original_mae']}"
                if "literature_mae" in row:
                    parts = row["literature_mae"].split(".")
                    assert len(parts[1]) == 4, f"literature_mae not 4dp: {row['literature_mae']}"

            # ── Path 2 structure ─────────────────────────────────────────
            assert "path2" in report
            p2 = report["path2"]
            assert "baseline_mae" in p2
            assert "branches" in p2
            assert "best_branch" in p2

            for branch in p2["branches"]:
                assert "branch_id" in branch
                assert "reactions_added" in branch
                assert "mae" in branch
                assert "improvement" in branch
                assert "delta_mae" in branch
                # MAE formatted to 4dp
                parts = branch["mae"].split(".")
                assert len(parts[1]) == 4, f"branch mae not 4dp: {branch['mae']}"
        finally:
            for p in patches:
                p.stop()


class TestUploadIngestion:
    """Test the upload paper source mode reaches simulation."""

    @pytest.mark.asyncio
    async def test_upload_ingestion(self, tmp_path: Path):
        """Upload mode: verify ingestion completes and simulation is reached."""
        paper = _load_paper()
        paper.si_path = str(FIXTURES / "models")
        document = _mock_paper_document()

        config = _make_config(
            tmp_path,
            run_path1=True,
            run_path2=False,
            paper_mode="upload",
            paper_value=str(PAPER_JSON),  # Any existing file works
        )

        patches = _patch_ingestion_and_conversion(paper, document)

        mock_conditions = [
            SimConditions(
                reactor_type=ReactorType.SHOCK_TUBE,
                T=1200.0,
                P=1.0,
                X={"H2": 0.1, "O2": 0.05, "AR": 0.85},
                observable_type=ObservableType.IDT,
            ),
        ]
        patches.append(
            patch(
                "src.pipelines.path1.extract_conditions",
                new_callable=AsyncMock,
                return_value=mock_conditions,
            ),
        )
        patches.append(
            patch("src.pipelines.path1._get_model_species", return_value=[]),
        )
        patches.append(patch("src.orchestrator.LLMClient"))

        for p in patches:
            p.start()

        try:
            from src.orchestrator import Orchestrator

            orch = Orchestrator(config)
            report_path = await orch.run()

            # The key assertion: ingestion completed and simulation ran
            assert report_path.exists(), "Upload ingestion should complete to report"
            report = yaml.safe_load(report_path.read_text())
            assert "path1" in report
            assert report["path1"]["conditions_tested"] >= 1
        finally:
            for p in patches:
                p.stop()
