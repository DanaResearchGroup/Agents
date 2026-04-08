"""Tests for src/schemas/experimental.py — instantiate each model, test validation."""

import pytest
from pydantic import ValidationError

from src.schemas.experimental import (
    # Section 1 — simulation-facing
    ReactorType,
    ObservableType,
    ExperimentalCondition,
    ExperimentalDataset,
    SimConditions,
    SimResult,
    MAEResult,
    # Section 2 — ingestion-facing
    ValidationResult,
    PageText,
    FigureCaption,
    TableConditionValue,
    TableColumn,
    TableRow,
    ParsedTable,
    PaperDocument,
    SectionSpan,
    EvidenceSnippet,
    ExperimentCandidate,
    FieldConfidence,
    ExperimentalScenario,
    NormalizedCondition,
    NormalizedComposition,
    SimulationPlan,
)


# ═══════════════════════════════════════════════════════════════════════════
# Section 1 — Simulation-facing models
# ═══════════════════════════════════════════════════════════════════════════


class TestReactorType:
    def test_enum_values(self):
        assert ReactorType.SHOCK_TUBE == "shock_tube"
        assert ReactorType.JSR == "jsr"
        assert ReactorType.PFR == "pfr"
        assert ReactorType.FLAME == "flame"
        assert ReactorType.RCM == "rcm"


class TestObservableType:
    def test_enum_values(self):
        assert ObservableType.IDT == "idt"
        assert ObservableType.FLAME_SPEED == "flame_speed"
        assert ObservableType.SPECIES_PROFILE == "species_profile"
        assert ObservableType.K_EXT == "k_ext"


class TestExperimentalCondition:
    def test_scalar_values(self):
        ec = ExperimentalCondition(
            reactor_type=ReactorType.SHOCK_TUBE,
            temperature_K=1400.0,
            pressure_atm=1.2,
            mixture={"NH3": 0.02, "O2": 0.02, "Ar": 0.96},
            observable_type=ObservableType.IDT,
            measured_value=0.00045,
            error_threshold=0.0001,
        )
        assert ec.reactor_type == ReactorType.SHOCK_TUBE
        assert ec.observable_label is None

    def test_list_values(self):
        ec = ExperimentalCondition(
            reactor_type=ReactorType.JSR,
            temperature_K=[900, 1000, 1100],
            pressure_atm=1.0,
            mixture={"NH3": 0.01, "O2": 0.0095, "N2": 0.9805},
            observable_type=ObservableType.SPECIES_PROFILE,
            observable_label="NH3",
            measured_value=[0.0095, 0.006, 0.003],
            error_threshold=0.002,
        )
        assert len(ec.temperature_K) == 3


class TestExperimentalDataset:
    def test_with_source(self):
        ds = ExperimentalDataset(
            name="test_dataset",
            conditions=[
                ExperimentalCondition(
                    reactor_type=ReactorType.SHOCK_TUBE,
                    temperature_K=1400.0,
                    pressure_atm=1.0,
                    mixture={"H2": 0.04, "O2": 0.02, "Ar": 0.94},
                    observable_type=ObservableType.IDT,
                    measured_value=0.001,
                    error_threshold=0.0005,
                ),
            ],
            source="10.1016/j.combustflame.2023.999999",
        )
        assert ds.source is not None
        assert len(ds.conditions) == 1

    def test_without_source(self):
        ds = ExperimentalDataset(name="bare", conditions=[])
        assert ds.source is None


class TestSimConditions:
    def test_basic(self):
        sc = SimConditions(
            reactor_type=ReactorType.JSR,
            T=1100.0,
            P=1.0,
            X={"NH3": 0.01, "O2": 0.0095, "N2": 0.9805},
            observable_type=ObservableType.SPECIES_PROFILE,
            observable_label="NH3",
        )
        assert sc.T == 1100.0
        assert sc.observable_label == "NH3"


class TestSimResult:
    def test_success(self):
        cond = SimConditions(
            reactor_type=ReactorType.SHOCK_TUBE,
            T=1400.0,
            P=1.2,
            X={"NH3": 0.02, "Ar": 0.98},
            observable_type=ObservableType.IDT,
        )
        sr = SimResult(
            conditions=cond,
            simulated_value=0.00042,
            units="s",
            success=True,
            error_message=None,
        )
        assert sr.success is True
        assert sr.error_message is None

    def test_failure(self):
        cond = SimConditions(
            reactor_type=ReactorType.FLAME,
            T=300.0,
            P=1.0,
            X={"CH4": 0.1, "O2": 0.2, "N2": 0.7},
            observable_type=ObservableType.FLAME_SPEED,
        )
        sr = SimResult(
            conditions=cond,
            simulated_value=0.0,
            units="m/s",
            success=False,
            error_message="Flame failed to converge",
        )
        assert sr.success is False
        assert "converge" in sr.error_message


class TestMAEResult:
    def test_passed(self):
        r = MAEResult(
            model_name="original",
            conditions_id="T1400_P1atm",
            mae=0.03,
            threshold=0.05,
            passed=True,
        )
        assert r.passed is True

    def test_failed(self):
        r = MAEResult(
            model_name="literature",
            conditions_id="T1400_P1atm",
            mae=0.08,
            threshold=0.05,
            passed=False,
        )
        assert r.passed is False


# ═══════════════════════════════════════════════════════════════════════════
# Section 2 — Ingestion-facing models
# ═══════════════════════════════════════════════════════════════════════════


class TestValidationResult:
    def test_valid(self):
        vr = ValidationResult(valid=True)
        assert vr.errors == []

    def test_invalid(self):
        vr = ValidationResult(valid=False, errors=["bad value"])
        assert len(vr.errors) == 1


class TestParserModels:
    def test_page_text(self):
        pt = PageText(page_num=1, text="Some page content")
        assert pt.page_num == 1

    def test_figure_caption(self):
        fc = FigureCaption(
            figure_id="Figure 1",
            label_type="figure",
            page_num=2,
            caption="NH3 mole fraction vs temperature",
        )
        assert fc.label_type == "figure"

    def test_section_span(self):
        ss = SectionSpan(
            name="methods", page_start=3, page_end=5, line_start=10, weight=1.0
        )
        assert ss.page_end == 5


class TestTableModels:
    def test_table_condition_value(self):
        tcv = TableConditionValue(
            mode="single", raw_text="1200 K", values=[1200.0], unit="K"
        )
        assert tcv.values == [1200.0]

    def test_table_column(self):
        tc = TableColumn(header="T (K)", role="temperature", unit="K")
        assert tc.species is None

    def test_table_row(self):
        cell = TableConditionValue(mode="single", raw_text="1 atm", values=[1.0], unit="atm")
        tr = TableRow(row_index=0, cells={"pressure": cell})
        assert "pressure" in tr.cells

    def test_parsed_table(self):
        col = TableColumn(header="T (K)", role="temperature", unit="K")
        cell = TableConditionValue(mode="single", raw_text="1200", values=[1200.0], unit="K")
        row = TableRow(row_index=0, cells={"temperature": cell})
        pt = ParsedTable(
            table_id="Table 1",
            page_num=4,
            caption="Experimental conditions",
            columns=[col],
            rows=[row],
        )
        assert len(pt.rows) == 1


class TestPaperDocument:
    def test_full_text(self):
        doc = PaperDocument(
            pdf_path="/tmp/paper.pdf",
            title="Test Paper",
            abstract="An abstract.",
            pages=[
                PageText(page_num=1, text="Page one content."),
                PageText(page_num=2, text="Page two content."),
            ],
            captions=[],
        )
        assert "Page one" in doc.full_text()
        assert "Page two" in doc.full_text()


class TestEvidenceSnippet:
    def test_basic(self):
        es = EvidenceSnippet(
            kind="temperature",
            value_text="1200 K",
            page_num=3,
            source_text="experiments at 1200 K",
            confidence=0.9,
            role="setpoint",
        )
        assert es.confidence == 0.9

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            EvidenceSnippet(
                kind="pressure",
                value_text="1 atm",
                page_num=1,
                source_text="at 1 atm",
                confidence=1.5,
            )


class TestExperimentCandidate:
    def test_basic(self):
        ev = EvidenceSnippet(
            kind="temperature",
            value_text="1200 K",
            page_num=3,
            source_text="at 1200 K",
            confidence=0.8,
        )
        ec = ExperimentCandidate(
            candidate_id="cand_01",
            experiment_family="jsr",
            confirmed_observables=["species_profile"],
            possible_observables=["conversion"],
            section="methods",
            page_start=3,
            page_end=5,
            evidence=[ev],
            confidence=0.85,
            is_primary=True,
            simulatable=True,
            planning_priority="high",
        )
        assert ec.simulatable is True
        assert ec.notes == []


class TestFieldConfidence:
    def test_overall_all_positive(self):
        fc = FieldConfidence(
            composition=0.9, temperature=0.8, pressure=0.7, family=0.9, observable=0.8
        )
        assert fc.overall == 0.7

    def test_overall_zero_field(self):
        fc = FieldConfidence(composition=0.0, temperature=0.8, pressure=0.7)
        assert fc.overall == 0.0

    def test_panel_extraction_optional(self):
        fc = FieldConfidence()
        assert fc.panel_extraction is None

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            FieldConfidence(composition=1.5)


class TestExperimentalScenario:
    def test_defaults(self):
        sc = ExperimentalScenario(
            scenario_id="sc_01",
            experiment_family="shock_tube",
            scenario_type="single_point",
            source="table",
            source_label="Table 2 row 1",
            source_page=4,
            simulatable=True,
        )
        assert sc.status == "accepted"
        assert sc.review_reasons == []
        assert sc.confidence.overall == 0.0


class TestNormalizedCondition:
    def test_setpoint(self):
        nc = NormalizedCondition(
            raw_text="1200 K",
            kind="temperature",
            role="setpoint",
            is_range=False,
            min_value=1200.0,
            max_value=1200.0,
            unit="K",
            source_page=3,
        )
        assert nc.mode == "setpoint"

    def test_range(self):
        nc = NormalizedCondition(
            raw_text="900-1300 K",
            kind="temperature",
            role="setup_range",
            is_range=True,
            min_value=900.0,
            max_value=1300.0,
            unit="K",
            source_page=3,
        )
        assert nc.mode == "range"

    def test_min_gt_max_rejected(self):
        with pytest.raises(ValidationError):
            NormalizedCondition(
                raw_text="bad",
                kind="temperature",
                role="setpoint",
                is_range=False,
                min_value=1500.0,
                max_value=1000.0,
                unit="K",
                source_page=1,
            )


class TestNormalizedComposition:
    def test_valid(self):
        nc = NormalizedComposition(
            species={"NH3": 0.01, "O2": 0.0095, "N2": 0.9805},
            balance_species="N2",
            composition_complete=True,
            raw_mentions=["0.45% NH3 in N2/O2"],
            source_pages=[3],
        )
        assert nc.composition_complete is True

    def test_negative_fraction_rejected(self):
        with pytest.raises(ValidationError):
            NormalizedComposition(
                species={"NH3": -0.01, "Ar": 1.01},
                composition_complete=False,
                raw_mentions=[],
                source_pages=[1],
            )

    def test_sum_gt_one_rejected(self):
        with pytest.raises(ValidationError):
            NormalizedComposition(
                species={"NH3": 0.5, "O2": 0.6},
                composition_complete=True,
                raw_mentions=[],
                source_pages=[1],
            )


class TestSimulationPlan:
    def test_executable(self):
        temp = NormalizedCondition(
            raw_text="1200 K",
            kind="temperature",
            role="setpoint",
            is_range=False,
            min_value=1200.0,
            max_value=1200.0,
            unit="K",
            source_page=3,
        )
        pres = NormalizedCondition(
            raw_text="1 atm",
            kind="pressure",
            role="setpoint",
            is_range=False,
            min_value=1.0,
            max_value=1.0,
            unit="atm",
            source_page=3,
        )
        comp = NormalizedComposition(
            species={"NH3": 0.01, "N2": 0.99},
            composition_complete=True,
            raw_mentions=["1% NH3"],
            source_pages=[3],
        )
        plan = SimulationPlan(
            scenario_id="sc_01",
            experiment_family="jsr",
            template_family="jsr_const_tp",
            plan_status="executable",
            temperature=temp,
            pressure=pres,
            composition=comp,
            target_observables=["species_profile"],
            auto_generation_allowed=True,
        )
        assert plan.plan_status == "executable"

    def test_executable_with_missing_fields_rejected(self):
        with pytest.raises(ValidationError):
            SimulationPlan(
                scenario_id="sc_02",
                experiment_family="jsr",
                template_family="jsr_const_tp",
                plan_status="executable",
                target_observables=["species_profile"],
                missing_fields=["temperature"],
                auto_generation_allowed=True,
            )

    def test_unsupported_must_be_blocked(self):
        with pytest.raises(ValidationError):
            SimulationPlan(
                scenario_id="sc_03",
                experiment_family="unknown",
                template_family="unsupported",
                plan_status="draft",
                target_observables=[],
                auto_generation_allowed=False,
            )

    def test_blocked_plan(self):
        plan = SimulationPlan(
            scenario_id="sc_04",
            experiment_family="unknown",
            template_family="unsupported",
            plan_status="blocked",
            target_observables=[],
            blocking_fields=["temperature", "composition"],
            auto_generation_allowed=False,
        )
        assert plan.plan_status == "blocked"
