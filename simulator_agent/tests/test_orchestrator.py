"""Tests for the orchestrator layer.

Tests spec_builder conversions, review queue routing,
and SimulatorAgent orchestration paths.
"""

import tempfile
from pathlib import Path

import pytest
import yaml

from core.simulation_spec import SimulationSpec
from core.exp_loader import ExperimentalDataset, ExperimentalDataPoint

from models import (
    SimulationPlan, NormalizedCondition, NormalizedComposition,
)
from orchestrator.spec_builder import spec_from_dataset, spec_from_plan
from orchestrator.review import ReviewQueue, ReviewItem
from orchestrator.agent import SimulatorAgent

try:
    import cantera as ct
    HAS_CANTERA = True
except ImportError:
    HAS_CANTERA = False


# ═══════════════════════════════════════════════════════════════════════════
# spec_builder tests
# ═══════════════════════════════════════════════════════════════════════════

class TestSpecFromDataset:

    def test_passthrough(self):
        """spec_from_dataset returns the spec already in the dataset."""
        spec = SimulationSpec(
            experiment_id="ds1", reactor_type="shock_tube",
            observable="ignition_delay",
            temperature=1200, pressure=101325,
            composition={"H2": 0.02},
        )
        ds = ExperimentalDataset(
            spec=spec,
            data=[ExperimentalDataPoint(conditions={"temperature": 1200}, value=0.001)],
            metadata={"source": "test"},
        )
        result = spec_from_dataset(ds)
        assert result is spec  # same object, not a copy


class TestSpecFromPlan:

    def _make_plan(self, **overrides) -> SimulationPlan:
        defaults = dict(
            scenario_id="plan_1",
            experiment_family="shock_tube",
            template_family="idt_const_uv",
            plan_status="executable",
            temperature=NormalizedCondition(
                raw_text="1200 K", kind="temperature", role="scenario_field",
                is_range=False, min_value=1200.0, max_value=1200.0,
                unit="K", source_section=None, source_page=0,
            ),
            pressure=NormalizedCondition(
                raw_text="5 atm", kind="pressure", role="scenario_field",
                is_range=False, min_value=5.0, max_value=5.0,
                unit="atm", source_section=None, source_page=0,
            ),
            composition=NormalizedComposition(
                species={"H2": 0.02, "O2": 0.01, "AR": 0.97},
                balance_species="AR",
                composition_complete=True,
                raw_mentions=["2% H2 in Ar"],
                source_pages=[],
            ),
            residence_time=None,
            equivalence_ratio=None,
            target_observables=["ignition_delay"],
            mechanism=None,
            assumptions=[],
            warnings=[],
            missing_fields=[],
            blocking_fields=[],
            auto_generation_allowed=True,
        )
        defaults.update(overrides)
        return SimulationPlan(**defaults)

    def test_shock_tube_mapping(self):
        """idt_const_uv → shock_tube reactor_type."""
        plan = self._make_plan()
        spec = spec_from_plan(plan)

        assert spec.experiment_id == "plan_1"
        assert spec.reactor_type == "shock_tube"
        assert spec.observable == "ignition_delay"
        assert spec.temperature == 1200.0
        assert spec.pressure == pytest.approx(5 * 101325.0, rel=1e-3)
        assert spec.composition == {"H2": 0.02, "O2": 0.01, "AR": 0.97}

    def test_rcm_mapping(self):
        """idt_const_hp → rcm reactor_type."""
        plan = self._make_plan(template_family="idt_const_hp", experiment_family="rcm")
        spec = spec_from_plan(plan)
        assert spec.reactor_type == "rcm"

    def test_jsr_mapping(self):
        """jsr_const_tp → jsr reactor_type, with residence_time."""
        plan = self._make_plan(
            template_family="jsr_const_tp",
            experiment_family="jsr",
            residence_time=NormalizedCondition(
                raw_text="500 ms", kind="residence_time", role="scenario_field",
                is_range=False, min_value=500.0, max_value=500.0,
                unit="ms", source_section=None, source_page=0,
            ),
        )
        spec = spec_from_plan(plan)
        assert spec.reactor_type == "jsr"
        assert spec.residence_time == pytest.approx(0.5, rel=1e-3)  # 500 ms → 0.5 s

    def test_flame_mapping(self):
        """flame_speed → flame reactor_type."""
        plan = self._make_plan(
            template_family="flame_speed",
            experiment_family="flame",
            target_observables=["flame_speed"],
        )
        spec = spec_from_plan(plan)
        assert spec.reactor_type == "flame"
        assert spec.observable == "flame_speed"

    def test_pressure_unit_conversion(self):
        """Pressure in bar should convert to Pa."""
        plan = self._make_plan(
            pressure=NormalizedCondition(
                raw_text="2 bar", kind="pressure", role="scenario_field",
                is_range=False, min_value=2.0, max_value=2.0,
                unit="bar", source_section=None, source_page=0,
            ),
        )
        spec = spec_from_plan(plan)
        assert spec.pressure == pytest.approx(2e5, rel=1e-3)

    def test_temperature_range_creates_list(self):
        """A temperature range plan should produce temperature_list."""
        plan = self._make_plan(
            temperature=NormalizedCondition(
                raw_text="1000-1500 K", kind="temperature", role="scenario_field",
                is_range=True, min_value=1000.0, max_value=1500.0,
                unit="K", source_section=None, source_page=0,
            ),
        )
        spec = spec_from_plan(plan)
        assert spec.temperature == 1000.0
        assert spec.temperature_list == [1000.0, 1500.0]

    def test_mechanism_passthrough(self):
        """Mechanism file passed to spec_from_plan should appear in spec."""
        plan = self._make_plan()
        spec = spec_from_plan(plan, mechanism_file="/path/to/mech.yaml")
        assert spec.mechanism_file == "/path/to/mech.yaml"

    def test_non_executable_raises(self):
        """Draft plans should raise ValueError."""
        plan = self._make_plan(plan_status="draft", auto_generation_allowed=False,
                               blocking_fields=["pressure"])
        with pytest.raises(ValueError, match="non-executable"):
            spec_from_plan(plan)


# ═══════════════════════════════════════════════════════════════════════════
# ReviewQueue tests
# ═══════════════════════════════════════════════════════════════════════════

class TestReviewQueue:

    def _make_plan(self, status, scenario_id="s1", **kw):
        defaults = dict(
            scenario_id=scenario_id,
            experiment_family="shock_tube",
            template_family="idt_const_uv",
            plan_status=status,
            temperature=None, pressure=None, composition=None,
            residence_time=None, equivalence_ratio=None,
            target_observables=[],
            mechanism=None,
            assumptions=kw.get("assumptions", []),
            warnings=[],
            missing_fields=kw.get("missing_fields", []),
            blocking_fields=kw.get("blocking_fields", []),
            auto_generation_allowed=(status == "executable"),
        )
        return SimulationPlan(**defaults)

    def test_draft_goes_to_drafts(self):
        rq = ReviewQueue()
        plan = self._make_plan("draft", blocking_fields=["pressure assumed"])
        rq.add_plan(plan)

        assert len(rq.drafts) == 1
        assert len(rq.blocked) == 0
        assert rq.drafts[0].scenario_id == "s1"

    def test_blocked_goes_to_blocked(self):
        rq = ReviewQueue()
        plan = self._make_plan("blocked", blocking_fields=["template"])
        rq.add_plan(plan)

        assert len(rq.drafts) == 0
        assert len(rq.blocked) == 1

    def test_executable_ignored(self):
        rq = ReviewQueue()
        plan = self._make_plan("executable")
        rq.add_plan(plan)

        assert rq.is_empty

    def test_mixed_routing(self):
        rq = ReviewQueue()
        rq.add_plan(self._make_plan("draft", scenario_id="d1", blocking_fields=["T"]))
        rq.add_plan(self._make_plan("blocked", scenario_id="b1", blocking_fields=["template"]))
        rq.add_plan(self._make_plan("executable", scenario_id="e1"))
        rq.add_plan(self._make_plan("draft", scenario_id="d2", blocking_fields=["P"]))

        assert rq.total == 3
        assert len(rq.drafts) == 2
        assert len(rq.blocked) == 1

    def test_summary(self):
        rq = ReviewQueue()
        rq.add_plan(self._make_plan("draft", blocking_fields=["T weak"],
                                     missing_fields=["composition"]))
        s = rq.summary()
        assert s["draft_count"] == 1
        assert s["drafts"][0]["blocking_reasons"] == ["T weak"]
        assert s["drafts"][0]["missing_fields"] == ["composition"]


# ═══════════════════════════════════════════════════════════════════════════
# SimulatorAgent integration tests (require Cantera)
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not HAS_CANTERA, reason="Cantera not installed")
class TestSimulatorAgentEvaluate:

    def _write_exp_yaml(self, tmpdir: Path) -> Path:
        data = {
            "experiments": [
                {
                    "id": "st_001",
                    "reactor_type": "shock_tube",
                    "observable": "ignition_delay",
                    "temperature": 1200,
                    "pressure_atm": 5.0,
                    "composition": {"H2": 0.02, "O2": 0.01, "AR": 0.97},
                    "end_time": 0.01,
                    "data": [
                        {"temperature": 1200, "value": 0.000175},
                    ],
                    "threshold": 0.001,
                }
            ],
        }
        path = tmpdir / "exp.yaml"
        with open(path, "w") as f:
            yaml.dump(data, f)
        return path

    def test_evaluate_models_returns_report(self, tmp_path):
        """Full YAML evaluation path produces a valid EvaluationReport."""
        exp_path = self._write_exp_yaml(tmp_path)

        agent = SimulatorAgent()
        report = agent.evaluate_models(
            exp_yaml=exp_path,
            model_paths=["h2o2.yaml"],
        )

        assert len(report.experiments) == 1
        er = report.experiments[0]
        assert er.experiment_id == "st_001"
        assert er.reactor_type == "shock_tube"
        assert len(er.models) == 1
        assert er.models[0].mae is not None
        assert er.models[0].n_points == 1

    def test_evaluate_with_threshold(self, tmp_path):
        """Threshold comparison marks models as pass/fail."""
        exp_path = self._write_exp_yaml(tmp_path)

        agent = SimulatorAgent()
        report = agent.evaluate_models(
            exp_yaml=exp_path,
            model_paths=["h2o2.yaml"],
        )

        er = report.experiments[0]
        ms = er.models[0]
        # threshold is 0.001 s (1 ms) — the IDT should be close to 175 us
        # so MAE should be small enough to pass
        assert ms.passed_threshold is not None

    def test_evaluate_two_models_ranking(self, tmp_path):
        """Two models get ranked by MAE."""
        exp_path = self._write_exp_yaml(tmp_path)

        agent = SimulatorAgent()
        # Same mechanism twice — ranking should still work
        report = agent.evaluate_models(
            exp_yaml=exp_path,
            model_paths=["h2o2.yaml", "h2o2.yaml"],
        )

        er = report.experiments[0]
        assert len(er.models) == 2
        assert len(er.ranking) == 2
        assert er.winner is not None
