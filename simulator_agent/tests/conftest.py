"""Shared fixtures for simulator agent tests."""

import sys
from pathlib import Path

import pytest

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import (
    ExperimentalScenario, SimulationPlan, FieldConfidence,
    NormalizedCondition, NormalizedComposition,
)


@pytest.fixture
def make_scenario():
    """Factory for ExperimentalScenario with sensible defaults."""
    def _make(**overrides):
        defaults = dict(
            scenario_id="test_scenario",
            experiment_family="shock_tube",
            scenario_type="single_point",
            source="figure_caption",
            source_label="Figure 1",
            source_page=3,
            temperature_text="2217K",
            pressure_text="1.22 atm",
            composition_text="0.45% NH3 in Ar",
            residence_time_text=None,
            equivalence_ratio_text=None,
            observable="species_profile",
            simulatable=True,
            status="accepted",
            confidence=FieldConfidence(
                composition=0.95, temperature=0.90, pressure=0.90,
                family=0.90, observable=0.80,
            ),
            review_reasons=[],
            notes=[],
        )
        defaults.update(overrides)
        return ExperimentalScenario(**defaults)
    return _make


@pytest.fixture
def make_plan():
    """Factory for SimulationPlan with sensible defaults."""
    def _make(**overrides):
        defaults = dict(
            scenario_id="test_scenario",
            experiment_family="shock_tube",
            template_family="idt_const_uv",
            plan_status="executable",
            temperature=NormalizedCondition(
                raw_text="2217K", kind="temperature", role="scenario_field",
                is_range=False, min_value=2217.0, max_value=2217.0,
                unit="K", source_section=None, source_page=0,
            ),
            pressure=NormalizedCondition(
                raw_text="1.22 atm", kind="pressure", role="scenario_field",
                is_range=False, min_value=1.22, max_value=1.22,
                unit="atm", source_section=None, source_page=0,
            ),
            composition=NormalizedComposition(
                species={"NH3": 0.0045, "Ar": 0.9955},
                balance_species="Ar",
                composition_complete=True,
                raw_mentions=["0.45% NH3 in Ar"],
                source_pages=[],
            ),
            residence_time=None,
            equivalence_ratio=None,
            target_observables=["species_profile"],
            mechanism=None,
            assumptions=[],
            warnings=[],
            missing_fields=[],
            blocking_fields=[],
            auto_generation_allowed=True,
        )
        defaults.update(overrides)
        return SimulationPlan(**defaults)
    return _make


@pytest.fixture
def accepted_scenario(make_scenario):
    """A valid accepted shock_tube scenario based on Figure 1 of the NH3 paper."""
    return make_scenario()


@pytest.fixture
def draft_scenario(make_scenario):
    """A needs_review scenario with fallback pressure."""
    return make_scenario(
        scenario_id="draft_scenario",
        status="needs_review",
        pressure_text="1 atm",
        confidence=FieldConfidence(
            composition=0.85, temperature=0.60, pressure=0.50,
            family=0.90, observable=0.80,
        ),
        review_reasons=["temperature from fallback (not caption)", "pressure assumed from default"],
        notes=["pressure filled from methods default: 1 atm"],
    )


@pytest.fixture
def rejected_scenario(make_scenario):
    """A rejected scenario with no temperature."""
    return make_scenario(
        scenario_id="rejected_scenario",
        status="rejected",
        temperature_text=None,
        simulatable=False,
        confidence=FieldConfidence(
            composition=0.75, temperature=0.0, pressure=0.90,
            family=0.90, observable=0.30,
        ),
    )


@pytest.fixture
def executable_plan(make_plan):
    """A valid executable shock_tube plan."""
    return make_plan()


@pytest.fixture
def draft_plan(make_plan):
    """A draft plan with blocking fields."""
    return make_plan(
        scenario_id="draft_plan",
        plan_status="draft",
        auto_generation_allowed=False,
        blocking_fields=["pressure: pressure assumed from default"],
        assumptions=["pressure assumed from family default: 1 atm"],
    )
