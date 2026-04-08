"""Tests for the validator module."""

from literature_support.validator.plan_validator import validate_plan, validate_scenario
from models import NormalizedCondition, NormalizedComposition


def test_accepted_scenario_passes(accepted_scenario):
    result = validate_scenario(accepted_scenario)
    assert result.valid
    assert not result.errors


def test_rejected_scenario_missing_temp(rejected_scenario):
    # A rejected scenario is internally consistent — status matches fields
    result = validate_scenario(rejected_scenario)
    assert result.valid  # rejected is a valid state, not an error


def test_accepted_scenario_missing_temp_fails(make_scenario):
    # Accepted but actually missing temperature → invalid
    bad = make_scenario(status="accepted", temperature_text=None)
    result = validate_scenario(bad)
    assert not result.valid
    assert any("temperature" in e for e in result.errors)


def test_executable_plan_passes(executable_plan):
    result = validate_plan(executable_plan)
    assert result.valid
    assert not result.errors


def test_executable_plan_with_assumption_demoted(make_plan):
    # Executable plan that relies on assumed pressure → should fail validation
    plan = make_plan(
        assumptions=["pressure assumed from family default: 1 atm"],
    )
    result = validate_plan(plan)
    assert not result.valid  # demote signal
    assert any("assumption" in w.lower() for w in result.warnings)


def test_family_template_mapping(make_plan):
    # Wrong template for family → warning
    plan = make_plan(template_family="jsr_const_tp")  # shock_tube should be idt_const_uv
    result = validate_plan(plan)
    assert any("does not match" in w for w in result.warnings)


def test_temperature_outside_bounds(make_plan):
    # Temperature far outside shock_tube range (1500-4000K)
    plan = make_plan(
        temperature=NormalizedCondition(
            raw_text="500K", kind="temperature", role="scenario_field",
            is_range=False, min_value=500.0, max_value=500.0,
            unit="K", source_section=None, source_page=0,
        ),
    )
    result = validate_plan(plan)
    assert any("temperature" in w and "bounds" in w for w in result.warnings)


def test_composition_fractions_invalid(make_plan):
    # Fractions sum to > 1
    plan = make_plan(
        composition=NormalizedComposition(
            species={"NH3": 0.5, "Ar": 0.6},
            balance_species=None,
            composition_complete=False,
            raw_mentions=["bad"],
            source_pages=[],
        ),
    )
    result = validate_plan(plan)
    assert any("sum" in e for e in result.errors)


def test_executable_plan_with_missing_field_fails(make_plan):
    # Executable but missing temperature
    plan = make_plan(
        temperature=None,
        missing_fields=["temperature"],
    )
    result = validate_plan(plan)
    assert not result.valid
    assert any("missing" in e.lower() for e in result.errors)
