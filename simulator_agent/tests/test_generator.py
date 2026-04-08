"""Tests for the generator module."""

import pytest
from generator.plan_exporter import export_plan_json
from generator.cantera_generator import generate_cantera_script


def test_export_executable_plan(executable_plan):
    data = export_plan_json(executable_plan)
    assert data["scenario_id"] == "test_scenario"
    assert data["experiment_family"] == "shock_tube"
    assert data["template_family"] == "idt_const_uv"
    assert "temperature" in data
    assert data["temperature"]["min_K"] == 2217.0
    assert "pressure" in data
    assert data["pressure"]["min_Pa"] > 0
    assert "composition" in data
    assert "NH3" in data["composition"]


def test_export_refuses_draft(draft_plan):
    with pytest.raises(ValueError, match="non-executable"):
        export_plan_json(draft_plan)


def test_cantera_script_shock_tube(executable_plan):
    script = generate_cantera_script(executable_plan)
    assert "import cantera as ct" in script
    assert "IdealGasReactor" in script
    assert "2217" in script  # temperature should appear
    assert "NH3" in script   # species should appear


def test_cantera_refuses_draft(draft_plan):
    with pytest.raises(ValueError, match="non-executable"):
        generate_cantera_script(draft_plan)


def test_cantera_refuses_unsupported_template(make_plan):
    plan = make_plan(template_family="unsupported")
    with pytest.raises(ValueError, match="not yet supported"):
        generate_cantera_script(plan)
