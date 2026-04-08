"""Tests for the planner module."""

from literature_support.planner.simulation_plan_builder import build_plans_from_scenarios


def test_accepted_produces_executable(accepted_scenario):
    plans = build_plans_from_scenarios([accepted_scenario])
    assert len(plans) == 1
    assert plans[0].plan_status == "executable"
    assert plans[0].auto_generation_allowed is True
    assert not plans[0].blocking_fields


def test_needs_review_produces_draft(draft_scenario):
    plans = build_plans_from_scenarios([draft_scenario])
    assert len(plans) == 1
    assert plans[0].plan_status == "draft"
    assert plans[0].auto_generation_allowed is False
    assert plans[0].blocking_fields  # has at least one blocker


def test_rejected_produces_no_plan(rejected_scenario):
    plans = build_plans_from_scenarios([rejected_scenario])
    assert len(plans) == 0


def test_unknown_family_produces_blocked(make_scenario):
    scenario = make_scenario(
        experiment_family="unknown",
        status="accepted",
    )
    plans = build_plans_from_scenarios([scenario])
    assert len(plans) == 1
    assert plans[0].plan_status == "blocked"
    assert plans[0].template_family == "unsupported"
