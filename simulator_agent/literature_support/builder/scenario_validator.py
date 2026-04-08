"""Score scenario confidence and assign status: accepted / needs_review / rejected.

Runs after enrichment. Computes per-field confidence based on source quality,
then assigns a three-state status based on overall confidence.

Thresholds:
  overall >= 0.7  → accepted
  overall >= 0.3  → needs_review
  overall < 0.3   → rejected
"""

from __future__ import annotations

import re

from models import ExperimentalScenario, FieldConfidence


# Status thresholds
_ACCEPT_THRESHOLD = 0.7
_REVIEW_THRESHOLD = 0.3


def validate_scenarios(
    scenarios: list[ExperimentalScenario],
) -> list[ExperimentalScenario]:
    """Score confidence and assign status for each scenario."""
    for scenario in scenarios:
        scenario.confidence = _score_confidence(scenario)

        overall = scenario.confidence.overall
        if overall >= _ACCEPT_THRESHOLD and scenario.simulatable:
            scenario.status = "accepted"
        elif overall >= _REVIEW_THRESHOLD:
            scenario.status = "needs_review"
            scenario.review_reasons = _build_review_reasons(scenario)
        else:
            scenario.status = "rejected"
            scenario.review_reasons = _build_review_reasons(scenario)

    return scenarios


def _score_confidence(scenario: ExperimentalScenario) -> FieldConfidence:
    """Compute per-field confidence for a scenario."""
    conf = FieldConfidence()

    # Composition confidence
    if scenario.composition_text:
        comp = scenario.composition_text
        if "%" in comp and "/" in comp:
            conf.composition = 0.95  # explicit multi-species with fractions
        elif "%" in comp:
            conf.composition = 0.85  # explicit fraction, single species
        elif re.search(r'(?:in\s+(?:Ar|N2|He))', comp, re.I):
            conf.composition = 0.75  # has diluent mention
        else:
            conf.composition = 0.5   # some composition text but weak
    else:
        conf.composition = 0.0

    # Temperature confidence
    if scenario.temperature_text:
        is_from_caption = not any("temperature filled" in n for n in scenario.notes)
        is_range = bool(re.search(r'\d.*[-\u2013\u2014].*\d', scenario.temperature_text))
        if is_from_caption:
            conf.temperature = 0.90 if is_range else 0.85
        else:
            conf.temperature = 0.60  # filled from nearby text
    else:
        conf.temperature = 0.0

    # Pressure confidence
    if scenario.pressure_text:
        is_from_caption = not any("pressure filled" in n for n in scenario.notes)
        if is_from_caption:
            conf.pressure = 0.90
        elif any("methods default" in n for n in scenario.notes):
            conf.pressure = 0.50  # assumed atmospheric
        else:
            conf.pressure = 0.60  # filled from nearby text
    else:
        conf.pressure = 0.0

    # Family confidence
    if scenario.experiment_family and scenario.experiment_family != "unknown":
        conf.family = 0.90  # from candidate builder
    else:
        conf.family = 0.3

    # Observable confidence
    if scenario.observable:
        conf.observable = 0.80
    else:
        conf.observable = 0.3

    # Panel extraction confidence (if applicable)
    if scenario.scenario_type == "sweep":
        if any("panel sub-scenarios" in n for n in scenario.notes):
            conf.panel_extraction = 0.75
        elif any("could not be validated" in n for n in scenario.notes):
            conf.panel_extraction = 0.30
        else:
            conf.panel_extraction = 0.50

    return conf


def _build_review_reasons(scenario: ExperimentalScenario) -> list[str]:
    """Identify why a scenario needs review."""
    reasons = []
    conf = scenario.confidence

    if conf.temperature == 0.0:
        reasons.append("temperature missing")
    elif conf.temperature < 0.7:
        reasons.append("temperature from fallback (not caption)")

    if conf.pressure == 0.0:
        reasons.append("pressure missing")
    elif conf.pressure <= 0.5:
        reasons.append("pressure assumed from default")

    if conf.composition == 0.0:
        reasons.append("composition missing")
    elif conf.composition < 0.7:
        reasons.append("composition incomplete or ambiguous")

    if scenario.scenario_type == "sweep" and conf.panel_extraction is not None:
        if conf.panel_extraction < 0.5:
            reasons.append("multi-panel conditions could not be reliably extracted")

    if not scenario.simulatable:
        reasons.append("missing required fields for simulation")

    return reasons
