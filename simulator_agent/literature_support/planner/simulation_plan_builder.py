"""Build simulation plans from scenarios.

Planner policy:
  accepted scenario     → build executable plan (auto-generation safe)
  needs_review scenario → build draft plan (blocking fields marked, no auto-run)
  rejected scenario     → do not build a plan

The planner does NOT re-extract conditions. It consumes what the
scenario already has (text fields) and structures them.
"""

from __future__ import annotations

import re

from models import (
    ExperimentalScenario, SimulationPlan,
    NormalizedCondition, NormalizedComposition,
)
from literature_support.builder.condition_normalizer import (
    build_normalized_condition,
    build_normalized_composition,
)
from literature_support.families import get_family


def build_plans_from_scenarios(
    scenarios: list[ExperimentalScenario],
) -> list[SimulationPlan]:
    """Build simulation plans for accepted and needs_review scenarios.

    Rejected scenarios are skipped entirely.
    """
    plans = []
    for scenario in scenarios:
        if scenario.status == "rejected":
            continue
        plan = _build_plan_from_scenario(scenario)
        if plan:
            plans.append(plan)
    return plans


def _build_plan_from_scenario(
    scenario: ExperimentalScenario,
) -> SimulationPlan | None:
    """Build one plan from one scenario."""
    family_rules = get_family(scenario.experiment_family)
    template = family_rules.template_family

    if template == "unsupported":
        return SimulationPlan(
            scenario_id=scenario.scenario_id,
            experiment_family=scenario.experiment_family,
            template_family="unsupported",
            plan_status="blocked",
            temperature=None,
            pressure=None,
            composition=None,
            residence_time=None,
            equivalence_ratio=None,
            target_observables=[],
            mechanism=None,
            assumptions=[],
            warnings=[f"no supported template for '{scenario.experiment_family}'"],
            missing_fields=[],
            blocking_fields=["template"],
            auto_generation_allowed=False,
        )

    assumptions: list[str] = []
    warnings: list[str] = []
    missing: list[str] = []
    blocking: list[str] = []

    # Build structured conditions from scenario text fields
    temperature = _parse_condition(scenario.temperature_text, "temperature", "K")
    pressure = _parse_condition(scenario.pressure_text, "pressure", "atm")
    composition = _parse_composition(scenario.composition_text)
    residence_time = None
    equivalence_ratio = None

    if "residence_time" in family_rules.required_conditions:
        residence_time = _parse_condition(scenario.residence_time_text, "residence_time", "s")

    if scenario.equivalence_ratio_text:
        equivalence_ratio = _parse_condition(scenario.equivalence_ratio_text, "equivalence_ratio", "")

    # Track missing fields
    if temperature is None:
        missing.append("temperature")
    if pressure is None:
        missing.append("pressure")
    if composition is None:
        missing.append("composition")
    if "residence_time" in family_rules.required_conditions and residence_time is None:
        missing.append("residence_time")

    # Determine blocking fields based on scenario status + confidence
    if scenario.status == "needs_review":
        for reason in scenario.review_reasons:
            if "temperature" in reason.lower():
                blocking.append("temperature: " + reason)
            if "pressure" in reason.lower():
                blocking.append("pressure: " + reason)
            if "composition" in reason.lower():
                blocking.append("composition: " + reason)
            if "panel" in reason.lower():
                blocking.append("panel_conditions: " + reason)

    # Copy scenario notes as assumptions/warnings
    for note in scenario.notes:
        if "filled from" in note or "assumed" in note:
            assumptions.append(note)
        elif "could not" in note or "approximate" in note:
            warnings.append(note)

    # Observable
    target_obs = [scenario.observable] if scenario.observable else []

    # Plan status policy
    if scenario.status == "accepted" and not missing:
        plan_status = "executable"
        auto_allowed = True
    elif scenario.status == "needs_review" or missing:
        plan_status = "draft"
        auto_allowed = False
        if missing:
            blocking.extend(f"{f} missing" for f in missing)
    else:
        plan_status = "blocked"
        auto_allowed = False

    return SimulationPlan(
        scenario_id=scenario.scenario_id,
        experiment_family=scenario.experiment_family,
        template_family=template,
        plan_status=plan_status,
        temperature=temperature,
        pressure=pressure,
        composition=composition,
        residence_time=residence_time,
        equivalence_ratio=equivalence_ratio,
        target_observables=target_obs,
        mechanism=None,  # mechanism comes from candidate level, not scenario
        assumptions=assumptions,
        warnings=warnings,
        missing_fields=missing,
        blocking_fields=blocking,
        auto_generation_allowed=auto_allowed,
    )


# ── Helpers ──────────────────────────────────────────────────────────────

_NUM_RE = re.compile(r'(\d+(?:\.\d+)?)')

_PRESSURE_UNITS = {"atm", "bar", "kpa", "mpa", "torr", "psi", "pa"}
_KNOWN_SPECIES = {
    "NH3", "H2", "O2", "N2", "Ar", "He", "Ne", "Kr",
    "CH4", "C2H6", "C2H4", "C2H2", "CO", "CO2", "H2O",
    "NO", "NO2", "N2O", "NH2", "OH", "HCN", "N2H4",
}


def _parse_condition(
    text: str | None,
    kind: str,
    default_unit: str,
) -> NormalizedCondition | None:
    """Parse a scenario text field into a NormalizedCondition."""
    if not text:
        return None

    nums = _NUM_RE.findall(text)
    if not nums:
        return None

    values = [float(n) for n in nums]
    unit = _detect_unit(text, default_unit)
    min_val = min(values)
    max_val = max(values) if len(values) >= 2 else min_val

    return NormalizedCondition(
        raw_text=text,
        kind=kind,
        role="scenario_field",
        is_range=len(values) >= 2 and min_val != max_val,
        min_value=min_val,
        max_value=max_val,
        unit=unit,
        source_section=None,
        source_page=0,
    )


def _parse_composition(text: str | None) -> NormalizedComposition | None:
    """Parse scenario composition text into structured form."""
    if not text:
        return None

    # Collapse species spaces
    collapsed = re.sub(r'([A-Z]) (\d)', r'\1\2', text)
    collapsed = re.sub(r'(\d) ([A-Z])', r'\1\2', collapsed)

    species: dict[str, float] = {}
    balance: str | None = None

    # Match "X% Species" patterns
    for match in re.finditer(r'(\d+(?:\.\d+)?)\s*%\s*([A-Z][A-Za-z0-9]*)', collapsed):
        val = float(match.group(1)) / 100.0
        sp = match.group(2)
        if sp in _KNOWN_SPECIES:
            species[sp] = val

    # Match "in Ar" / "in N2" diluent
    diluent = re.search(r'\bin\s+(Ar|N2|He|Ne|Kr)\b', collapsed, re.I)
    if diluent:
        balance = diluent.group(1)

    # Fill balance
    total = sum(species.values())
    if balance and balance not in species and total < 1.0:
        species[balance] = round(1.0 - total, 6)
        complete = True
    else:
        complete = abs(total - 1.0) < 0.01

    return NormalizedComposition(
        species=species if species else {},
        balance_species=balance,
        composition_complete=complete,
        raw_mentions=[text],
        source_pages=[],
    )


def _detect_unit(text: str, default: str) -> str:
    """Detect physical unit from text."""
    if re.search(r'\d\s*K\b', text):
        return "K"
    if re.search(r'\d\s*(?:\u00b0\s*)?C\b', text):
        return "C"
    text_lower = text.lower()
    for unit in _PRESSURE_UNITS:
        if unit in text_lower:
            return unit
    if "ms" in text_lower:
        return "ms"
    if re.search(r'\d\s*s\b', text):
        return "s"
    return default
