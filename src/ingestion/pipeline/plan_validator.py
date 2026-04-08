"""Family-aware validation for scenarios and simulation plans.

All family-specific logic accessed through the families/ registry.
No hardcoded family knowledge here.
"""

from __future__ import annotations

from src.schemas.experimental import (
    ExperimentalScenario, SimulationPlan, ValidationResult,
)
from src.ingestion.pipeline.families import get_family


def validate_scenario(scenario: ExperimentalScenario) -> ValidationResult:
    """Validate a scenario for internal consistency and completeness."""
    errors: list[str] = []
    warnings: list[str] = []

    # 1. Status/field consistency
    if scenario.status == "accepted":
        if not scenario.temperature_text:
            errors.append("accepted scenario missing temperature")
        if not scenario.pressure_text:
            errors.append("accepted scenario missing pressure")
        if not scenario.composition_text:
            errors.append("accepted scenario missing composition")
        if not scenario.simulatable:
            errors.append("accepted scenario has simulatable=False")

    # 2. Confidence consistency
    conf_result = scenario.confidence.validate()
    errors.extend(conf_result.errors)

    # 3. Family known
    family_rules = get_family(scenario.experiment_family)
    if family_rules.template_family == "unsupported":
        warnings.append(f"experiment family '{scenario.experiment_family}' is unsupported")

    return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)


def validate_plan(plan: SimulationPlan) -> ValidationResult:
    """Validate a simulation plan for correctness, completeness, and plausibility.

    Checks:
    1. Internal consistency (via plan.validate())
    2. Family/template compatibility
    3. Physical plausibility of conditions
    4. Assumption audit (demote executable if assumptions are too weak)
    """
    # Start with the plan's own structural validation
    result = plan.validate()
    errors = list(result.errors)
    warnings = list(result.warnings)

    family_rules = get_family(plan.experiment_family)

    # Family/template compatibility
    if plan.template_family != "unsupported":
        expected = family_rules.template_family
        if plan.template_family != expected:
            warnings.append(
                f"template '{plan.template_family}' does not match "
                f"family default '{expected}'"
            )

    # Required conditions from family
    if plan.plan_status == "executable":
        required = family_rules.required_conditions
        cond_map = {
            "temperature": plan.temperature,
            "pressure": plan.pressure,
            "composition": plan.composition,
            "residence_time": plan.residence_time,
        }
        for req in required:
            if cond_map.get(req) is None:
                errors.append(f"executable plan missing required condition: {req}")

    # Physical plausibility — temperature bounds from family
    if plan.temperature is not None:
        t_lo, t_hi = family_rules.temperature_bounds
        for val in (plan.temperature.min_value, plan.temperature.max_value):
            if not (t_lo <= val <= t_hi):
                warnings.append(
                    f"temperature {val} K outside family bounds [{t_lo}, {t_hi}]"
                )

    # Pressure plausibility
    if plan.pressure is not None:
        if plan.pressure.min_value <= 0:
            errors.append(f"non-positive pressure: {plan.pressure.min_value}")

    # Composition plausibility
    if plan.composition is not None:
        comp_result = plan.composition.validate()
        errors.extend(comp_result.errors)
        warnings.extend(comp_result.warnings)

    # Assumption audit: executable plan relying on assumed pressure → demote
    demote = False
    if plan.plan_status == "executable":
        for assumption in plan.assumptions:
            if "assumed" in assumption.lower() or "default" in assumption.lower():
                demote = True
                warnings.append(
                    f"executable plan relies on assumption: '{assumption}' — "
                    f"should be demoted to draft"
                )
                break

    return ValidationResult(
        valid=len(errors) == 0 and not demote,
        errors=errors,
        warnings=warnings,
    )
