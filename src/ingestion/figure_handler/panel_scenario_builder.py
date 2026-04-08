"""Convert panel-level conditions into sub-scenarios."""

from __future__ import annotations

from src.schemas.experimental import ExperimentalScenario
from .panel_text_extractor import PanelCondition


def build_panel_scenarios(
    parent_scenario: ExperimentalScenario,
    panel_conditions: list[PanelCondition],
) -> list[ExperimentalScenario]:
    """Split a multi-panel scenario into per-panel sub-scenarios.

    Each sub-scenario inherits the parent's composition and experiment family,
    but gets its own T and P from the OCR-extracted panel conditions.
    """
    sub_scenarios: list[ExperimentalScenario] = []

    for panel in panel_conditions:
        label = panel.panel_label or "?"
        scenario_id = f"{parent_scenario.scenario_id}_panel_{label}"

        has_pressure = bool(panel.pressure_text or parent_scenario.pressure_text)
        has_composition = bool(parent_scenario.composition_text)

        # Fix 5: panel sub-scenarios are non-simulatable by default
        # Only promote to simulatable if all conditions are clean
        simulatable = (
            panel.temperature_text is not None
            and panel.panel_label is not None  # must have valid label
            and has_pressure
            and has_composition
        )

        sub = ExperimentalScenario(
            scenario_id=scenario_id,
            experiment_family=parent_scenario.experiment_family,
            scenario_type="single_point",
            source="figure_caption",
            source_label=f"{parent_scenario.source_label} panel {label}",
            source_page=parent_scenario.source_page,
            temperature_text=panel.temperature_text,
            pressure_text=panel.pressure_text or parent_scenario.pressure_text,
            composition_text=parent_scenario.composition_text,
            residence_time_text=parent_scenario.residence_time_text,
            equivalence_ratio_text=parent_scenario.equivalence_ratio_text,
            observable=parent_scenario.observable,
            simulatable=simulatable,
            notes=[
                f"sub-scenario from {parent_scenario.source_label} panel ({label})",
                f"T from OCR: {panel.temperature_text}",
                f"P from OCR: {panel.pressure_text}" if panel.pressure_text else "P inherited from parent",
                "OCR-derived — verify before simulation" if simulatable else "OCR-derived — not validated",
            ],
        )
        sub_scenarios.append(sub)

    return sub_scenarios
