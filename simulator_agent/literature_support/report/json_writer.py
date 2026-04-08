"""Write JSON output files for scenarios and plans."""

from __future__ import annotations

import json
from pathlib import Path

from models import ExperimentalScenario, SimulationPlan


def write_scenarios_json(
    path: Path,
    pdf_path: str,
    title: str | None,
    primary_family: str | None,
    scenarios: list[ExperimentalScenario],
) -> None:
    _write(path, {
        "pdf_path": pdf_path,
        "title": title,
        "experiment_family": primary_family,
        "scenario_count": len(scenarios),
        "accepted": sum(1 for s in scenarios if s.status == "accepted"),
        "needs_review": sum(1 for s in scenarios if s.status == "needs_review"),
        "rejected": sum(1 for s in scenarios if s.status == "rejected"),
        "scenarios": [s.to_dict() for s in scenarios],
    })


def write_plans_json(
    path: Path,
    pdf_path: str,
    title: str | None,
    plans: list[SimulationPlan],
) -> None:
    _write(path, {
        "pdf_path": pdf_path,
        "title": title,
        "plan_count": len(plans),
        "executable": sum(1 for p in plans if p.plan_status == "executable"),
        "draft": sum(1 for p in plans if p.plan_status == "draft"),
        "plans": [p.to_dict() for p in plans],
    })


def write_debug_json(path: Path, data: dict) -> None:
    _write(path, data)


def _write(path: Path, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
