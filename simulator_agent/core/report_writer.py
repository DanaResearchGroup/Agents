"""Structured evaluation report writer.

Produces machine-readable JSON reports with:
  - experiment metadata
  - per-model MAE scores
  - threshold comparison
  - ranking (when multiple models)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class ModelScore:
    """Evaluation result for one model on one experiment."""
    model_name: str
    mechanism_file: str
    mae: float | None
    relative_mae: float | None = None
    n_points: int = 0
    passed_threshold: bool | None = None
    error: str | None = None


@dataclass
class ExperimentReport:
    """Evaluation report for one experiment across models."""
    experiment_id: str
    reactor_type: str
    observable: str
    observable_species: str | None = None
    models: list[ModelScore] = field(default_factory=list)
    threshold: float | None = None
    winner: str | None = None
    ranking: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationReport:
    """Top-level evaluation report across all experiments."""
    experiments: list[ExperimentReport] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


def build_experiment_report(
    experiment_id: str,
    reactor_type: str,
    observable: str,
    scores: list[ModelScore],
    threshold: float | None = None,
    observable_species: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> ExperimentReport:
    """Build an experiment report with ranking and threshold comparison."""
    report = ExperimentReport(
        experiment_id=experiment_id,
        reactor_type=reactor_type,
        observable=observable,
        observable_species=observable_species,
        models=scores,
        threshold=threshold,
        metadata=metadata or {},
    )

    # Apply threshold
    if threshold is not None:
        for score in scores:
            if score.mae is not None:
                score.passed_threshold = score.mae <= threshold

    # Rank by MAE (lower is better), excluding errors
    valid = [s for s in scores if s.mae is not None]
    valid.sort(key=lambda s: s.mae)
    report.ranking = [s.model_name for s in valid]
    if valid:
        report.winner = valid[0].model_name

    return report


def build_evaluation_report(
    experiment_reports: list[ExperimentReport],
) -> EvaluationReport:
    """Build a top-level report summarizing all experiments."""
    report = EvaluationReport(experiments=experiment_reports)

    # Summary stats
    total = len(experiment_reports)
    with_winner = [r for r in experiment_reports if r.winner]
    win_counts: dict[str, int] = {}
    for r in with_winner:
        win_counts[r.winner] = win_counts.get(r.winner, 0) + 1

    report.summary = {
        "total_experiments": total,
        "experiments_evaluated": len(with_winner),
        "win_counts": win_counts,
    }

    return report


def write_report_json(report: EvaluationReport, path: str | Path) -> None:
    """Write the evaluation report as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = _serialize(report)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _serialize(obj: Any) -> Any:
    """Recursively serialize dataclasses to dicts."""
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _serialize(v) for k, v in obj.__dict__.items()}
    if isinstance(obj, list):
        return [_serialize(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    return obj
