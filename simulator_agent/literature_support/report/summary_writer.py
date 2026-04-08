"""Generate human-readable run_summary.md."""

from __future__ import annotations

from pathlib import Path

from models import (
    PaperDocument, ExperimentCandidate, ExperimentalScenario, SimulationPlan,
)


def write_summary(
    path: Path,
    document: PaperDocument,
    candidates: list[ExperimentCandidate],
    scenarios: list[ExperimentalScenario],
    plans: list[SimulationPlan],
) -> None:
    accepted = [s for s in scenarios if s.status == "accepted"]
    review = [s for s in scenarios if s.status == "needs_review"]
    rejected = [s for s in scenarios if s.status == "rejected"]
    executable = [p for p in plans if p.plan_status == "executable"]
    drafts = [p for p in plans if p.plan_status == "draft"]

    lines = [
        f"# Simulator Agent Run Summary",
        f"",
        f"**Paper:** {document.title or '(unknown)'}",
        f"**Pages:** {len(document.pages)}",
        f"**Captions:** {len(document.captions)}",
        f"",
        f"## Experiment Candidates",
        f"",
    ]
    for c in candidates:
        tag = "primary" if c.is_primary else "secondary"
        lines.append(f"- **{c.experiment_family}** ({tag}, {c.planning_priority} priority)")

    lines += [
        f"",
        f"## Scenarios",
        f"",
        f"| Status | Count |",
        f"|--------|-------|",
        f"| Accepted | {len(accepted)} |",
        f"| Needs Review | {len(review)} |",
        f"| Rejected | {len(rejected)} |",
        f"| **Total** | **{len(scenarios)}** |",
        f"",
    ]

    if accepted:
        lines.append(f"### Accepted Scenarios")
        lines.append(f"")
        for s in accepted:
            lines.append(
                f"- **{s.source_label}**: "
                f"{s.composition_text or '?'} @ "
                f"{s.temperature_text or '?'}, "
                f"{s.pressure_text or '?'}"
            )
        lines.append(f"")

    if review:
        lines.append(f"### Needs Review")
        lines.append(f"")
        for s in review:
            reasons = ", ".join(s.review_reasons[:2]) if s.review_reasons else "unspecified"
            lines.append(f"- **{s.source_label}**: {reasons}")
        lines.append(f"")

    lines += [
        f"## Simulation Plans",
        f"",
        f"| Status | Count |",
        f"|--------|-------|",
        f"| Executable | {len(executable)} |",
        f"| Draft | {len(drafts)} |",
        f"| **Total** | **{len(plans)}** |",
        f"",
    ]

    if executable:
        lines.append(f"### Executable Plans")
        lines.append(f"")
        for p in executable:
            t = f"{p.temperature.min_value}-{p.temperature.max_value} {p.temperature.unit}" if p.temperature else "?"
            lines.append(f"- **{p.scenario_id}**: {p.template_family} | T={t}")
        lines.append(f"")

    if drafts:
        lines.append(f"### Draft Plans (blocked)")
        lines.append(f"")
        for p in drafts:
            blockers = ", ".join(p.blocking_fields[:2]) if p.blocking_fields else "unspecified"
            lines.append(f"- **{p.scenario_id}**: {blockers}")
        lines.append(f"")

    with open(path, "w") as f:
        f.write("\n".join(lines))
