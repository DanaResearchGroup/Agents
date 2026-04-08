"""Review queue for plans that cannot proceed automatically.

Draft plans need human or LLM review before simulation.
Blocked plans are recorded but skipped.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from models import SimulationPlan


@dataclass
class ReviewItem:
    """One plan that needs review before it can be simulated."""
    plan: SimulationPlan
    blocking_reasons: list[str]
    missing_fields: list[str]
    assumptions: list[str]

    @property
    def scenario_id(self) -> str:
        return self.plan.scenario_id


@dataclass
class ReviewQueue:
    """Collects plans that could not proceed automatically."""
    drafts: list[ReviewItem] = field(default_factory=list)
    blocked: list[ReviewItem] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.drafts) + len(self.blocked)

    @property
    def is_empty(self) -> bool:
        return self.total == 0

    def add_plan(self, plan: SimulationPlan) -> None:
        """Route a plan into the appropriate queue based on its status."""
        item = ReviewItem(
            plan=plan,
            blocking_reasons=list(plan.blocking_fields),
            missing_fields=list(plan.missing_fields),
            assumptions=list(plan.assumptions),
        )
        if plan.plan_status == "draft":
            self.drafts.append(item)
        elif plan.plan_status == "blocked":
            self.blocked.append(item)

    def summary(self) -> dict:
        """Return a structured summary of the queue."""
        return {
            "draft_count": len(self.drafts),
            "blocked_count": len(self.blocked),
            "drafts": [
                {
                    "scenario_id": d.scenario_id,
                    "blocking_reasons": d.blocking_reasons,
                    "missing_fields": d.missing_fields,
                }
                for d in self.drafts
            ],
            "blocked": [
                {
                    "scenario_id": b.scenario_id,
                    "blocking_reasons": b.blocking_reasons,
                }
                for b in self.blocked
            ],
        }
