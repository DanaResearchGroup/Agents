"""Thin coordinator that runs the full deterministic extraction pipeline.

Pipeline stages:
  1. extract_evidence     — regex-based evidence from parsed PDF
  2. build_candidates     — group evidence into experiment candidates
  3. extract_scenarios    — figure captions, methods, tables → scenarios
  4. enrich_scenarios     — fill missing fields from context
  5. validate_scenarios   — confidence scoring + status assignment
  6. validate_plan        — family-aware plan validation
  7. build_plans          — scenarios → simulation plans

Returns only executable plans (plan_status == "executable").
"""

from __future__ import annotations

import logging

from src.schemas.experimental import EvidenceSnippet, PaperDocument, SimulationPlan

from src.ingestion.pipeline.evidence_extractor import extract_evidence
from src.ingestion.pipeline.builder.candidate_builder import build_experiment_candidates
from src.ingestion.pipeline.builder.scenario_extractor import extract_scenarios
from src.ingestion.pipeline.builder.scenario_enricher import enrich_scenarios
from src.ingestion.pipeline.builder.scenario_validator import validate_scenarios
from src.ingestion.pipeline.plan_validator import validate_plan
from src.ingestion.pipeline.simulation_plan_builder import build_plans_from_scenarios

logger = logging.getLogger(__name__)


class PaperExtractionPipeline:
    """Runs the full deterministic pipeline on a PaperDocument."""

    def __init__(self) -> None:
        self.last_evidence: list[EvidenceSnippet] = []

    def extract(self, document: PaperDocument) -> list[SimulationPlan]:
        """Run pipeline and return executable plans only.

        Never raises — returns empty list on any failure.
        """
        try:
            return self._run(document)
        except Exception:
            logger.exception("Deterministic pipeline failed")
            return []

    def _run(self, document: PaperDocument) -> list[SimulationPlan]:
        # Stage 1: evidence extraction
        evidence = extract_evidence(document)
        self.last_evidence = evidence
        logger.info("Evidence snippets: %d", len(evidence))
        if not evidence:
            return []

        # Stage 2: candidate building
        candidates = build_experiment_candidates(evidence)
        logger.info("Experiment candidates: %d", len(candidates))
        if not candidates:
            return []

        # Stage 3–5: scenarios per candidate
        all_plans: list[SimulationPlan] = []

        for candidate in candidates:
            scenarios = extract_scenarios(document, candidate)
            logger.info(
                "Candidate '%s': %d raw scenarios",
                candidate.experiment_family,
                len(scenarios),
            )

            scenarios = enrich_scenarios(scenarios, document, candidate)
            scenarios = validate_scenarios(scenarios)

            accepted = [s for s in scenarios if s.status == "accepted"]
            logger.info(
                "Candidate '%s': %d accepted scenarios",
                candidate.experiment_family,
                len(accepted),
            )

            # Stage 6: build plans
            plans = build_plans_from_scenarios(scenarios)

            # Stage 7: validate each plan
            for plan in plans:
                result = validate_plan(plan)
                if not result.valid and plan.plan_status == "executable":
                    plan.plan_status = "draft"
                    plan.auto_generation_allowed = False
                    logger.info(
                        "Plan %s demoted to draft: %s",
                        plan.scenario_id,
                        result.warnings,
                    )

            all_plans.extend(plans)

        executable = [p for p in all_plans if p.plan_status == "executable"]
        logger.info(
            "Pipeline complete: %d total plans, %d executable",
            len(all_plans),
            len(executable),
        )
        return executable
