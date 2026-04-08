"""SimulatorAgent: top-level orchestration for model evaluation.

Three entry points:
  evaluate_models        — YAML experiments + mechanism(s) → evaluation report
  extract_experiment     — PDF → scenarios + plans
  evaluate_from_literature — PDF + mechanism(s) → evaluation report + review queue

This layer is purely orchestration. All physics lives in core/,
all parsing lives in literature_support/.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from core.simulation_spec import SimulationSpec
from core.exp_loader import load_experiment_yaml
from core.model_loader import load_mechanism, MechanismInfo
from core.runner import run_simulation
from core.observable_extractor import extract_observable
from core.mae import compute_mae
from core.report_writer import (
    ModelScore,
    EvaluationReport,
    build_experiment_report,
    build_evaluation_report,
)

from orchestrator.spec_builder import spec_from_dataset, spec_from_plan
from orchestrator.review import ReviewQueue


# ── Result containers ────────────────────────────────────────────────────

@dataclass
class ExtractionResult:
    """Output of extract_experiment: scenarios and plans from a PDF."""
    pdf_path: str
    title: str | None
    scenarios: list = field(default_factory=list)
    plans: list = field(default_factory=list)
    review_queue: ReviewQueue = field(default_factory=ReviewQueue)


@dataclass
class LiteratureEvaluationResult:
    """Output of evaluate_from_literature: evaluation + extraction artifacts."""
    extraction: ExtractionResult
    report: EvaluationReport
    review_queue: ReviewQueue


# ── SimulatorAgent ───────────────────────────────────────────────────────

class SimulatorAgent:
    """Orchestrates model evaluation against experimental data.

    Usage:
        agent = SimulatorAgent()
        report = agent.evaluate_models("experiments.yaml", ["mech.yaml"])
        result = agent.extract_experiment("paper.pdf")
        lit_result = agent.evaluate_from_literature("paper.pdf", ["mech.yaml"])
    """

    # ── Primary path: YAML-driven evaluation ────────────────────────────

    def evaluate_models(
        self,
        exp_yaml: str | Path,
        model_paths: list[str | Path],
        threshold: float | None = None,
    ) -> EvaluationReport:
        """Evaluate one or more models against experimental YAML data.

        This is the primary evaluation path. The YAML contains both
        the experimental conditions and the measured reference data.
        """
        datasets = load_experiment_yaml(exp_yaml)
        mechanisms = [load_mechanism(p) for p in model_paths]

        experiment_reports = []
        for dataset in datasets:
            spec = spec_from_dataset(dataset)
            observed = [dp.value for dp in dataset.data]
            ds_threshold = threshold
            if ds_threshold is None and "threshold" in dataset.metadata:
                ds_threshold = float(dataset.metadata["threshold"])

            scores = self._score_models(spec, mechanisms, observed)

            report = build_experiment_report(
                experiment_id=spec.experiment_id,
                reactor_type=spec.reactor_type,
                observable=spec.observable,
                scores=scores,
                threshold=ds_threshold,
                observable_species=spec.observable_species,
                metadata=dataset.metadata,
            )
            experiment_reports.append(report)

        return build_evaluation_report(experiment_reports)

    # ── Secondary path: PDF extraction ──────────────────────────────────

    def extract_experiment(
        self,
        pdf_path: str | Path,
        debug: bool = False,
    ) -> ExtractionResult:
        """Extract experimental conditions from a PDF paper.

        Returns scenarios, plans, and a review queue for non-executable plans.
        """
        scenarios, plans, document = self._run_literature_pipeline(pdf_path)

        review_queue = ReviewQueue()
        for plan in plans:
            if plan.plan_status != "executable":
                review_queue.add_plan(plan)

        return ExtractionResult(
            pdf_path=str(pdf_path),
            title=document.title,
            scenarios=scenarios,
            plans=plans,
            review_queue=review_queue,
        )

    # ── Combined path: literature → evaluation ──────────────────────────

    def evaluate_from_literature(
        self,
        pdf_path: str | Path,
        model_paths: list[str | Path],
        threshold: float | None = None,
    ) -> LiteratureEvaluationResult:
        """Extract experiments from PDF, then evaluate models against them.

        Executable plans are converted to SimulationSpecs and evaluated.
        Draft plans go into the review queue.
        Blocked plans are skipped.
        """
        extraction = self.extract_experiment(pdf_path)
        mechanisms = [load_mechanism(p) for p in model_paths]

        executable = [p for p in extraction.plans if p.plan_status == "executable"]

        experiment_reports = []
        for plan in executable:
            spec = spec_from_plan(plan, mechanism_file=mechanisms[0].path)

            scores = self._score_models(spec, mechanisms, observed=None)

            report = build_experiment_report(
                experiment_id=spec.experiment_id,
                reactor_type=spec.reactor_type,
                observable=spec.observable,
                scores=scores,
                threshold=threshold,
                observable_species=spec.observable_species,
            )
            experiment_reports.append(report)

        eval_report = build_evaluation_report(experiment_reports)

        return LiteratureEvaluationResult(
            extraction=extraction,
            report=eval_report,
            review_queue=extraction.review_queue,
        )

    # ── Internal helpers ────────────────────────────────────────────────

    def _score_models(
        self,
        spec: SimulationSpec,
        mechanisms: list[MechanismInfo],
        observed: list[float] | None,
    ) -> list[ModelScore]:
        """Run simulation for each mechanism and compute MAE if observed data exists."""
        scores = []
        for mech in mechanisms:
            result = run_simulation(spec, mech.path)

            if not result.success:
                scores.append(ModelScore(
                    model_name=mech.name,
                    mechanism_file=mech.path,
                    mae=None,
                    error=result.error,
                ))
                continue

            predicted = extract_observable(spec, result)

            if observed is None:
                # Literature path: no reference data, just record that it ran
                scores.append(ModelScore(
                    model_name=mech.name,
                    mechanism_file=mech.path,
                    mae=None,
                    n_points=len(predicted),
                    error=None,
                ))
                continue

            if not predicted or not observed:
                scores.append(ModelScore(
                    model_name=mech.name,
                    mechanism_file=mech.path,
                    mae=None,
                    n_points=0,
                    error="no comparable data points",
                ))
                continue

            n = min(len(predicted), len(observed))
            mae = compute_mae(predicted[:n], observed[:n])

            scores.append(ModelScore(
                model_name=mech.name,
                mechanism_file=mech.path,
                mae=mae,
                n_points=n,
            ))

        return scores

    def _run_literature_pipeline(self, pdf_path: str | Path):
        """Run the full literature extraction pipeline. Returns (scenarios, plans, document)."""
        from literature_support.parser.pdf_parser import parse_pdf
        from literature_support.parser.section_detector import detect_sections
        from literature_support.extractor.evidence_extractor import extract_evidence
        from literature_support.builder.candidate_builder import build_experiment_candidates
        from literature_support.builder.scenario_extractor import extract_scenarios
        from literature_support.builder.scenario_enricher import enrich_scenarios
        from literature_support.builder.scenario_validator import validate_scenarios
        from literature_support.planner.simulation_plan_builder import build_plans_from_scenarios
        from literature_support.validator.plan_validator import validate_plan

        document = parse_pdf(str(pdf_path))
        sections = detect_sections(document.pages)
        evidence = extract_evidence(document, sections)
        candidates = build_experiment_candidates(evidence)

        simulatable = [c for c in candidates if c.simulatable]
        if not simulatable:
            return [], [], document

        best = simulatable[0]
        scenarios = extract_scenarios(document, best)
        scenarios = enrich_scenarios(scenarios, document, best)
        scenarios = validate_scenarios(scenarios)

        plans = build_plans_from_scenarios(scenarios)

        for plan in plans:
            vr = validate_plan(plan)
            if not vr.valid and plan.plan_status == "executable":
                plan.plan_status = "draft"
                plan.auto_generation_allowed = False
                plan.blocking_fields.extend(
                    e for e in vr.errors if e not in plan.blocking_fields
                )

        return scenarios, plans, document
