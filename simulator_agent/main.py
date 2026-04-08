"""CLI entry point for the simulator agent.

Primary path (model evaluation):
    python main.py evaluate --exp-yaml experiments.yaml --model mechanism.yaml
    python main.py --exp-yaml experiments.yaml --model original.yaml --compare-model literature.yaml

Secondary path (literature extraction):
    python main.py extract --pdf paper.pdf [--outdir outputs/] [--debug]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════
# CLI commands — thin wrappers around SimulatorAgent
# ═══════════════════════════════════════════════════════════════════════════

def cmd_evaluate(args: argparse.Namespace) -> int:
    """Evaluate model(s) against experimental YAML data."""
    from orchestrator import SimulatorAgent
    from core.report_writer import write_report_json
    from core.model_loader import load_mechanism

    exp_path = Path(args.exp_yaml)
    if not exp_path.exists():
        print(f"Error: experiment YAML not found: {exp_path}", file=sys.stderr)
        return 1

    outdir = Path(args.outdir) if args.outdir else exp_path.parent
    outdir.mkdir(parents=True, exist_ok=True)

    model_paths = [args.model]
    if args.compare_model:
        model_paths.extend(args.compare_model)

    # Print loading info
    print(f"Loading experiments: {exp_path}")
    print(f"Loading {len(model_paths)} mechanism(s)")
    for mp in model_paths:
        info = load_mechanism(mp)
        print(f"  {info.name}: {info.n_species} species, {info.n_reactions} reactions")

    agent = SimulatorAgent()
    report = agent.evaluate_models(exp_path, model_paths, threshold=args.threshold)

    # Write report
    report_path = outdir / "evaluation_report.json"
    write_report_json(report, report_path)

    # Print summary
    _print_evaluation_summary(report)
    print(f"\nReport written to {report_path}")
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    """Extract experimental conditions from a PDF paper."""
    from orchestrator import SimulatorAgent
    from literature_support.report import (
        write_scenarios_json, write_plans_json, write_debug_json, write_summary,
    )
    from literature_support.parser.pdf_parser import parse_pdf

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"Error: PDF not found: {pdf_path}", file=sys.stderr)
        return 1

    outdir = Path(args.outdir) if args.outdir else pdf_path.parent
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Parsing: {pdf_path}")
    agent = SimulatorAgent()
    result = agent.extract_experiment(pdf_path)

    # Print extraction summary
    print(f"  Title: {result.title or '(not detected)'}")
    scenarios = result.scenarios
    plans = result.plans

    accepted = [s for s in scenarios if s.status == "accepted"]
    review = [s for s in scenarios if s.status == "needs_review"]
    rejected = [s for s in scenarios if s.status == "rejected"]
    print(f"  Scenarios: {len(scenarios)} ({len(accepted)} accepted, "
          f"{len(review)} needs review, {len(rejected)} rejected)")

    for s in scenarios:
        tag = " OK" if s.status == "accepted" else ("REV" if s.status == "needs_review" else "REJ")
        conf = f"{s.confidence.overall:.0%}"
        print(f"    [{tag} {conf:>4s}] {s.source_label}: "
              f"{s.composition_text or '?'} @ {s.temperature_text or '?'}")

    executable = [p for p in plans if p.plan_status == "executable"]
    drafts = [p for p in plans if p.plan_status == "draft"]
    print(f"  Plans: {len(plans)} ({len(executable)} executable, {len(drafts)} draft)")

    for p in plans:
        if p.plan_status == "executable":
            print(f"    [EXEC] {p.scenario_id}: {p.template_family}")
        elif p.plan_status == "draft":
            blockers = ", ".join(p.blocking_fields[:2])
            print(f"    [DRFT] {p.scenario_id}: {p.template_family} — blocked: {blockers}")

    if not result.review_queue.is_empty:
        rq = result.review_queue
        print(f"  Review queue: {rq.total} ({len(rq.drafts)} draft, {len(rq.blocked)} blocked)")

    # Write outputs
    document = parse_pdf(str(pdf_path))  # re-parse for report writing (lightweight)
    from literature_support.extractor.evidence_extractor import extract_evidence
    from literature_support.parser.section_detector import detect_sections
    from literature_support.builder.candidate_builder import build_experiment_candidates

    sections = detect_sections(document.pages)
    evidence = extract_evidence(document, sections)
    candidates = build_experiment_candidates(evidence)

    primary_family = None
    simulatable = [c for c in candidates if c.simulatable]
    if simulatable:
        primary_family = simulatable[0].experiment_family

    scen_path = outdir / "scenarios.json"
    plan_path = outdir / "simulation_plans.json"
    summary_path = outdir / "run_summary.md"

    write_scenarios_json(scen_path, document.pdf_path, document.title, primary_family, scenarios)
    write_plans_json(plan_path, document.pdf_path, document.title, plans)
    write_summary(summary_path, document, candidates, scenarios, plans)

    written = [scen_path.name, plan_path.name, summary_path.name]

    if args.debug:
        write_debug_json(outdir / "parsed_document.json", document.to_dict())
        write_debug_json(outdir / "evidence.json", {
            "pdf_path": document.pdf_path,
            "title": document.title,
            "evidence_count": len(evidence),
            "evidence": [e.to_dict() for e in evidence],
        })
        write_debug_json(outdir / "experiment_candidates.json", {
            "pdf_path": document.pdf_path,
            "title": document.title,
            "candidate_count": len(candidates),
            "candidates": [c.to_dict() for c in candidates],
        })
        written.extend(["parsed_document.json", "evidence.json", "experiment_candidates.json"])

    print(f"\nOutputs written to {outdir}/")
    for name in written:
        print(f"  {name}")
    return 0


# ── Output formatting ───────────────────────────────────────────────────

def _print_evaluation_summary(report) -> None:
    print(f"\n{'═' * 60}")
    print("Evaluation Summary")
    print(f"{'═' * 60}")
    for er in report.experiments:
        print(f"\n  {er.experiment_id} ({er.reactor_type} / {er.observable}):")
        for ms in er.models:
            if ms.mae is not None:
                threshold_tag = ""
                if ms.passed_threshold is not None:
                    threshold_tag = " PASS" if ms.passed_threshold else " FAIL"
                print(f"    {ms.model_name}: MAE={ms.mae:.6g}{threshold_tag}")
            else:
                print(f"    {ms.model_name}: ERROR — {ms.error}")
        if er.winner:
            print(f"    → winner: {er.winner}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI argument parsing
# ═══════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Simulator agent: evaluate kinetic models against experimental data",
    )
    sub = p.add_subparsers(dest="command")

    # ── Primary: evaluate ────────────────────────────────────────────────
    eval_p = sub.add_parser("evaluate", help="Evaluate model(s) against experimental YAML")
    eval_p.add_argument("--exp-yaml", required=True, help="Path to experimental YAML")
    eval_p.add_argument("--model", required=True, help="Path to primary mechanism YAML")
    eval_p.add_argument(
        "--compare-model", nargs="+", default=None,
        help="Additional mechanism(s) to compare",
    )
    eval_p.add_argument("--outdir", default=None, help="Output directory")
    eval_p.add_argument("--threshold", type=float, default=None, help="MAE threshold for pass/fail")

    # ── Secondary: extract ───────────────────────────────────────────────
    ext_p = sub.add_parser("extract", help="Extract experimental conditions from PDF")
    ext_p.add_argument("--pdf", required=True, help="Path to the PDF file")
    ext_p.add_argument("--outdir", default=None, help="Output directory")
    ext_p.add_argument("--debug", action="store_true", help="Write intermediate debug files")

    # ── Convenience: direct flags for backward compat ────────────────────
    p.add_argument("--exp-yaml", default=None, help=argparse.SUPPRESS)
    p.add_argument("--model", default=None, help=argparse.SUPPRESS)
    p.add_argument("--compare-model", nargs="+", default=None, help=argparse.SUPPRESS)
    p.add_argument("--pdf", default=None, help=argparse.SUPPRESS)
    p.add_argument("--outdir", default=None, help=argparse.SUPPRESS)
    p.add_argument("--debug", action="store_true", default=False, help=argparse.SUPPRESS)
    p.add_argument("--threshold", type=float, default=None, help=argparse.SUPPRESS)

    args = p.parse_args()

    if args.command is None:
        if args.exp_yaml and args.model:
            args.command = "evaluate"
        elif args.pdf:
            args.command = "extract"
        else:
            p.print_help()
            sys.exit(1)

    return args


def main() -> int:
    args = _parse_args()

    if args.command == "evaluate":
        return cmd_evaluate(args)
    elif args.command == "extract":
        return cmd_extract(args)
    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
