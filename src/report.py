"""Report generator: writes structured YAML reports from pipeline results."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.schemas.experimental import (
    ExperimentalDataset,
    Path1Results,
    Path2Results,
)

logger = logging.getLogger(__name__)


def _fmt_mae(value: float) -> str:
    """Format MAE float to 4 decimal places."""
    return f"{value:.4f}"


class ReportGenerator:
    """Generates a structured YAML report from pipeline results."""

    def generate(
        self,
        original_model: Path,
        experimental_data: ExperimentalDataset,
        output_dir: Path,
        path1_results: Path1Results | None = None,
        path2_results: Path2Results | None = None,
    ) -> Path:
        """Write a YAML report and return the file path.

        Output structure follows ARCHITECTURE.md section 3.7.
        Omits path1/path2 keys entirely when results are None.
        """
        now = datetime.now(tz=timezone.utc)
        run_id = f"run_{now.strftime('%Y%m%d_%H%M%S')}"

        report: dict = {
            "run_id": run_id,
            "original_model": str(original_model),
            "experimental_data": experimental_data.name,
        }

        if path1_results is not None:
            report["path1"] = self._build_path1(path1_results)

        if path2_results is not None:
            report["path2"] = self._build_path2(path2_results)

        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / f"report_{now.strftime('%Y%m%d_%H%M%S')}.yaml"
        report_path.write_text(yaml.dump(report, default_flow_style=False, sort_keys=False))
        logger.info("Report written to %s", report_path)
        return report_path

    def _build_path1(self, results: Path1Results) -> dict:
        """Build the path1 section of the report."""
        # Group MAE results by condition_id
        condition_ids = sorted({r.conditions_id for r in results.mae_results})
        result_rows = []
        for cid in condition_ids:
            lit = next(
                (r for r in results.mae_results
                 if r.conditions_id == cid and r.model_name == "literature"),
                None,
            )
            orig = next(
                (r for r in results.mae_results
                 if r.conditions_id == cid and r.model_name == "original"),
                None,
            )
            row: dict = {"condition_id": cid}
            if orig:
                row["original_mae"] = _fmt_mae(orig.mae)
                row["threshold"] = _fmt_mae(orig.threshold)
            row["literature_mae"] = _fmt_mae(lit.mae) if lit else "N/A"
            row["literature_better"] = (lit.mae < orig.mae) if (lit and orig) else False
            result_rows.append(row)

        return {
            "literature_model": str(results.literature_model_path),
            "source_doi": results.source_doi,
            "conditions_tested": results.conditions_tested,
            "results": result_rows,
            "overall_literature_better": results.overall_literature_better,
        }

    def _build_path2(self, results: Path2Results) -> dict:
        """Build the path2 section of the report."""
        branches = []
        for br in results.branches:
            branches.append({
                "branch_id": br.branch_id,
                "reactions_added": br.reactions_added,
                "mae": _fmt_mae(br.mae),
                "threshold": _fmt_mae(br.threshold),
                "passed": br.passed,
                "improvement": br.improved,
                "delta_mae": _fmt_mae(br.delta_mae),
            })

        return {
            "baseline_mae": _fmt_mae(results.baseline_mae),
            "branches": branches,
            "best_branch": results.best_branch_id,
        }
