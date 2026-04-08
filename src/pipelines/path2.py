"""Path 2 pipeline: Targeted Model Improvements.

Mines candidate reactions from a paper, creates model branches with each
reaction added individually, simulates all branches, and identifies which
(if any) improve on the original model's MAE.
"""

import logging
from pathlib import Path

from src.agents.llm_client import LLMClient
from src.agents.model_branching import ModelBranchingAgent
from src.agents.reaction_mining import ReactionMiningAgent
from src.agents.validators import _normalize_equation
from src.ingestion.pdf_parser import parse_pdf
from src.schemas.experimental import (
    BranchResult,
    CandidateReaction,
    ExperimentalDataset,
    Path1Results,
    Path2Results,
    ReactorType,
    ObservableType,
)
from src.schemas.ingestion import PaperRecord
from src.simulation.core.mae import compute_mae
from src.simulation.core.runner import SimulationResult, run_simulation
from src.simulation.core.simulation_spec import SimulationSpec

logger = logging.getLogger(__name__)

# ── Type maps (shared with path1) ───────────────────────────────────────────

_REACTOR_MAP: dict[ReactorType, str] = {
    ReactorType.SHOCK_TUBE: "shock_tube",
    ReactorType.JSR: "jsr",
    ReactorType.PFR: "flow_reactor",
    ReactorType.FLAME: "flame",
    ReactorType.RCM: "rcm",
}

_OBSERVABLE_MAP: dict[ObservableType, str] = {
    ObservableType.IDT: "ignition_delay",
    ObservableType.FLAME_SPEED: "flame_speed",
    ObservableType.SPECIES_PROFILE: "species_profile",
    ObservableType.K_EXT: "ignition_delay",
}


# ── Helpers ──────────────────────────────────────────────────────────────────


def _to_spec(cond, index: int, mechanism: str, prefix: str = "path2") -> SimulationSpec:
    """Convert an ExperimentalCondition to a SimulationSpec."""
    return SimulationSpec(
        experiment_id=f"{prefix}_cond_{index}",
        reactor_type=_REACTOR_MAP.get(cond.reactor_type, cond.reactor_type.value),
        observable=_OBSERVABLE_MAP.get(cond.observable_type, "ignition_delay"),
        temperature=cond.temperature_K if isinstance(cond.temperature_K, (int, float)) else cond.temperature_K[0],
        pressure=cond.pressure_atm * 101325.0 if isinstance(cond.pressure_atm, (int, float)) else cond.pressure_atm[0] * 101325.0,
        composition=cond.mixture,
        mechanism_file=mechanism,
    )


def _extract_observable(result: SimulationResult) -> float | None:
    """Extract a scalar observable value from raw simulation output."""
    if "flame_speed" in result.extra:
        return float(result.extra["flame_speed"])

    if len(result.times) >= 2 and len(result.temperature_history) >= 2:
        max_grad = 0.0
        idt_time = result.times[-1]
        for i in range(len(result.times) - 1):
            dt = result.times[i + 1] - result.times[i]
            if dt <= 0:
                continue
            grad = (result.temperature_history[i + 1] - result.temperature_history[i]) / dt
            if grad > max_grad:
                max_grad = grad
                idt_time = result.times[i + 1]
        if max_grad > 0:
            return idt_time

    for history in result.species_histories.values():
        if history:
            return history[-1]

    return None


def _reactions_from_flux(interp) -> list[CandidateReaction]:
    """Convert flux pathways to candidate reactions."""
    candidates = []
    for pathway in interp.dominant_pathways:
        candidates.append(CandidateReaction(
            reaction_string=f"{pathway.from_species} => {pathway.to_species}",
            rate_params=None,  # flux diagrams rarely include rate parameters
            source_section="flux_diagram",
            justification=f"Dominant pathway (rank {pathway.rank}): {pathway.evidence}",
            confidence=interp.confidence,
        ))
    return candidates


def _reactions_from_sensitivity(interp) -> list[CandidateReaction]:
    """Convert sensitivity-ranked reactions to candidate reactions."""
    candidates = []
    for rxn in interp.top_reactions:
        candidates.append(CandidateReaction(
            reaction_string=rxn.reaction,
            rate_params=None,  # sensitivity figures show importance, not rates
            source_section="sensitivity_analysis",
            justification=f"Sensitivity rank {rxn.rank}, impact: {rxn.impact}",
            confidence=interp.confidence,
        ))
    return candidates


# ── Pipeline ─────────────────────────────────────────────────────────────────


async def run_path2(
    paper: PaperRecord,
    original_model: Path,
    experimental_data: ExperimentalDataset,
    llm_client: LLMClient,
    output_dir: Path,
    path1_results: Path1Results | None = None,
) -> Path2Results:
    """Execute the full Path 2 pipeline: targeted model improvements.

    Steps:
      1. Mine candidate reactions from paper figures
      2. Run original model baseline
      3. Create model branches (one per reaction)
      4. Simulate each branch
      5. Build and return Path2Results
    """
    # ── Step 1: Mine candidate reactions ─────────────────────────────────
    document = parse_pdf(paper.pdf_path)
    mining_agent = ReactionMiningAgent(llm_client)

    all_candidates: list[CandidateReaction] = []
    for caption in document.captions:
        classification = await mining_agent.classify_figure(
            image_b64=None,
            caption=caption.caption,
            figure_id=caption.figure_id,
            context_text="",
        )

        if classification.label == "flux_diagram":
            interp = await mining_agent.interpret_flux(
                image_b64=None,
                caption=caption.caption,
                figure_id=caption.figure_id,
            )
            all_candidates.extend(_reactions_from_flux(interp))

        elif classification.label == "sensitivity_analysis":
            interp = await mining_agent.interpret_sensitivity(
                image_b64=None,
                caption=caption.caption,
                figure_id=caption.figure_id,
            )
            all_candidates.extend(_reactions_from_sensitivity(interp))

    # Filter by confidence
    candidates = [c for c in all_candidates if c.confidence >= 0.5]
    if not candidates:
        raise ValueError("No high-confidence candidate reactions found")

    logger.info("Mined %d candidate reactions (%d passed confidence filter)",
                len(all_candidates), len(candidates))

    # ── Step 1b: Enrich candidates with rate params from Chemkin SI ──────
    if path1_results and path1_results.extracted_rates:
        rates_db = path1_results.extracted_rates
        enriched = 0
        for cand in candidates:
            if cand.rate_params is not None:
                continue
            key = _normalize_equation(cand.reaction_string)
            if key in rates_db:
                cand.rate_params = rates_db[key]
                enriched += 1
        logger.info("Enriched %d/%d candidates with Chemkin rate params",
                    enriched, len(candidates))

    # ── Step 2: Run original model baseline ──────────────────────────────
    conditions = experimental_data.conditions
    baseline_values: list[tuple[int, float, float]] = []  # (index, simulated, measured)

    for i, cond in enumerate(conditions):
        try:
            spec = _to_spec(cond, i, str(original_model))
            result = run_simulation(spec, str(original_model))
            if not result.success:
                logger.warning("Baseline sim failed for cond_%d: %s", i, result.error)
                continue
            value = _extract_observable(result)
            if value is None:
                logger.warning("Could not extract observable for baseline cond_%d", i)
                continue
            measured = cond.measured_value if isinstance(cond.measured_value, (int, float)) else cond.measured_value[0]
            baseline_values.append((i, value, measured))
        except Exception as e:
            logger.warning("Baseline cond_%d failed: %s", i, e)
            continue

    if not baseline_values:
        raise ValueError("All baseline simulations failed — cannot compute MAE")

    simulated = [v[1] for v in baseline_values]
    measured = [v[2] for v in baseline_values]
    baseline_mae = compute_mae(simulated, measured)
    successful_indices = {v[0] for v in baseline_values}

    logger.info("Baseline MAE: %.6f (%d conditions)", baseline_mae, len(baseline_values))

    # ── Step 3: Create model branches ────────────────────────────────────
    branching_agent = ModelBranchingAgent(original_model, output_dir)
    branches = []

    for j, rxn in enumerate(candidates):
        branch_id = f"branch_{j:03d}"
        try:
            branch = branching_agent.create_branch([rxn], branch_id)
            branches.append(branch)
        except ValueError as e:
            logger.warning("Branch %s creation failed: %s", branch_id, e)
            continue

    # ── Step 4: Simulate each branch ─────────────────────────────────────
    branch_results: list[BranchResult] = []

    for branch in branches:
        branch_sim: list[float] = []
        branch_meas: list[float] = []

        for i, cond in enumerate(conditions):
            if i not in successful_indices:
                continue  # skip conditions that failed on baseline
            try:
                spec = _to_spec(cond, i, str(branch.model_path), prefix=branch.branch_id)
                result = run_simulation(spec, str(branch.model_path))
                if not result.success:
                    logger.warning("Branch %s cond_%d failed: %s", branch.branch_id, i, result.error)
                    continue
                value = _extract_observable(result)
                if value is None:
                    continue
                measured_val = cond.measured_value if isinstance(cond.measured_value, (int, float)) else cond.measured_value[0]
                branch_sim.append(value)
                branch_meas.append(measured_val)
            except Exception as e:
                logger.warning("Branch %s cond_%d error: %s", branch.branch_id, i, e)
                continue

        if not branch_sim:
            logger.warning("Branch %s: all simulations failed", branch.branch_id)
            branch_mae = float("inf")
        else:
            branch_mae = compute_mae(branch_sim, branch_meas)

        delta = branch_mae - baseline_mae
        branch_results.append(BranchResult(
            branch_id=branch.branch_id,
            reactions_added=[r.reaction_string for r in branch.reactions_added],
            mae=branch_mae,
            delta_mae=delta,
            improved=delta < 0,
        ))

    # ── Step 5: Build Path2Results ───────────────────────────────────────
    improved = [br for br in branch_results if br.improved]
    best_id = min(improved, key=lambda b: b.mae).branch_id if improved else None

    return Path2Results(
        baseline_mae=baseline_mae,
        branches=branch_results,
        best_branch_id=best_id,
    )
