"""Path 1 pipeline: Literature Model Evaluation.

Coordinates condition extraction, mechanism conversion, model isolation
validation, simulation, and MAE computation for a literature model vs
the user's original model.
"""

import logging
from pathlib import Path

from src.agents.condition_extraction import ConditionExtractionAgent
from src.agents.conversion import ChemkinConverter
from src.agents.llm_client import LLMClient
from src.agents.validators import ModelIsolationValidator
from src.ingestion.pdf_parser import parse_pdf
from src.ingestion.pipeline.extractor import PaperExtractionPipeline
from src.schemas.experimental import (
    ExperimentalCondition,
    ExperimentalDataset,
    MAEResult,
    ObservableType,
    Path1Results,
    ReactorType,
    SimConditions,
    SimulationPlan,
)
from src.schemas.ingestion import PaperRecord
from src.simulation.core.mae import compute_mae
from src.simulation.core.runner import SimulationResult, run_simulation
from src.simulation.core.simulation_spec import SimulationSpec

logger = logging.getLogger(__name__)

# Mapping from schema enums to SimulationSpec Literal values
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
    ObservableType.K_EXT: "ignition_delay",  # best-effort fallback
}

# File extensions that indicate a Chemkin mechanism file
_CHEMKIN_EXTENSIONS = {".inp", ".dat", ".ck", ".chem"}

# Map experiment_family strings (from pipeline) to ReactorType enums
_FAMILY_TO_REACTOR: dict[str, ReactorType] = {
    "shock_tube": ReactorType.SHOCK_TUBE,
    "jsr": ReactorType.JSR,
    "flow_reactor": ReactorType.PFR,
    "flame": ReactorType.FLAME,
    "rcm": ReactorType.RCM,
}

# Map observable strings (from pipeline) to ObservableType enums
_OBS_TO_ENUM: dict[str, ObservableType] = {
    "ignition_delay": ObservableType.IDT,
    "species_profile": ObservableType.SPECIES_PROFILE,
    "flame_speed": ObservableType.FLAME_SPEED,
    "extinction_strain_rate": ObservableType.K_EXT,
}


def _plan_to_conditions(plan: SimulationPlan) -> SimConditions | None:
    """Convert a SimulationPlan to SimConditions.

    Returns None if the plan is missing required fields (T, P, composition)
    or uses an unsupported experiment family.
    """
    reactor = _FAMILY_TO_REACTOR.get(plan.experiment_family)
    if reactor is None:
        logger.debug("Unsupported family for conditions: %s", plan.experiment_family)
        return None

    if plan.temperature is None or plan.pressure is None or plan.composition is None:
        logger.debug("Plan %s missing T/P/X", plan.scenario_id)
        return None

    # Use midpoint for ranges
    T = (plan.temperature.min_value + plan.temperature.max_value) / 2.0
    P_raw = (plan.pressure.min_value + plan.pressure.max_value) / 2.0

    # Convert pressure to atm (SimConditions expects atm)
    unit = plan.pressure.unit.lower()
    if unit == "bar":
        P = P_raw / 1.01325
    elif unit in ("kpa",):
        P = P_raw / 101.325
    elif unit in ("mpa",):
        P = P_raw / 0.101325
    elif unit == "pa":
        P = P_raw / 101325.0
    else:  # atm or unknown
        P = P_raw

    X = plan.composition.species
    if not X:
        logger.debug("Plan %s has empty composition", plan.scenario_id)
        return None

    obs = ObservableType.IDT  # default
    if plan.target_observables:
        mapped = _OBS_TO_ENUM.get(plan.target_observables[0])
        if mapped:
            obs = mapped

    return SimConditions(
        reactor_type=reactor,
        T=T,
        P=P,
        X=X,
        observable_type=obs,
    )


def _deduplicate_conditions(conditions: list[SimConditions]) -> list[SimConditions]:
    """Remove duplicate conditions by (reactor_type, T, P, observable_type).

    Uses rounded T and P for comparison to avoid float noise.
    """
    seen: set[tuple] = set()
    deduped: list[SimConditions] = []
    for c in conditions:
        key = (c.reactor_type, round(c.T, 1), round(c.P, 4), c.observable_type)
        if key not in seen:
            seen.add(key)
            deduped.append(c)
    return deduped


def _find_mechanism(paper: PaperRecord) -> Path:
    """Locate a Chemkin mechanism file in the paper's SI directory."""
    if not paper.si_path:
        raise ValueError("No mechanism found in SI")

    si = Path(paper.si_path)

    # si_path could be a file directly
    if si.is_file() and si.suffix.lower() in _CHEMKIN_EXTENSIONS:
        return si

    # si_path is a directory — scan for Chemkin files
    if si.is_dir():
        for ext in _CHEMKIN_EXTENSIONS:
            matches = list(si.glob(f"*{ext}"))
            if matches:
                return matches[0]

    raise ValueError("No mechanism found in SI")


def _to_spec(cond: SimConditions, index: int, mechanism: str) -> SimulationSpec:
    """Convert a SimConditions to a SimulationSpec for the runner."""
    return SimulationSpec(
        experiment_id=f"path1_cond_{index}",
        reactor_type=_REACTOR_MAP.get(cond.reactor_type, cond.reactor_type.value),
        observable=_OBSERVABLE_MAP.get(cond.observable_type, "ignition_delay"),
        temperature=cond.T,
        pressure=cond.P * 101325.0,  # atm → Pa
        composition=cond.X,
        mechanism_file=mechanism,
    )


def _extract_observable(result: SimulationResult) -> float | None:
    """Extract a scalar observable value from raw simulation output."""
    # Flame speed stored in extra
    if "flame_speed" in result.extra:
        return float(result.extra["flame_speed"])

    # IDT: time at maximum temperature gradient (onset of ignition)
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

    # Species profile: final value of first tracked species
    for history in result.species_histories.values():
        if history:
            return history[-1]

    return None


def _match_experimental(
    cond: SimConditions,
    dataset: ExperimentalDataset,
) -> ExperimentalCondition | None:
    """Find an experimental condition matching by reactor, observable, T, P."""
    for exp in dataset.conditions:
        if exp.reactor_type != cond.reactor_type:
            continue
        if exp.observable_type != cond.observable_type:
            continue

        # Temperature match (allow list or scalar)
        exp_temps = exp.temperature_K if isinstance(exp.temperature_K, list) else [exp.temperature_K]
        if cond.T not in exp_temps:
            continue

        # Pressure match
        exp_pressures = exp.pressure_atm if isinstance(exp.pressure_atm, list) else [exp.pressure_atm]
        if cond.P not in exp_pressures:
            continue

        return exp
    return None


async def run_path1(
    paper: PaperRecord,
    original_model: Path,
    experimental_data: ExperimentalDataset,
    llm_client: LLMClient,
) -> Path1Results:
    """Execute the full Path 1 pipeline: literature model evaluation.

    Steps:
      1. Find Chemkin mechanism in paper SI
      2. Convert to Cantera YAML
      3. Validate model isolation
      4. Extract simulation conditions from paper
      5. Run simulations on both models
      6. Compute MAE against experimental data
      7. Build and return Path1Results
    """
    # ── Step 1: Find mechanism ───────────────────────────────────────────
    chemkin_path = _find_mechanism(paper)
    logger.info("Found Chemkin mechanism: %s", chemkin_path)

    # ── Step 2: Convert mechanism ────────────────────────────────────────
    converter = ChemkinConverter(llm_client)
    output_dir = chemkin_path.parent
    conversion = await converter.convert(chemkin_path, output_dir)
    if not conversion.success:
        raise ValueError(f"Mechanism conversion failed: {conversion.errors}")
    literature_model = conversion.output_path
    logger.info("Converted mechanism: %s", literature_model)

    # ── Step 2b: Extract rate parameters from raw Chemkin ──────────────
    extracted_rates = converter.extract_rates(chemkin_path)
    logger.info("Extracted %d rate entries from Chemkin SI", len(extracted_rates))

    # ── Step 3: Validate isolation ───────────────────────────────────────
    validator = ModelIsolationValidator()
    validator.validate_path1(literature_model, original_model)

    # ── Step 4: Extract conditions ───────────────────────────────────────
    document = parse_pdf(paper.pdf_path)

    # Stage 1: deterministic pipeline
    pipeline = PaperExtractionPipeline()
    plans = pipeline.extract(document)
    conditions: list[SimConditions] = []
    for plan in plans:
        cond = _plan_to_conditions(plan)
        if cond is not None:
            conditions.append(cond)
    logger.info("Deterministic pipeline: %d conditions", len(conditions))

    # Stage 2: LLM fallback (only if deterministic pipeline returns zero)
    if not conditions:
        logger.warning(
            "Deterministic pipeline returned 0 conditions, "
            "falling back to LLM extraction"
        )
        agent = ConditionExtractionAgent(llm_client)
        conditions = await agent.extract(document)

    # Deduplicate by (reactor_type, T, P, observable_type)
    conditions = _deduplicate_conditions(conditions)

    if not conditions:
        raise ValueError("No conditions extracted")
    logger.info("Total conditions after dedup: %d", len(conditions))

    # ── Step 5 & 6: Simulate and compute MAE ─────────────────────────────
    mae_results: list[MAEResult] = []

    for i, cond in enumerate(conditions):
        cond_id = f"cond_{i}"
        try:
            spec = _to_spec(cond, i, "placeholder")

            # Run both models
            lit_result = run_simulation(spec, str(literature_model))
            orig_result = run_simulation(spec, str(original_model))

            if not lit_result.success:
                logger.warning("Literature sim failed for %s: %s", cond_id, lit_result.error)
                continue
            if not orig_result.success:
                logger.warning("Original sim failed for %s: %s", cond_id, orig_result.error)
                continue

            # Extract observable values
            lit_value = _extract_observable(lit_result)
            orig_value = _extract_observable(orig_result)
            if lit_value is None or orig_value is None:
                logger.warning("Could not extract observable for %s", cond_id)
                continue

            # Match against experimental data
            exp_cond = _match_experimental(cond, experimental_data)
            if exp_cond is None:
                logger.warning("No experimental match for %s", cond_id)
                continue

            measured = (
                exp_cond.measured_value
                if isinstance(exp_cond.measured_value, list)
                else [exp_cond.measured_value]
            )

            lit_mae = compute_mae([lit_value], measured)
            orig_mae = compute_mae([orig_value], measured)

            mae_results.append(MAEResult(
                model_name="literature",
                conditions_id=cond_id,
                mae=lit_mae,
                threshold=exp_cond.error_threshold,
                passed=lit_mae <= exp_cond.error_threshold,
            ))
            mae_results.append(MAEResult(
                model_name="original",
                conditions_id=cond_id,
                mae=orig_mae,
                threshold=exp_cond.error_threshold,
                passed=orig_mae <= exp_cond.error_threshold,
            ))

        except Exception as e:
            logger.warning("Condition %s failed: %s", cond_id, e)
            continue

    # ── Step 7: Build results ────────────────────────────────────────────
    tested_ids = {r.conditions_id for r in mae_results}

    lit_better_count = 0
    for cid in tested_ids:
        lit_mae = next((r.mae for r in mae_results if r.conditions_id == cid and r.model_name == "literature"), None)
        orig_mae = next((r.mae for r in mae_results if r.conditions_id == cid and r.model_name == "original"), None)
        if lit_mae is not None and orig_mae is not None and lit_mae < orig_mae:
            lit_better_count += 1

    total = len(tested_ids)

    return Path1Results(
        literature_model_path=literature_model,
        source_doi=paper.doi,
        conditions_tested=total,
        mae_results=mae_results,
        literature_better_count=lit_better_count,
        overall_literature_better=lit_better_count > total / 2 if total > 0 else False,
        extracted_rates=extracted_rates or None,
    )
