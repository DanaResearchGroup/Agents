"""Generate standalone Cantera simulation scripts from validated plans.

This is an export layer — the real simulation engine is core/runner.py.
Generated scripts mirror the runner's reactor setups for reproducibility
and standalone execution.

Usage:
    from generator.cantera_generator import generate_cantera_script
    script = generate_cantera_script(plan)
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from models import SimulationPlan
from .plan_exporter import export_plan_json


# ── Template registry ────────────────────────────────────────────────────
# Maps template_family (from SimulationPlan) to template filename.

_TEMPLATE_REGISTRY: dict[str, str] = {
    "idt_const_uv":  "shock_tube_idt_const_uv.py.j2",
    "idt_const_hp":  "shock_tube_idt_const_uv.py.j2",  # same template, const-UV
    "jsr_const_tp":  "jsr_const_tp.py.j2",
    "pfr_const_tp":  "flow_reactor_const_tp.py.j2",
    "flame_speed":   "flame_speed.py.j2",
    "batch_const_tp": "flow_reactor_const_tp.py.j2",  # same reactor approach
}

_TEMPLATES_DIR = Path(__file__).parent.parent / "templates"


def _get_env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        keep_trailing_newline=True,
    )


def generate_cantera_script(plan: SimulationPlan) -> str:
    """Generate a standalone Cantera Python script from an executable plan.

    Raises ValueError for non-executable or unsupported plans.
    """
    if plan.plan_status != "executable":
        raise ValueError(
            f"Cannot generate script for non-executable plan: {plan.scenario_id}"
        )

    template_file = _TEMPLATE_REGISTRY.get(plan.template_family)
    if template_file is None:
        raise ValueError(
            f"Template '{plan.template_family}' not yet supported for code generation. "
            f"Supported: {sorted(_TEMPLATE_REGISTRY.keys())}"
        )

    data = export_plan_json(plan)
    context = _prepare_context(data, plan)

    env = _get_env()
    template = env.get_template(template_file)
    return template.render(context)


def _prepare_context(data: dict, plan: SimulationPlan) -> dict:
    """Prepare template rendering context from plan export data."""
    # Composition string for Cantera
    comp = data.get("composition", {})
    comp_str = ", ".join(f"{sp}: {frac}" for sp, frac in comp.items())

    # Temperature
    t = data.get("temperature", {})
    t_min = t.get("min_K", 1000)
    t_max = t.get("max_K", t_min)
    t_mode = t.get("mode", "setpoint")

    # Pressure
    p = data.get("pressure", {})
    p_min = p.get("min_Pa", 101325.0)

    return {
        "scenario_id": plan.scenario_id,
        "experiment_family": plan.experiment_family,
        "template_family": plan.template_family,
        "composition_str": comp_str,
        "composition_species": list(comp.keys()),
        "temperature_min": t_min,
        "temperature_max": t_max,
        "temperature_mode": t_mode,
        "n_temperatures": 10,
        "pressure_min": p_min,
        "mechanism": data.get("mechanism", "mechanism.yaml"),
        "target_observables": data.get("target_observables", []),
        "residence_time": (plan.residence_time.min_value
                          if plan.residence_time else None),
        "end_time": 1.0,
    }
