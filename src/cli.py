"""CLI entry point for the chem-agent command."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from src.schemas.experimental import PaperSource, RunConfig


def parse_paper_source(raw: str) -> PaperSource:
    """Parse a --paper argument into a PaperSource.

    Accepted formats:
        doi:10.1016/xxx        -> PaperSource(mode="doi", value="10.1016/xxx")
        file:path/to.pdf       -> PaperSource(mode="upload", value="path/to.pdf")
        search:query string    -> PaperSource(mode="search", value="query string")
    """
    prefixes = {"doi:": "doi", "file:": "upload", "search:": "search"}
    for prefix, mode in prefixes.items():
        if raw.startswith(prefix):
            return PaperSource(mode=mode, value=raw[len(prefix):])
    raise argparse.ArgumentTypeError(
        f"Invalid paper source: {raw!r}. "
        "Use doi:<DOI>, file:<path>, or search:<query>."
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="chem-agent",
        description="Multi-agent framework for evaluating and improving chemical kinetic models.",
    )
    sub = parser.add_subparsers(dest="command")

    # ── run ──────────────────────────────────────────
    run_p = sub.add_parser("run", help="Run the full evaluation pipeline")
    run_p.add_argument(
        "--model", required=True, type=Path,
        help="Path to Cantera/Chemkin mechanism file",
    )
    run_p.add_argument(
        "--experiment", required=True, type=Path,
        help="Path to experimental data YAML file",
    )
    run_p.add_argument(
        "--paper", required=True, type=str,
        help="Paper source: doi:<DOI>, file:<path>, or search:<query>",
    )
    run_p.add_argument("--path1", action="store_true", help="Run Path 1 only")
    run_p.add_argument("--path2", action="store_true", help="Run Path 2 only")
    run_p.add_argument(
        "--output", type=Path, default=Path("data/reports"),
        help="Output directory (default: data/reports)",
    )
    run_p.add_argument(
        "--llm-config", type=Path, default=Path("config/llm_config.yaml"),
        help="LLM configuration file (default: config/llm_config.yaml)",
    )
    run_p.add_argument(
        "--literature-model", type=Path, default=None,
        help="Path to literature mechanism YAML/Chemkin. "
             "Bypasses SI download. Required for --path1 with file: ingestion.",
    )
    run_p.add_argument(
        "--species-aliases", type=Path, default=None,
        help="Path to YAML file mapping plain species names to model names "
             "(e.g. {NH3: 'NH3(1)', O2: 'O2(6)'}).",
    )
    run_p.add_argument(
        "--literature-aliases", type=Path, default=None,
        help="Path to YAML file mapping original model species to literature "
             "model names (e.g. {'NH3(1)': 'NH3', 'O2(6)': 'O2'}).",
    )
    run_p.add_argument(
        "--browser-auth", action="store_true", default=False,
        help="Use browser cookies for authenticated PDF downloads (institutional access).",
    )

    # ── validate-model ──────────────────────────────
    val_p = sub.add_parser("validate-model", help="Quick model sanity check")
    val_p.add_argument(
        "--model", required=True, type=Path,
        help="Path to Cantera/Chemkin mechanism file",
    )

    # ── convert ─────────────────────────────────────
    conv_p = sub.add_parser("convert", help="Convert Chemkin mechanism to Cantera YAML")
    conv_p.add_argument(
        "--input", required=True, type=Path,
        help="Path to Chemkin mechanism file (.inp)",
    )
    conv_p.add_argument(
        "--output", required=True, type=Path,
        help="Output directory for converted YAML",
    )
    conv_p.add_argument(
        "--llm-config", type=Path, default=Path("config/llm_config.yaml"),
        help="LLM configuration file (default: config/llm_config.yaml)",
    )

    # ── search ───────────────────────────────────────
    search_p = sub.add_parser("search", help="Search for kinetics papers")
    search_p.add_argument(
        "query", type=str,
        help="Search query (e.g. 'HOCHO pyrolysis')",
    )
    search_p.add_argument(
        "--max-results", type=int, default=10,
        help="Max results per search strategy (default: 10)",
    )
    search_p.add_argument(
        "--snowball", action="store_true",
        help="Enable citation snowballing on high-priority results",
    )
    search_p.add_argument(
        "--download-si", action="store_true",
        help="Download supplementary information files",
    )
    search_p.add_argument(
        "--output", type=Path, default=Path("data/papers"),
        help="Output directory (default: data/papers)",
    )

    # ── mae ──────────────────────────────────────────
    mae_p = sub.add_parser("mae", help="Run simulation vs experimental MAE evaluation")
    mae_p.add_argument(
        "--model", required=True, type=Path,
        help="Path to Cantera mechanism YAML file",
    )
    mae_p.add_argument(
        "--experiment", required=True, type=Path,
        help="Path to experimental data YAML file",
    )
    mae_p.add_argument(
        "--end-time", type=float, default=1.0,
        help="Simulation end time in seconds (default: 1.0)",
    )

    # ── download ─────────────────────────────────────
    dl_p = sub.add_parser("download", help="Download a paper PDF by DOI")
    dl_p.add_argument(
        "--doi", required=True, type=str,
        help="DOI of the paper to download (e.g. 10.1021/acsomega.5c11182)",
    )
    dl_p.add_argument(
        "--output", type=Path, default=Path("data/papers"),
        help="Output directory (default: data/papers)",
    )
    dl_p.add_argument(
        "--browser-auth", action="store_true", default=False,
        help="Use browser cookies for authenticated PDF downloads.",
    )

    return parser


def _resolve_paths(args: argparse.Namespace) -> None:
    """Resolve paths on the namespace so they are absolute."""
    for attr in ("model", "experiment", "output", "llm_config", "input",
                 "literature_model", "species_aliases", "literature_aliases"):
        val = getattr(args, attr, None)
        if isinstance(val, Path):
            setattr(args, attr, val.resolve())


def cmd_run(args: argparse.Namespace) -> None:
    """Execute the full orchestrator pipeline."""
    import logging
    import yaml as _yaml
    from src.orchestrator import Orchestrator

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
    )

    paper = parse_paper_source(args.paper)

    # If neither --path1 nor --path2 is given, run both
    run_path1 = True
    run_path2 = True
    if args.path1 or args.path2:
        run_path1 = args.path1
        run_path2 = args.path2

    # Load species aliases from YAML if provided
    # BaseLoader keeps all keys as strings (safe_load would parse NO/YES as booleans)
    aliases: dict[str, str] = {}
    if args.species_aliases is not None:
        aliases = _yaml.load(args.species_aliases.read_text(), Loader=_yaml.BaseLoader) or {}

    lit_aliases: dict[str, str] = {}
    if getattr(args, "literature_aliases", None) is not None:
        lit_aliases = _yaml.load(args.literature_aliases.read_text(), Loader=_yaml.BaseLoader) or {}

    config = RunConfig(
        original_model=args.model,
        experimental_data=args.experiment,
        paper_source=paper,
        run_path1=run_path1,
        run_path2=run_path2,
        output_dir=args.output,
        llm_config=args.llm_config,
        literature_model=args.literature_model,
        species_aliases=aliases,
        literature_aliases=lit_aliases,
        browser_auth=args.browser_auth,
    )
    orchestrator = Orchestrator(config)
    report_path = asyncio.run(orchestrator.run())
    print(f"Report written to {report_path}")


def cmd_validate_model(args: argparse.Namespace) -> None:
    """Run ModelIsolationValidator.reaction_ids() and print the count."""
    from src.agents.validators import ModelIsolationValidator

    validator = ModelIsolationValidator()
    ids = validator.reaction_ids(args.model)
    print(f"Model: {args.model}")
    print(f"Reactions: {len(ids)}")


def cmd_convert(args: argparse.Namespace) -> None:
    """Run ChemkinConverter standalone."""
    from src.agents.llm_client import LLMClient
    from src.agents.conversion import ChemkinConverter

    client = LLMClient(config_path=args.llm_config)
    converter = ChemkinConverter(llm_client=client)
    result = asyncio.run(converter.convert(args.input, args.output))
    if result.success:
        print(f"Converted: {result.output_path}")
    else:
        print(f"Conversion failed: {'; '.join(result.errors)}", file=sys.stderr)
        sys.exit(1)


def cmd_search(args: argparse.Namespace) -> None:
    """Search for kinetics papers and print results."""
    import logging
    from src.ingestion.registry_builder import RegistryBuilder
    from src.ingestion.utils import save_registry

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
    )

    builder = RegistryBuilder()
    result = builder.build_registry(
        query=args.query,
        download_si=args.download_si,
        outdir=str(args.output),
        per_strategy_limit=args.max_results,
        snowball=args.snowball,
    )

    save_registry(result, args.output)
    print(f"\nFound {len(result.papers)} papers. Registry saved to {args.output}/")

    for paper in result.papers:
        doi_str = f"doi:{paper.doi}" if paper.doi else "(no DOI)"
        print(f"  {paper.title[:70]}  [{doi_str}]")

    if result.papers:
        print("\nTo run evaluation on a paper:")
        for paper in result.papers:
            if paper.doi:
                print(f"  chem-agent run --paper doi:{paper.doi} ...")
                break


_OBS_MAP: dict[str, str] = {
    "idt": "ignition_delay",
    "flame_speed": "flame_speed",
    "species_profile": "species_profile",
    "k_ext": "conversion",
}


def _extract_value(sim_result, cond) -> float | None:
    """Extract a scalar observable value from a SimulationResult."""
    obs = cond.observable_type.value
    if obs == "species_profile":
        label = cond.observable_label
        if label and label in sim_result.species_histories:
            history = sim_result.species_histories[label]
            return history[-1] if history else None
    elif obs == "idt":
        # Ignition delay: time of max dT/dt
        temps = sim_result.temperature_history
        times = sim_result.times
        if len(temps) >= 2:
            max_dt, idt = 0.0, times[-1]
            for i in range(1, len(temps)):
                dt_val = (temps[i] - temps[i - 1]) / (times[i] - times[i - 1])
                if dt_val > max_dt:
                    max_dt = dt_val
                    idt = times[i]
            return idt
    elif obs == "flame_speed":
        return sim_result.extra.get("flame_speed")
    return None


def cmd_mae(args: argparse.Namespace) -> None:
    """Run simulation vs experimental MAE evaluation."""
    import yaml as _yaml
    from src.schemas.experimental import ExperimentalDataset
    from src.simulation.core.simulation_spec import SimulationSpec
    from src.simulation.core.runner import run_simulation

    raw = _yaml.safe_load(args.experiment.read_text())
    dataset = ExperimentalDataset(**raw)
    mechanism = str(args.model)

    pass_count = 0
    total = len(dataset.conditions)

    for i, cond in enumerate(dataset.conditions):
        temp = cond.temperature_K if isinstance(cond.temperature_K, (int, float)) else cond.temperature_K[0]
        pres = cond.pressure_atm if isinstance(cond.pressure_atm, (int, float)) else cond.pressure_atm[0]
        measured = cond.measured_value if isinstance(cond.measured_value, (int, float)) else cond.measured_value[0]

        spec = SimulationSpec(
            experiment_id=f"{dataset.name}_{i}",
            reactor_type=cond.reactor_type.value,
            observable=_OBS_MAP.get(cond.observable_type.value, cond.observable_type.value),
            observable_species=cond.observable_label,
            temperature=temp,
            pressure=pres * 101325.0,
            composition=cond.mixture,
            end_time=args.end_time,
        )

        sim_result = run_simulation(spec, mechanism)

        if not sim_result.success:
            print(f"T={temp}K P={pres}atm | ERROR: {sim_result.error}")
            continue

        simulated = _extract_value(sim_result, cond)
        if simulated is None:
            print(f"T={temp}K P={pres}atm | ERROR: could not extract observable")
            continue

        mae = abs(simulated - measured)
        passed = mae <= cond.error_threshold
        if passed:
            pass_count += 1
        status = "PASS" if passed else "FAIL"
        print(f"T={temp}K P={pres}atm | simulated={simulated:.6f} | measured={measured:.6f} | MAE={mae:.6f} | threshold={cond.error_threshold} | {status}")

    print(f"\n{pass_count}/{total} conditions pass threshold")


def cmd_download(args: argparse.Namespace) -> None:
    """Download a paper PDF by DOI, optionally using browser cookies."""
    from src.ingestion.retrieval import CrossrefClient, OpenAlexClient
    from src.ingestion.utils import download_file

    doi = args.doi
    client = OpenAlexClient()
    paper = client.get_by_doi(doi)

    if paper is None:
        crossref = CrossrefClient()
        paper = crossref.search_by_doi(doi)

    if paper is None:
        print(f"No paper found for DOI: {doi}", file=sys.stderr)
        sys.exit(1)

    url = paper.oa_url or paper.source_url
    if not url:
        print(f"No download URL available for DOI: {doi}", file=sys.stderr)
        sys.exit(1)

    safe_doi = doi.replace("/", "_").replace(":", "_")
    result = download_file(
        url, args.output, f"{safe_doi}.pdf",
        use_browser_auth=args.browser_auth,
    )
    if result:
        print(f"Downloaded: {result}")
    else:
        print(f"Download failed for {url}", file=sys.stderr)
        sys.exit(1)


def main(argv: list[str] | None = None) -> None:
    """Entry point for chem-agent command."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    _resolve_paths(args)

    commands = {
        "run": cmd_run,
        "validate-model": cmd_validate_model,
        "convert": cmd_convert,
        "search": cmd_search,
        "mae": cmd_mae,
        "download": cmd_download,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
