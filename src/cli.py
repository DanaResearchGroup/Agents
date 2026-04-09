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

    return parser


def _resolve_paths(args: argparse.Namespace) -> None:
    """Resolve paths on the namespace so they are absolute."""
    for attr in ("model", "experiment", "output", "llm_config", "input",
                 "literature_model", "species_aliases"):
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
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
