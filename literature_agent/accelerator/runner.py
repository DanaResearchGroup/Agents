from __future__ import annotations

import argparse
import json
import sys
import os

# Force unbuffered output immediately to fix conda run/pipe buffering
sys.stdout.reconfigure(line_buffering=True)

from .step1_intake import parse_mismatch_sentence
from .step2_retrieve import retrieve_sources
from .step3_extract import extract_candidate_update


def run(sentence: str) -> dict:  
    print(f"\n" + "="*70, flush=True)
    print(" [STEP 1] AGENTIC INTAKE ".center(70, "="), flush=True)
    print("="*70, flush=True)
    print(f"Input: '{sentence}'", flush=True)
    
    brief = parse_mismatch_sentence(sentence)
    print("\nEXTRACTED FAILURE BRIEF:", flush=True)
    print(brief.model_dump_json(indent=2), flush=True)
    
    print(f"\n" + "="*70, flush=True)
    print(" [STEP 2] TARGETED RETRIEVAL & AI FILTERING ".center(70, "="), flush=True)
    print("="*70, flush=True)
    print(f"Searching Semantic Scholar and filtering for kinetics relevance...", flush=True)
    
    hits = retrieve_sources(brief)
    
    print(f"\nFOUND {len(hits)} RELEVANT SOURCES:", flush=True)

    with open("curated_search_results.txt", "w") as f_curated:
        f_curated.write("=== CURATED RETRIEVAL & AI FILTERED RESULTS ===\n")
        f_curated.write(f"Total relevant sources found: {len(hits)}\n\n")

        def _make_clickable(url: str, text: str | None = None) -> str:
            """OSC 8 escape sequence for clickable terminal links."""
            t = text or url
            return f"\033]8;;{url}\033\\{t}\033]8;;\033\\"

        def _print_hit(index: int, hit: Any, file_out: Any) -> None:
            if hit.authors:
                author_str = hit.authors[0].split()[-1] + " et al." if len(hit.authors) > 1 else hit.authors[0]
            else:
                author_str = "Unknown Authors"
            
            year_str = str(hit.year) if hit.year else "n.d."
            venue_str = f" ({hit.venue})" if hit.venue else ""
            
            header = f"  {index}   {author_str}, {year_str} — “{hit.title}”{venue_str}"
            reason = f"     • Why it matched: {hit.match_reason or 'Direct relevance to kinetics discrepancy.'}"
            contains = f"     • Likely contains: {hit.likely_contains or 'Structured kinetic parameters and model evaluation.'}"
            use = f"     • Best use: {hit.best_use or 'Primary source for model refinement.'}"
            link = f"     • Link: {hit.url}" if hit.url else ""
            doi = f"     (DOI: {hit.doi})" if hit.doi else "     (DOI: None)"

            print(header, flush=True)
            print(reason, flush=True)
            print(contains, flush=True)
            print(use, flush=True)
            if hit.url:
                print(f"     • Link: {_make_clickable(hit.url)}", flush=True)
            print(f"{doi}\n", flush=True)

            file_out.write(header + "\n")
            file_out.write(reason + "\n")
            file_out.write(contains + "\n")
            file_out.write(use + "\n")
            if link:
                file_out.write(link + "\n")
            file_out.write(doi + "\n\n")

        for i, hit in enumerate(hits, 1):
            _print_hit(i, hit, f_curated)
    
    print(f"\n" + "="*70, flush=True)
    print(" [STEP 3] EVIDENCE EXTRACTION & SYNTHESIS ".center(70, "="), flush=True)
    print("="*70, flush=True)
    
    candidate = extract_candidate_update(brief, hits)
    print("\nPROPOSED CANDIDATE UPDATE:", flush=True)
    print(candidate.model_dump_json(indent=2), flush=True)
    
    print("\n" + "="*70, flush=True)
    print(" WORKFLOW COMPLETE ".center(70, "="), flush=True)
    print("="*70, flush=True)
    
    return {
        "failure_brief": brief.model_dump(),
        "candidate_update": candidate.model_dump(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the 3-step accelerator workflow on a mismatch sentence."
    )
    parser.add_argument(
        "sentence",
        nargs="?",
        default="At 900 K and 1 bar JSR, the model overpredicts CO and slightly underpredicts CO2 for HOCHO oxidation.",
        help="Mismatch sentence to analyze (e.g., 'At 900 K, 1 bar JSR...').",
    )
    args = parser.parse_args()
    run(args.sentence)


if __name__ == "__main__":
    main()
