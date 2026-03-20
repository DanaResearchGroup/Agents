from __future__ import annotations

import argparse
import json
import sys
import os
from datetime import datetime
from pathlib import Path

# Force unbuffered output immediately to fix conda run/pipe buffering
sys.stdout.reconfigure(line_buffering=True)

from .step1_intake import parse_mismatch_sentence
from .step2_retrieve import retrieve_sources


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
    


    # [STEP 4] SUPPLEMENTARY INFORMATION (SI) DOWNLOAD
    print(f"\n" + "="*70, flush=True)
    print(" [STEP 4] SUPPLEMENTARY INFORMATION DOWNLOAD ".center(70, "="), flush=True)
    print("="*70, flush=True)
    
    from .io_utils import download_file
    from pathlib import Path
    
    # Pre-emptively create the results subfolder to store everything together
    safe_sentence = "".join([c if c.isalnum() else "_" for c in sentence[:30]]).strip("_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    res_dir = Path("outputs") / f"run_{safe_sentence}_{timestamp}"
    res_dir.mkdir(parents=True, exist_ok=True)
    
    si_dir = res_dir / "si_files"
    download_count = 0
    
    for hit in hits:
        if hit.si_link_found and hit.si_links:
            print(f"Discovered SI for: {hit.title[:50]}...", flush=True)
            for i, url in enumerate(hit.si_links):
                print(f"  -> Attempting download: {url}", flush=True)
                safe_title = "".join([c if c.isalnum() else "_" for c in hit.title[:30]])
                filename = f"{safe_title}_si_{i}.file"
                filepath = download_file(url, si_dir, filename)
                if filepath:
                    print(f"  [✓] Saved to: {filepath}", flush=True)
                    download_count += 1
                else:
                    print(f"  [✗] Failed to download from: {url}", flush=True)

    if download_count == 0:
        print("No SI files found or downloaded.", flush=True)
    else:
        print(f"Successfully downloaded {download_count} SI file(s).", flush=True)
    
    # [STEP 5] SAVE STRUCTURED RESULTS
    with open(res_dir / "failure_brief.json", "w") as f:
        f.write(brief.model_dump_json(indent=2))
    
    with open(res_dir / "curated_search_results.json", "w") as f:
        import json
        json.dump([h.model_dump() for h in hits], f, indent=2)
        

        
    print(f"\n[✓] All results saved to: {res_dir}", flush=True)

    print("\n" + "="*70, flush=True)
    print(" WORKFLOW COMPLETE ".center(70, "="), flush=True)
    print("="*70, flush=True)
    
    return {
        "failure_brief": brief.model_dump(),
        "hits": [h.model_dump() for h in hits],
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
