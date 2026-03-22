from __future__ import annotations
import argparse
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any
import os

# Bypass local proxy misconfigurations that are causing connection failures
for key in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"]:
    os.environ.pop(key, None)

from .registry_builder import RegistryBuilder, PaperRecord, SearchQuery
from .utils import save_registry

# Force unbuffered output for terminal display
sys.stdout.reconfigure(line_buffering=True)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

def run(query_input: str or SearchQuery | List[str], 
        max_results: int = 30, 
        download_si: bool = False, 
        download_papers: bool = False,
        use_llm: bool = False,
        outdir: str = "outputs",
        snowball: bool = False) -> dict:
    
    print(f"\n" + "="*70, flush=True)
    print(" LITERATURE DISCOVERY WORKFLOW ".center(70, "="), flush=True)
    print("="*70, flush=True)
    
    # Explicit Mode Selection
    llm_enabled = (use_llm == True) 
    
    print(f"Input: '{query_input if isinstance(query_input, str) else 'JSON Object'}'", flush=True)
    print(f"Mode: {'Agentic Discovery' if llm_enabled else 'Manual Keyword Search'}", flush=True)

    builder = RegistryBuilder(use_llm=llm_enabled)
    
    # Ensure snowball is set if we wrap query_input in SearchQuery later or pass it directly
    if isinstance(query_input, SearchQuery):
        query_input.snowball = snowball or query_input.snowball
    elif snowball and not isinstance(query_input, (str, list, SearchQuery)):
        # Fallback
        pass

    # 2. Setup Output Directory
    safe_name = "".join([c if c.isalnum() else "_" for c in str(query_input)[:30]]).strip("_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    res_dir = Path(outdir) / f"run_{safe_name}_{timestamp}"
    res_dir.mkdir(parents=True, exist_ok=True)

    # 3. Run Pipeline
    print(f"\nAnalyzing input and orchestrating discovery...", flush=True)
    per_strat = max(5, max_results // 2) if llm_enabled else max_results
    
    effective_query = query_input
    if snowball and not isinstance(query_input, SearchQuery):
        if isinstance(query_input, str):
            effective_query = SearchQuery(topic=query_input, snowball=True)

    # Pass the res_dir to build_registry so si_files are nested inside
    result = builder.build_registry(effective_query, download_si=download_si, download_papers=download_papers, outdir=str(res_dir), per_strategy_limit=per_strat, snowball=snowball)
    hits = result.papers
    
    print(f"\nFOUND {len(hits)} RELEVANT SOURCES:", flush=True)

    # 4. Handle Outputs
    with open(res_dir / "curated_search_results.txt", "w") as f_curated:
        f_curated.write("=== CURATED RETRIEVAL & AI FILTERED RESULTS ===\n")
        f_curated.write(f"Total relevant sources found: {len(hits)}\n\n")

        def _make_clickable(url: str, text: str | None = None) -> str:
            t = text or url
            return f"\033]8;;{url}\033\\{t}\033]8;;\033\\"

        def _print_hit(index: int, hit: PaperRecord, file_out: Any) -> None:
            author_str = (hit.authors[0].split()[-1] + " et al.") if len(hit.authors) > 1 else (hit.authors[0] if hit.authors else "Unknown Authors")
            year_str = str(hit.year) if hit.year else "n.d."
            venue_str = f" ({hit.venue})" if hit.venue else ""
            
            header = f"  {index}   {author_str}, {year_str} — “{hit.title}”{venue_str}"
            reason = f"     • Why it matched: {hit.match_reason or 'Keyword relevance.'}"
            contains = f"     • Likely contains: {hit.likely_contains or 'Kinetic data.'}"
            doi = f"     (DOI: {hit.doi})" if hit.doi else "     (DOI: None)"

            print(header, flush=True)
            print(reason, flush=True)
            print(contains, flush=True)
            if hit.source_url:
                print(f"     • Link: {_make_clickable(hit.source_url)}", flush=True)
            print(f"{doi}\n", flush=True)

            file_out.write(f"{header}\n{reason}\n{contains}\n{hit.source_url}\n{doi}\n\n")

        for i, hit in enumerate(hits, 1):
            _print_hit(i, hit, f_curated)
    
    save_registry(result, res_dir)
    print(f"\n[✓] All results saved to: {res_dir}", flush=True)
    print("\n" + "="*70, flush=True)
    return {"hits": [h.model_dump() for h in hits]}

def main() -> None:
    parser = argparse.ArgumentParser(description="Unified Literature Discovery Agent")
    parser.add_argument("query", nargs="?", default=None, help="Search topic or mismatch sentence.")
    parser.add_argument("--query", dest="query_flag", type=str, help="Alternative way to provide query.")
    parser.add_argument("--max-results", type=int, default=30, help="Results to retrieve.")
    parser.add_argument("--download-si", action="store_true", help="Download SI files.")
    parser.add_argument("--download-papers", action="store_true", help="Download full OA paper PDFs.")
    parser.add_argument("--mode", choices=["agent", "manual"], default="agent", help="Discovery mode (default: agent).")
    parser.add_argument("--outdir", type=str, default="outputs", help="Output directory.")
    parser.add_argument("--snowball", action="store_true", help="Expand discovery by traversing citations/references.")
    
    args = parser.parse_args()
    
    query = args.query or args.query_flag
    if not query and Path("query.txt").exists():
        query = "query.txt"
        print(f"Loading default query from {query}...", flush=True)
    
    if not query:
        print("Error: No query provided. Use: python -m accelerator.runner 'your query' or create a 'query.txt' file.")
        sys.exit(1)

    # Mode logic
    use_llm = (args.mode == "agent")

    # File input support
    search_input = query
    p = Path(query)
    if p.exists() and p.is_file():
        if query.endswith(".json"):
            with open(query, "r") as f:
                search_input = SearchQuery(**json.load(f))
        elif query.endswith(".yaml") or query.endswith(".yml"):
            import yaml
            with open(query, "r") as f:
                search_input = SearchQuery(**yaml.safe_load(f))
        elif query.endswith(".txt") or query == "query.txt":
            with open(query, "r") as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
                if not lines:
                    search_input = ""
                elif len(lines) == 1:
                    search_input = lines[0]
                else:
                    search_input = lines
                    print(f"Loaded {len(lines)} queries from {query}.", flush=True)
    else:
        # Check if it looks like JSON string
        try:
            data = json.loads(query)
            if isinstance(data, dict): search_input = SearchQuery(**data)
        except: pass

    run(search_input, max_results=args.max_results, download_si=args.download_si, download_papers=args.download_papers, use_llm=use_llm, outdir=args.outdir, snowball=args.snowball)

if __name__ == "__main__":
    main()
