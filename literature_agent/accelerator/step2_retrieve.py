from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any
import requests
from .models import FailureBrief, SourceHit

S2_SEARCH_ENDPOINT = "https://api.semanticscholar.org/graph/v1/paper/search"
CACHE_DIR = Path(".cache/accelerator")

import re

# --- RETRIEVAL CONFIGURATION ---
NUM_SEARCH_STRATEGIES = 4  # How many different queries the AI should brainstorm
ROWS_PER_QUERY = 4        # How many results to fetch for EACH query
# ------------------------------

from .llm_utils import call_deepseek

# Agentic Query Generation Prompt
QUERY_GEN_SYSTEM = "You are a Chemical Kinetics Research Assistant."
QUERY_GEN_USER = """
Based on this Failure Brief, generate {num} distinct, highly targeted search queries for scientific literature.
Each query should focus on a different aspect (e.g., specific reaction pathways, chemical species, reactor conditions).

CRITICAL: At least 2 queries MUST include keywords like "supplementary information", "supporting info", or "kinetic data" to favor papers with datasets.
GUIDANCE: Ensure at least one SI-focused query is BROAD (e.g., "{fuel} pyrolysis kinetics supplementary information") to capture more results, while others can be specific to JSR/ST conditions.

FAILURE BRIEF:
{brief_json}

RULES:
- Do not over-specify (e.g., avoid including exact 900K/1bar in EVERY query as it limits results).
- Focus on kinetic mechanisms and experimental profiles.
- Output ONLY a JSON object: {{"queries": ["query1", "query2", ...]}}
"""

# Agentic Filter Prompt
FILTER_PROMPT_SYSTEM = "You are a Chemical Kinetics Librarian Agent."
FILTER_PROMPT_USER = """
Is this search result relevant to chemical kinetics for {fuel}? 

RESULT TITLE: "{title}"
RESULT VENUE: "{venue}"
ABSTRACT: "{abstract}"
HAS_SI_KEYWORDS_IN_METADATA: {si_advertised}

RULES:
- Reject non-kinetics (e.g., ritual knives, medicine).
- EXTREMELY IMPORTANT: Prioritize papers that mention 'supplementary information' or 'supporting data'.
- If a paper has NO chance of containing SI, set keep=false if strict SI filtering is requested.

OUTPUT FORMAT:
Return ONLY a JSON object: 
{{
  "keep": true/false, 
  "rationale": "one sentence reason",
  "match_reason": "why this matches",
  "likely_contains": "what kinetic data is expected",
  "best_use": "how to use this"
}}
"""

def _generate_queries(brief: FailureBrief) -> list[str]:
    brief_json = brief.model_dump_json()
    prompt = QUERY_GEN_USER.format(num=NUM_SEARCH_STRATEGIES, brief_json=brief_json, fuel=brief.conditions.fuel)
    
    llm_json = call_deepseek(prompt, system_prompt=QUERY_GEN_SYSTEM)
    
    if llm_json and "queries" in llm_json:
        return llm_json["queries"]
    
    # Fallback to basic queries
    fuel = brief.conditions.fuel or "chemical"
    return [
        f"{fuel} oxidation kinetics",
        f"{fuel} JSR CO CO2 profiles",
        f"{brief.conditions.fuel} reaction mechanism",
        f"HOCO radical kinetics" # Domain-specific fallback
    ]

def _call_llm_filter(item: dict[str, Any], brief: FailureBrief, si_advertised: bool = False) -> dict[str, Any]:
    title = item.get("title", "Untitled")
    venue = item.get("venue") or ""
    abstract = item.get("abstract") or "No abstract available."
    
    prompt = FILTER_PROMPT_USER.format(
        fuel=brief.conditions.fuel, 
        title=title, 
        venue=venue,
        abstract=abstract[:2000], # Cap abstract length
        si_advertised=si_advertised
    )
    llm_json = call_deepseek(prompt, system_prompt=FILTER_PROMPT_SYSTEM)
    
    if llm_json is None:
        # Simulation Fallback
        lower_title = title.lower()
        if "fish" in lower_title or "ceremony" in lower_title:
            return {"keep": False, "rationale": "Irrelevant: Detected cultural topic."}
        return {
            "keep": True, 
            "rationale": "Relevant: Likely kinetics study.",
            "match_reason": "Matches kinetics keywords.",
            "likely_contains": "Kinetic parameters.",
            "best_use": "Model refinement."
        }
    
    return llm_json

def _search_s2(query: str, limit: int = ROWS_PER_QUERY) -> list[dict]:
    import time
    params = {
        "query": query,
        "limit": limit,
        "fields": "title,authors,year,venue,externalIds,abstract,url"
    }
    
    # S2 Public API limit is ~1 req/sec. Adding deliberate delay + retries.
    time.sleep(1.0) 
    
    for attempt in range(3):
        try:
            response = requests.get(S2_SEARCH_ENDPOINT, params=params, timeout=30)
            if response.status_code == 429:
                delay = 2 ** attempt
                print(f"       (S2 Rate limited. Waiting {delay}s before retry...)")
                time.sleep(delay)
                continue
            response.raise_for_status()
            return response.json().get("data", []) or []
        except Exception as e:
            print(f"       (S2 Search Error: {e})")
            return []
    return []



from .clients.crossref_client import CrossrefClient
from .clients.europepmc_client import EuropePMCClient
from .clients.web_search_client import WebSearchClient

# Keywords for SI detection
SI_KEYWORDS = [
    "supplementary", "supporting information", "appendices", "appendix", 
    "dataset", "repository", "si material", "electronic supplementary"
]

def _discover_si_in_hit(item: dict[str, Any], doi: str | None) -> tuple[bool, list[str]]:
    """Helper to find possible SI links in metadata, prioritizing Europe PMC."""
    links = set()
    
    # 1. Europe PMC Enrichment (Primary for automated SI)
    if doi or item.get("title"):
        try:
            epmc = EuropePMCClient()
            epmc_data = None
            if doi:
                epmc_data = epmc.get_by_doi(doi)
            if not epmc_data and item.get("title"):
                epmc_data = epmc.get_by_title(item.get("title"))
                
            if epmc_data:
                if epmc_data.get("hasSuppl") == "Y":
                    pmcid = epmc_data.get("pmcid")
                    if pmcid:
                        direct_links = epmc.get_si_metadata(pmcid)
                        if direct_links:
                            links.update(direct_links)
                        else:
                            links.add(f"https://europepmc.org/articles/{pmcid}#sec5")
        except: pass

    # 2. Crossref Enrichment (Secondary)
    if doi:
        try:
            cr = CrossrefClient()
            if "10.1021" in doi: # ACS pattern fallback
                links.add(f"https://pubs.acs.org/doi/suppl/{doi}/suppl_file/{doi.split('/')[-1]}.s001.pdf")
        except: pass

    # 3. Semantic Scholar metadata / general URL
    url = item.get("url")
    if url and any(kw in url.lower() for kw in SI_KEYWORDS):
        links.add(url)

    # 4. Deep Discovery Web Search Fallback
    if not links:
        try:
            web_search = WebSearchClient()
            title = item.get("title")
            if title or doi:
                web_links = web_search.find_si_links(title=title, doi=doi)
                if web_links:
                    links.update(web_links)
        except: pass

    return len(links) > 0, list(links)

def _to_source_hit(item: dict[str, Any], score: float, analysis: dict[str, Any], found_si: bool = False, si_links: list[str] = []) -> SourceHit:
    title = item.get("title", "Untitled")
    doi = item.get("externalIds", {}).get("DOI") if "externalIds" in item else item.get("doi")
    
    authors = []
    for a in item.get("authors", []):
        if "name" in a:
            authors.append(a["name"])
        elif "family" in a:
            authors.append(f"{a.get('given', '')} {a['family']}".strip())

    return SourceHit(
        title=title,
        doi=doi,
        url=item.get("url"),
        year=item.get("year"),
        authors=authors,
        venue=item.get("venue"),
        abstract=item.get("abstract"),
        provenance=item.get("provenance", "semantic_scholar"),
        evidence_type=["experimental_study"] if item.get("provenance") != "rmg_database" else ["database_entry"],
        score=score,
        reasons=[analysis.get("rationale", "")],
        match_reason=analysis.get("match_reason"),
        likely_contains=analysis.get("likely_contains"),
        best_use=analysis.get("best_use"),
        si_link_found=found_si,
        si_links=si_links
    )

def retrieve_sources(brief: FailureBrief, rows_per_query: int = ROWS_PER_QUERY) -> list[SourceHit]:
    # 1. GENERATE TARGETED QUERIES
    queries = _generate_queries(brief)
    print(f"       -> AI generated {len(queries)} specialized search strategies.", flush=True)
    
    results_map: dict[str, SourceHit] = {} # DOI/Title -> SourceHit for deduplication
    all_raw_literatures = []

    # 2. EXECUTE SEARCHES
    cr = CrossrefClient()
    
    for query in queries:
        # A. Semantic Scholar Discovery
        print(f"       (Searching Semantic Scholar: '{query}')", flush=True)
        s2_items = _search_s2(query, limit=rows_per_query)
        
        # B. Crossref Discovery
        print(f"       (Searching Crossref: '{query}')", flush=True)
        cr_items = cr.search(query, limit=rows_per_query)
        
        # Combine items from both sources for processing
        # Note: We tag them with provenance for tracking
        combined_items = []
        for it in s2_items:
            it["provenance"] = "semantic_scholar"
            combined_items.append(it)
        for it in cr_items:
            it["provenance"] = "crossref"
            # Normalize title to string if it's a list (Crossref style)
            if isinstance(it.get("title"), list) and it["title"]:
                it["title"] = it["title"][0]
            combined_items.append(it)

        for item in combined_items:
            if not item or not isinstance(item, dict):
                continue
            # 3. AGENTIC FILTERING & DEDUPLICATION
            # Unify DOI/Title extraction
            doi = (item.get("externalIds", {}).get("DOI") or item.get("doi") or item.get("DOI") or "").lower()
            title = item.get("title", "").lower()
            dedup_key = doi if doi else title
            
            if not dedup_key or dedup_key in results_map:
                continue
                
            # Persistence for Demo: Track all found items before filtering
            raw_info = {
                "title": item.get("title"),
                "venue": item.get("venue") or item.get("container-title", [None])[0],
                "year": item.get("year") or (item.get("issued", {}).get("date-parts", [[None]])[0][0]),
                "abstract": (item.get("abstract") or "No abstract")[:200] + "..."
            }
            if raw_info not in all_raw_literatures:
                all_raw_literatures.append(raw_info)

            # Check for SI BEFORE filtering to inform the LLM
            found_si, si_links = _discover_si_in_hit(item, doi)
            
            # AGENTIC FILTERING
            analysis = _call_llm_filter(item, brief, si_advertised=found_si)
            
            if analysis.get("keep", True):
                prov = item.get('provenance', 'unknown')
                title_disp = (item.get('title') or 'Untitled')[:60]
                print(f"         [✓] ACCEPTED ({prov}{' +SI' if found_si else ''}): {title_disp}...", flush=True)
                results_map[dedup_key] = _to_source_hit(
                    item, 1.0, analysis, found_si=found_si, si_links=si_links
                )
            else:
                prov = item.get('provenance', 'unknown')
                title_disp = (item.get('title') or 'Untitled')[:60]
                print(f"         [✗] REJECTED ({prov}{' +SI' if found_si else ''}): {title_disp}...", flush=True)
    
    # Save raw results for demonstration
    with open("raw_found_sources.txt", "w") as f:
        f.write("=== RAW UNFILTERED SEARCH RESULTS (SEMANTIC SCHOLAR) ===\n")
        f.write(f"Total entries found: {len(all_raw_literatures)}\n\n")
        for i, res in enumerate(all_raw_literatures, 1):
            f.write(f"{i}. {res['title']} ({res['venue']}, {res['year']})\n")
            f.write(f"   Abstract snippet: {res['abstract']}\n\n")

    return list(results_map.values())
