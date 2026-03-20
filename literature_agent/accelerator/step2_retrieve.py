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

S2_SEARCH_ENDPOINT = "https://api.semanticscholar.org/graph/v1/paper/search"
CACHE_DIR = Path(".cache/accelerator")
RMG_DATABASE_PATH = Path("/Users/jintao/Documents/Code/RMG-database")

# --- RETRIEVAL CONFIGURATION ---
NUM_SEARCH_STRATEGIES = 2  # How many different queries the AI should brainstorm
ROWS_PER_QUERY = 2        # How many results to fetch for EACH query
# ------------------------------

from .llm_utils import call_deepseek

# Agentic Query Generation Prompt
QUERY_GEN_SYSTEM = "You are a Chemical Kinetics Research Assistant."
QUERY_GEN_USER = f"""
Based on this Failure Brief, generate {{num}} distinct, highly targeted search queries for scientific literature.
Each query should focus on a different aspect (e.g., specific reaction pathways, chemical species, reactor conditions).
""" + """
FAILURE BRIEF:
{brief_json}

RULES:
- Focus on kinetic mechanisms (e.g., 'HOCHO oxidation mechanisms').
- Focus on specific species (e.g., 'HOCO radical kinetics').
- Focus on experimental data (e.g., 'formic acid JSR CO CO2 profiles').
- Output ONLY a JSON object with a list of strings: {{"queries": ["query1", "query2", ...]}}
"""

# Agentic Filter Prompt
FILTER_PROMPT_SYSTEM = "You are a Chemical Kinetics Librarian Agent."
FILTER_PROMPT_USER = """
Is this search result relevant to chemical kinetics, combustion, or thermochemistry for {fuel}?

RESULT TITLE: "{title}"
RESULT VENUE: "{venue}"
ABSTRACT: "{abstract}"

RULES:
- Reject Japanese ritual knives (Hocho) or dietary habits.
- Keep chemical kinetics, mechanics, or experimental studies.
- Use the abstract to verify if actual kinetic data is likely inside.

OUTPUT FORMAT:
Return ONLY a JSON object: 
{{
  "keep": true/false, 
  "rationale": "one sentence reason based on abstract",
  "match_reason": "why this matches kinetics/discrepancy (1 sentence)",
  "likely_contains": "what kinetic data is expected inside (1 sentence)",
  "best_use": "how to use this in a model update (1 sentence)"
}}
"""

def _generate_queries(brief: FailureBrief) -> list[str]:
    # ... (remains same)
    brief_json = brief.model_dump_json()
    prompt = QUERY_GEN_USER.format(num=NUM_SEARCH_STRATEGIES, brief_json=brief_json)
    
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

def _call_llm_filter(item: dict[str, Any], brief: FailureBrief) -> dict[str, Any]:
    # ... (remains same)
    title = item.get("title", "Untitled")
    venue = item.get("venue") or ""
    abstract = item.get("abstract") or "No abstract available."
    
    prompt = FILTER_PROMPT_USER.format(
        fuel=brief.conditions.fuel, 
        title=title, 
        venue=venue,
        abstract=abstract[:2000] # Cap abstract length
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
    # ... (remains same)
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
            response = requests.get(S2_SEARCH_ENDPOINT, params=params, timeout=15)
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

def _search_rmg(brief: FailureBrief) -> list[dict]:
    """
    Search the local RMG-database for relevant reactions using regex.
    This avoids the overhead of loading the full RMG database libraries.
    """
    hits = []
    fuel = (brief.conditions.fuel or "").strip()
    if not fuel:
        return []

    # Potential synonym/label search (e.g., HOCHO, formic acid)
    search_terms = {fuel.lower()}
    if fuel.lower() == "formic acid":
        search_terms.add("hocho")
    elif fuel.lower() == "hocho":
        search_terms.add("formic acid")
    
    print(f"       (Searching RMG-database for: {search_terms})", flush=True)
    
    kinetics_path = RMG_DATABASE_PATH / "input" / "kinetics"
    if not kinetics_path.exists():
        return []

    reaction_files = list(kinetics_path.glob("libraries/*/reactions.py")) + \
                     list(kinetics_path.glob("families/*/reactions.py"))

    for fpath in reaction_files:
        try:
            content = fpath.read_text()
            
            # Extract Mechanism Metadata (name, shortDesc, longDesc)
            mech_name_match = re.search(r'^name\s*=\s*["\'](.*?)["\']', content, re.MULTILINE)
            mech_short_desc_match = re.search(r'^shortDesc\s*=\s*([\"\']{1,3})(.*?)\1', content, re.MULTILINE | re.DOTALL)
            mech_long_desc_match = re.search(r'^longDesc\s*=\s*([\"\']{1,3})(.*?)\1', content, re.MULTILINE | re.DOTALL)
            
            mech_name = mech_name_match.group(1) if mech_name_match else fpath.parent.name
            mech_short = mech_short_desc_match.group(2).strip() if mech_short_desc_match else ""
            mech_long = mech_long_desc_match.group(2).strip() if mech_long_desc_match else ""

            # Find entry blocks.
            entry_chunks = re.split(r"^entry\(", content, flags=re.MULTILINE)
            for entry_text in entry_chunks[1:]: # Skip text before first entry
                # Check if any search term is in label
                label_match = re.search(r'label\s*=\s*["\'](.*?)["\']', entry_text)
                if label_match:
                    label = label_match.group(1)
                    if any(term in label.lower() for term in search_terms):
                        # Extract Entry Metadata
                        index_match = re.search(r'index\s*=\s*(\d+)', entry_text)
                        entry_short_match = re.search(r'shortDesc\s*=\s*([\"\']{1,3})(.*?)\1', entry_text, re.DOTALL)
                        entry_long_match = re.search(r'longDesc\s*=\s*([\"\']{1,3})(.*?)\1', entry_text, re.DOTALL)
                        
                        index = index_match.group(1) if index_match else "?"
                        entry_short = entry_short_match.group(2).strip() if entry_short_match else ""
                        entry_long = entry_long_match.group(2).strip() if entry_long_match else ""

                        # Extract Structured Reference (Article, Book, etc.)
                        ref_match = re.search(r'reference\s*=\s*(Article|Book|Thesis|Reference)\((.*?)\s*\),', entry_text, re.DOTALL)
                        ref_data = {}
                        if ref_match:
                            ref_type = ref_match.group(1)
                            ref_body = ref_match.group(2)
                            # Simple extraction of common fields from structured ref
                            for field in ['authors', 'title', 'journal', 'year', 'doi', 'publisher']:
                                f_match = re.search(rf'{field}\s*=\s*(?:\[(.*?)\]|[\"\'](.*?)[\"\'])', ref_body, re.DOTALL)
                                if f_match:
                                    ref_data[field] = (f_match.group(1) or f_match.group(2)).strip()

                        # Consolidate hit info
                        authors = []
                        if 'authors' in ref_data:
                            # Clean up author string ['Author 1', 'Author 2']
                            authors_raw = ref_data['authors'].replace("'", "").replace('"', "").split(",")
                            authors = [{"name": a.strip()} for a in authors_raw if a.strip()]
                        elif mech_long:
                            # Use first line of longDesc as a proxy for authors if no structured ref
                            authors = [{"name": mech_long.split("\n")[0][:100]}]
                        else:
                            authors = [{"name": f"RMG {fpath.parts[-3]} Library"}]

                        year = 2024
                        if 'year' in ref_data:
                            try: year = int(re.search(r'\d{4}', ref_data['year']).group())
                            except: pass

                        doi = ref_data.get('doi', f"RMG:{mech_name}:{index}:{label}")
                        
                        abstract = f"Elementary reaction entry found in RMG {fpath.parts[-3]} mechanism '{mech_name}'.\n"
                        if entry_short: abstract += f"Short Desc: {entry_short}\n"
                        if entry_long: abstract += f"Long Desc: {entry_long}\n"
                        if ref_data.get('title'): abstract += f"Reference Title: {ref_data['title']}\n"
                        if mech_short: abstract += f"Mechanism Info: {mech_short}\n"
                        
                        hits.append({
                            "title": f"RMG Reaction: {label}",
                            "doi": doi,
                            "authors": authors,
                            "year": year,
                            "venue": ref_data.get('journal') or mech_name,
                            "abstract": abstract.strip(),
                            "url": f"file://{fpath}",
                            "provenance": "rmg_database"
                        })
                        
                        if len(hits) >= 12: # Slightly higher limit for RMG
                            return hits
        except Exception as e:
            continue
            
    return hits

def _to_source_hit(item: dict[str, Any], score: float, analysis: dict[str, Any]) -> SourceHit:
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
    )

def retrieve_sources(brief: FailureBrief, rows_per_query: int = ROWS_PER_QUERY) -> list[SourceHit]:
    # 1. GENERATE TARGETED QUERIES
    queries = _generate_queries(brief)
    print(f"       -> AI generated {len(queries)} specialized search strategies.", flush=True)
    
    results_map: dict[str, SourceHit] = {} # DOI/Title -> SourceHit for deduplication
    all_raw_literatures = []

    # 2. EXECUTE SEARCHES
    # A. Semantic Scholar
    for query in queries:
        print(f"       (Searching Semantic Scholar: '{query}')", flush=True)
        s2_items = _search_s2(query, limit=rows_per_query)
        
        for item in s2_items:
            # Persistence for Demo: Track all found items before filtering
            raw_info = {
                "title": item.get("title"),
                "venue": item.get("venue"),
                "year": item.get("year"),
                "abstract": (item.get("abstract") or "No abstract")[:200] + "..."
            }
            if raw_info not in all_raw_literatures:
                all_raw_literatures.append(raw_info)

            # Simple deduplication by title or DOI
            doi = (item.get("externalIds", {}).get("DOI") or item.get("doi") or "").lower()
            title = item.get("title", "").lower()
            dedup_key = doi if doi else title
            
            if not dedup_key or dedup_key in results_map:
                continue
                
            # 3. AGENTIC FILTERING
            analysis = _call_llm_filter(item, brief)
            
            if analysis.get("keep", True):
                print(f"         [✓] ACCEPTED: {item.get('title')[:60]}...", flush=True)
                results_map[dedup_key] = _to_source_hit(
                    item, 1.0, analysis
                )
            else:
                print(f"         [✗] REJECTED: {item.get('title')[:60]}...", flush=True)
    
    # Save raw results for demonstration
    with open("raw_found_sources.txt", "w") as f:
        f.write("=== RAW UNFILTERED SEARCH RESULTS (SEMANTIC SCHOLAR) ===\n")
        f.write(f"Total entries found: {len(all_raw_literatures)}\n\n")
        for i, res in enumerate(all_raw_literatures, 1):
            f.write(f"{i}. {res['title']} ({res['venue']}, {res['year']})\n")
            f.write(f"   Abstract snippet: {res['abstract']}\n\n")

    # B. RMG Database
    rmg_items = _search_rmg(brief)
    for item in rmg_items:
        dedup_key = item["doi"].lower()
        if dedup_key in results_map:
            continue
            
        # For database entries, we can skip LLM filter or use a simplified one
        analysis = {
            "keep": True,
            "rationale": "Relevant: Found in RMG database.",
            "match_reason": "Direct species match in database.",
            "likely_contains": "Elementary reaction rates and kinetics.",
            "best_use": "Reference kinetics for key reactions."
        }
        results_map[dedup_key] = _to_source_hit(item, 1.2, analysis) # Boost RMG score slightly
    
    return list(results_map.values())
