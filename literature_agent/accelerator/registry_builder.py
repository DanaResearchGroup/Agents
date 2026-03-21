from __future__ import annotations
import logging
import json
import hashlib
from typing import Any, List, Optional, Literal
from pathlib import Path
from pydantic import BaseModel, Field
from .models import PaperRecord, SearchQuery, RegistryResult, FailureBrief
from .clients import CrossrefClient, OpenAlexClient, EuropePMCClient, SemanticScholarClient, WebSearchClient
from .utils import call_deepseek, download_file

logger = logging.getLogger(__name__)

# --- 1. REGISTRY BUILDER & DISCOVERY ENGINE ---

INTAKE_PROMPT_SYSTEM = "You are a Chemical Kinetics Expert Agent."
INTAKE_PROMPT_USER = """
Convert this human discrepancy report into a structured JSON 'Failure Brief'.
SENTENCE: "{sentence}"
REQUIRED JSON STRUCTURE:
{{
  "raw_sentence": "{sentence}",
  "targets": [{{ "observable": "string", "direction": "string", "severity": 0.5, "notes": "string" }}],
  "conditions": {{ "reactor": "string", "T_K": 1000.0, "P_bar": 1.0, "fuel": "string" }},
  "focus_set": {{ "species": ["string"], "tags": ["string"] }},
  "evidence_needs": ["string"],
  "keywords": ["string"]
}}
Output ONLY the raw JSON.
"""

QUERY_GEN_SYSTEM = "You are a Chemical Kinetics Research Assistant."
QUERY_GEN_USER = """
Based on this Failure Brief, generate {num} distinct, highly targeted search queries for scientific literature.
Queries should be CONCISE (max 10 words each) to avoid API length limits.
failure_brief: {brief_json}
Output ONLY a JSON object: {{"queries": ["query1", "query2", ...]}}
"""

FILTER_PROMPT_SYSTEM = "You are a Chemical Kinetics Librarian Agent."
FILTER_PROMPT_USER = """
Is this search result relevant to chemical kinetics for {fuel}? 
TITLE: "{title}"
ABSTRACT: "{abstract}"
SI_ADVERTISED: {si_advertised}
Return ONLY a JSON object: 
{{ "keep": true/false, "match_reason": "string", "likely_contains": "string", "best_use": "string" }}
"""

SI_KEYWORDS = ["supplementary", "supporting info", "sm", "appendix", "table s", "si"]
MECH_KEYWORDS = ["chemkin", "cantera", "mechanism file", ".inp", ".cti", ".yaml", "ctml"]

class RegistryBuilder:
    def __init__(self, mailto: str = "your-email@example.com", use_llm: bool = False):
        import os
        self.mailto = mailto
        self.use_llm = use_llm
        self.crossref = CrossrefClient(mailto=mailto)
        self.openalex = OpenAlexClient(mailto=mailto)
        self.epmc = EuropePMCClient(mailto=mailto)
        s2_key = os.getenv("S2_API_KEY")
        self.s2 = SemanticScholarClient(api_key=s2_key)

####### Step 1: Intake ########

    def intake_sentence(self, sentence: str) -> FailureBrief:
        if self.use_llm:
            prompt = INTAKE_PROMPT_USER.format(sentence=sentence)
            llm_json = call_deepseek(prompt, system_prompt=INTAKE_PROMPT_SYSTEM)
            if llm_json: return FailureBrief(**llm_json)
        
        case_hash = hashlib.md5(sentence.encode()).hexdigest()[:8]
        return FailureBrief(raw_sentence=sentence, case_id=f"case_{case_hash}", keywords=[w for w in sentence.split() if len(w) > 4][:5])

####### Step 2: Query Generation ########

    def _generate_agentic_queries(self, brief: FailureBrief, num: int = 2) -> List[str]:
        if self.use_llm:
            prompt = QUERY_GEN_USER.format(num=num, brief_json=brief.model_dump_json())
            llm_json = call_deepseek(prompt, system_prompt=QUERY_GEN_SYSTEM)
            if llm_json: return llm_json.get("queries", [])
        return [f"{brief.conditions.get('fuel', '')} {kw}" for kw in brief.keywords[:num]]

####### Step 3: Build Registry ########

    def build_registry(self, query: SearchQuery | FailureBrief | str | List[str], download_si: bool = False, outdir: str = "outputs", per_strategy_limit: int = 4, snowball: bool = False) -> RegistryResult:
        raw_query = None
        if isinstance(query, str):
            raw_query = query
            query = self.intake_sentence(query)
        
        if isinstance(query, list):
            # 1. Multiple raw queries
            search_topics, keywords, fuel_focus = query, [], "relevant topic"
            logger.info(f"Using manual search topics: {search_topics}")
        elif isinstance(query, FailureBrief):
            # 2. Agentic Discovery (interpreted from raw input)
            keywords, fuel_focus = query.keywords, query.conditions.get('fuel', 'relevant topic')
            agentic_topics = self._generate_agentic_queries(query) if self.use_llm else []
            search_topics = list(set([raw_query] + agentic_topics)) if raw_query else agentic_topics
            logger.info(f"Using search strategies: {search_topics}")
        elif isinstance(query, SearchQuery):
            # 3. Manual SearchQuery object
            search_topics, keywords, fuel_focus = [query.topic], query.keywords, "relevant topic"
            logger.info(f"Using manual search topic: {search_topics}")
        else:
            raise ValueError(f"Unsupported query type: {type(query)}")

        registry, raw_metadata = {}, {}
        for topic in search_topics:
            sq = SearchQuery(topic=topic, max_results=per_strategy_limit)
            for client in [self.openalex]:
                for item in client.search(sq):
                    rec = client.normalize(item)
                    self._update_registry(registry, raw_metadata, rec, client.__class__.__name__[:2].lower(), item)

        results = []
        si_dir = Path(outdir) / "si_files"
        # Initial Filtering Pass
        for pid, record in registry.items():
            # Europe PMC enrichment (Try DOI first, then fallback to title)
            epmc_item = self.epmc.get_by_doi(record.doi) if record.doi else None
            if not epmc_item and record.title:
                epmc_item = self.epmc.get_by_title(record.title)

            if epmc_item:
                epmc_rec = self.epmc.normalize(epmc_item)
                logger.info(f"   [EPMC] Found record for: {record.title[:40]}...")
                if not record.journal: record.journal = epmc_rec.journal
                if epmc_rec.si_link_found:
                    logger.info(f"   [EPMC] Found {len(epmc_rec.si_links)} potential SI links.")
                    record.si_link_found = True
                    for link in epmc_rec.si_links:
                        if link not in record.si_links: record.si_links.append(link)
            else:
                logger.debug(f"   [EPMC] No record found for: {record.title[:40]}...")

            self._discover_si_links(record, meta=raw_metadata.get(pid, {}))
            ##### LLM Filter #####
            if not self._llm_filter(record, fuel_focus): continue
            self._assign_tags_and_priority(record)
            results.append(record)

        # Citation Snowballing Pass (Second Pass)
        if snowball:
            logger.info("Starting Citation Snowballing Expansion...")
            high_priority = [r for r in results if r.priority == "high"]
            seen_ids = set(registry.keys())
            snowball_results = []
            
            for seed in high_priority:
                oa_id = seed.id.split("/")[-1] if "openalex" in (seed.provenance or "") else None
                if not oa_id: continue

                # 1. Forward Snowballing (Works that cite this seed)
                citing_works = self.openalex.get_citations(oa_id, limit=5)
                # 2. Backward Snowballing (Works cited by this seed)
                referenced_works = self.openalex.get_works(seed.references[:10]) # Top 10 references
                
                for item in (citing_works + referenced_works):
                    rec = self.openalex.normalize(item)
                    if rec.id not in seen_ids:
                        seen_ids.add(rec.id)
                        if self._llm_filter(rec, fuel_focus):
                            self._assign_tags_and_priority(rec)
                            snowball_results.append(rec)
            
            logger.info(f"Snowballing found {len(snowball_results)} additional relevant papers.")
            results.extend(snowball_results)

        # Final Handling
        for record in results:
                # Resolve FETCH_REAL links from Europe PMC
                real_links = []
                for link in record.si_links:
                    if link.startswith("FETCH_REAL: "):
                        pmcid = link.replace("FETCH_REAL: ", "")
                        real_links.extend(self.epmc.get_supplementary_links(pmcid))
                    else:
                        real_links.append(link)

                record.si_links = list(set(real_links))
                if record.si_links: record.si_link_found = True
                
                if record.si_link_found:
                    self._download_si_files(record, si_dir)

        return RegistryResult(papers=results)

    def _llm_filter(self, record: PaperRecord, fuel: str) -> bool:
        if not self.use_llm: return True
        prompt = FILTER_PROMPT_USER.format(fuel=fuel, title=record.title, abstract=(record.abstract or "")[:1000], si_advertised=record.si_link_found)
        analysis = call_deepseek(prompt, system_prompt=FILTER_PROMPT_SYSTEM)
        if analysis and isinstance(analysis, dict):
            if not analysis.get("keep", True):
                reason = analysis.get("match_reason", "No reason provided.")
                logger.info(f"FILTERED OUT: {record.title[:60]}... (Reason: {reason})")
                return False
            record.match_reason, record.likely_contains, record.best_use = analysis.get("match_reason"), analysis.get("likely_contains"), analysis.get("best_use")
        return True

    def _discover_si_links(self, record: PaperRecord, meta: dict = None):
        """Integrated SI discovery logic (consolidated from si_discovery.py)."""
        links, mech_links = set(record.si_links), set(record.mechanism_links)
        if meta:
            # Check Crossref/OpenAlex/EPMC metadata
            for m in meta.values():
                if m and 'supplementary' in str(m).lower(): record.si_link_found = True

        text = f"{record.title or ''} {record.abstract or ''}".lower()
        if any(kw in text for kw in SI_KEYWORDS): record.si_link_found = True
        if any(kw in text for kw in MECH_KEYWORDS): record.mechanism_link_found = True
        record.si_links, record.mechanism_links = list(links), list(mech_links)
        if record.si_links: record.si_link_found = True

    def _assign_tags_and_priority(self, record: PaperRecord) -> None:
        tags = [record.oa_status] if record.oa_status in ["oa", "non_oa"] else ["unknown_access"]
        text = f"{record.title or ''} {record.abstract or ''}".lower()
        for kw in ["chemkin", "cantera", "mechanism", "kinetic"]:
            if kw in text: tags.append(f"possible_{kw}" if kw != "kinetic" else "kinetics_relevant")
        if record.si_link_found: tags.append("possible_si")
        
        record.priority = "high" if record.match_reason else "medium"
        tags.append(f"{record.priority}_interest")
        if record.oa_status == "non_oa" and record.priority == "high":
            record.manual_review_needed = record.keep_for_manual_review = True
            tags.append("manual_review_needed")
        record.tags = list(set(tags))

    def _update_registry(self, registry, raw_metadata, record, source_key, raw_item):
        key = record.doi.lower() if record.doi else record.id
        if key in registry:
            existing = registry[key]
            if not existing.abstract and record.abstract: existing.abstract = record.abstract
            if not existing.oa_url and record.oa_url: existing.oa_url = record.oa_url
            raw_metadata[key][source_key] = raw_item
        else:
            registry[key], raw_metadata[key] = record, {source_key: raw_item}

    def _download_si_files(self, record, si_dir):
        if not record.si_links:
            logger.info(f"   [SI] skipping {record.title[:40]}... (si_link_found is True but no URLs resolved)")
            return
        
        safe_id = record.id.replace("/", "_").replace(":", "_")
        for i, url in enumerate(record.si_links):
            fname = f"{safe_id}_si_{i}.file"
            download_url = url.split(": ", 1)[-1] if ": " in url else url
            filepath = download_file(download_url, si_dir, fname)
            record.access_notes = (record.access_notes or "") + (f" Saved SI to {filepath.name}. " if filepath else f" Failed download {i}. ")
