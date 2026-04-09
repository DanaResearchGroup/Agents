"""Registry builder: discovers, filters, and catalogues kinetics papers."""

from __future__ import annotations

import hashlib
import logging
import os
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

from src.ingestion.retrieval import (
    CrossrefClient,
    EuropePMCClient,
    OpenAlexClient,
    SemanticScholarClient,
    WebSearchClient,
)
from src.ingestion.utils import call_llm_sync, download_file
from src.schemas.ingestion import (
    FailureBrief,
    PaperRecord,
    RegistryResult,
    SearchQuery,
)

if TYPE_CHECKING:
    from src.agents.llm_client import LLMClient

logger = logging.getLogger(__name__)

_DEFAULT_EMAIL = os.environ.get("CONTACT_EMAIL", "agent@example.com")

# ── LLM prompt templates ────────────────────────────────────────────────────

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
    """Discovers, filters, and catalogues kinetics papers from multiple APIs."""

    def __init__(
        self,
        mailto: str = _DEFAULT_EMAIL,
        use_llm: bool = False,
        llm_client: LLMClient | None = None,
    ):
        self.mailto = mailto
        self.use_llm = use_llm
        self.llm_client = llm_client if use_llm else None
        self.crossref = CrossrefClient(mailto=mailto)
        self.openalex = OpenAlexClient(mailto=mailto)
        self.epmc = EuropePMCClient(mailto=mailto)
        s2_key = os.getenv("S2_API_KEY")
        self.s2 = SemanticScholarClient(api_key=s2_key)

    # ── Step 1: Intake ───────────────────────────────────────────────────

    def intake_sentence(self, sentence: str) -> FailureBrief:
        if self.use_llm:
            prompt = INTAKE_PROMPT_USER.format(sentence=sentence)
            llm_json = call_llm_sync(
                prompt, system_prompt=INTAKE_PROMPT_SYSTEM,
                llm_client=self.llm_client, agent_name="registry_intake",
            )
            if llm_json:
                return FailureBrief(**llm_json)

        case_hash = hashlib.md5(sentence.encode()).hexdigest()[:8]
        return FailureBrief(
            raw_sentence=sentence,
            case_id=f"case_{case_hash}",
            keywords=[w for w in sentence.split() if len(w) > 4][:5],
        )

    # ── Step 2: Query Generation ─────────────────────────────────────────

    def _generate_agentic_queries(self, brief: FailureBrief, num: int = 2) -> list[str]:
        if self.use_llm:
            prompt = QUERY_GEN_USER.format(num=num, brief_json=brief.model_dump_json())
            llm_json = call_llm_sync(
                prompt, system_prompt=QUERY_GEN_SYSTEM,
                llm_client=self.llm_client, agent_name="registry_query_gen",
            )
            if llm_json:
                return llm_json.get("queries", [])
        return [f"{brief.conditions.get('fuel', '')} {kw}" for kw in brief.keywords[:num]]

    # ── Step 3: Build Registry ───────────────────────────────────────────

    def build_registry(
        self,
        query: SearchQuery | FailureBrief | str | list[str],
        download_si: bool = False,
        download_papers: bool = False,
        outdir: str = "outputs",
        per_strategy_limit: int = 4,
        snowball: bool = False,
    ) -> RegistryResult:
        raw_query = None
        if isinstance(query, str):
            raw_query = query
            query = self.intake_sentence(query)

        if isinstance(query, list):
            search_topics, keywords, fuel_focus = query, [], "relevant topic"
            logger.info("Using manual search topics: %s", search_topics)
        elif isinstance(query, FailureBrief):
            keywords = query.keywords
            fuel_focus = query.conditions.get("fuel", "relevant topic")
            agentic_topics = self._generate_agentic_queries(query) if self.use_llm else []
            search_topics = list(set([raw_query] + agentic_topics)) if raw_query else agentic_topics
            logger.info("Using search strategies: %s", search_topics)
        elif isinstance(query, SearchQuery):
            search_topics, keywords, fuel_focus = [query.topic], query.keywords, "relevant topic"
            logger.info("Using manual search topic: %s", search_topics)
        else:
            raise ValueError(f"Unsupported query type: {type(query)}")

        registry: dict[str, PaperRecord] = {}
        raw_metadata: dict[str, dict] = {}
        for topic in search_topics:
            sq = SearchQuery(topic=topic, max_results=per_strategy_limit)
            for client in [self.openalex]:
                for item in client.search(sq):
                    rec = client.normalize(item)
                    self._update_registry(
                        registry, raw_metadata, rec,
                        client.__class__.__name__[:2].lower(), item,
                    )

        results: list[PaperRecord] = []
        si_dir = Path(outdir) / "si_files"
        papers_dir = Path(outdir) / "papers"

        # Initial filtering pass.
        for pid, record in registry.items():
            epmc_item = self.epmc.get_by_doi(record.doi) if record.doi else None
            if not epmc_item and record.title:
                epmc_item = self.epmc.get_by_title(record.title)

            if epmc_item:
                epmc_rec = self.epmc.normalize(epmc_item)
                logger.info("   [EPMC] Found record for: %s...", record.title[:40])
                if not record.journal:
                    record.journal = epmc_rec.journal
                if epmc_rec.si_link_found:
                    logger.info("   [EPMC] Found %d potential SI links.", len(epmc_rec.si_links))
                    record.si_link_found = True
                    for link in epmc_rec.si_links:
                        if link not in record.si_links:
                            record.si_links.append(link)
            else:
                logger.debug("   [EPMC] No record found for: %s...", record.title[:40])

            self._discover_si_links(record, meta=raw_metadata.get(pid, {}))
            if not self._llm_filter(record, fuel_focus):
                continue
            self._assign_tags_and_priority(record)
            results.append(record)

        # Citation snowballing pass.
        if snowball:
            logger.info("Starting Citation Snowballing Expansion...")
            high_priority = [r for r in results if r.priority == "high"]
            seen_ids = set(registry.keys())
            snowball_results: list[PaperRecord] = []

            for seed in high_priority:
                oa_id = seed.id.split("/")[-1] if "openalex" in (seed.provenance or "") else None
                if not oa_id:
                    continue
                citing_works = self.openalex.get_citations(oa_id, limit=5)
                referenced_works = self.openalex.get_works(seed.references[:10])

                for item in citing_works + referenced_works:
                    rec = self.openalex.normalize(item)
                    if rec.id not in seen_ids:
                        seen_ids.add(rec.id)
                        if self._llm_filter(rec, fuel_focus):
                            self._assign_tags_and_priority(rec)
                            snowball_results.append(rec)

            logger.info("Snowballing found %d additional relevant papers.", len(snowball_results))
            results.extend(snowball_results)

        # Final handling: resolve SI links, download.
        for record in results:
            real_links: list[str] = []
            for link in record.si_links:
                if link.startswith("FETCH_REAL: "):
                    pmcid = link.replace("FETCH_REAL: ", "")
                    real_links.extend(self.epmc.get_supplementary_links(pmcid))
                else:
                    real_links.append(link)

            record.si_links = list(set(real_links))
            if record.si_links:
                record.si_link_found = True

            if download_si and record.si_link_found:
                self._download_si_files(record, si_dir)

            if download_papers and record.oa_url:
                self._download_paper(record, papers_dir)

        return RegistryResult(papers=results)

    # ── Internal helpers ─────────────────────────────────────────────────

    def _llm_filter(self, record: PaperRecord, fuel: str) -> bool:
        if not self.use_llm:
            return True
        prompt = FILTER_PROMPT_USER.format(
            fuel=fuel, title=record.title,
            abstract=(record.abstract or "")[:1000],
            si_advertised=record.si_link_found,
        )
        analysis = call_llm_sync(
            prompt, system_prompt=FILTER_PROMPT_SYSTEM,
            llm_client=self.llm_client, agent_name="registry_filter",
        )
        if analysis and isinstance(analysis, dict):
            if not analysis.get("keep", True):
                reason = analysis.get("match_reason", "No reason provided.")
                logger.info("FILTERED OUT: %s... (Reason: %s)", record.title[:60], reason)
                return False
            record.match_reason = analysis.get("match_reason")
            record.likely_contains = analysis.get("likely_contains")
            record.best_use = analysis.get("best_use")
        return True

    def _discover_si_links(self, record: PaperRecord, meta: dict | None = None) -> None:
        links = set(record.si_links)
        mech_links = set(record.mechanism_links)
        if meta:
            for m in meta.values():
                if m and "supplementary" in str(m).lower():
                    record.si_link_found = True

        text = f"{record.title or ''} {record.abstract or ''}".lower()
        if any(kw in text for kw in SI_KEYWORDS):
            record.si_link_found = True
        if any(kw in text for kw in MECH_KEYWORDS):
            record.mechanism_link_found = True
        record.si_links = list(links)
        record.mechanism_links = list(mech_links)
        if record.si_links:
            record.si_link_found = True

    def _assign_tags_and_priority(self, record: PaperRecord) -> None:
        tags = [record.oa_status] if record.oa_status in ("oa", "non_oa") else ["unknown_access"]
        text = f"{record.title or ''} {record.abstract or ''}".lower()
        for kw in ["chemkin", "cantera", "mechanism", "kinetic"]:
            if kw in text:
                tags.append(f"possible_{kw}" if kw != "kinetic" else "kinetics_relevant")
        if record.si_link_found:
            tags.append("possible_si")

        record.priority = "high" if record.match_reason else "medium"
        tags.append(f"{record.priority}_interest")
        if record.oa_status == "non_oa" and record.priority == "high":
            record.manual_review_needed = record.keep_for_manual_review = True
            tags.append("manual_review_needed")
        record.tags = list(set(tags))

    def _update_registry(
        self, registry: dict, raw_metadata: dict,
        record: PaperRecord, source_key: str, raw_item: dict,
    ) -> None:
        key = record.doi.lower() if record.doi else record.id
        if key in registry:
            existing = registry[key]
            if not existing.abstract and record.abstract:
                existing.abstract = record.abstract
            if not existing.oa_url and record.oa_url:
                existing.oa_url = record.oa_url
            raw_metadata[key][source_key] = raw_item
        else:
            registry[key] = record
            raw_metadata[key] = {source_key: raw_item}

    def _download_si_files(self, record: PaperRecord, si_dir: Path) -> None:
        if not record.si_links:
            logger.info(
                "   [SI] skipping %s... (si_link_found is True but no URLs resolved)",
                record.title[:40],
            )
            return

        safe_id = record.id.replace("/", "_").replace(":", "_")
        for i, url in enumerate(record.si_links):
            fname = f"{safe_id}_si_{i}.file"
            download_url = url.split(": ", 1)[-1] if ": " in url else url
            filepath = download_file(download_url, si_dir, fname)
            note = f" Saved SI to {filepath.name}. " if filepath else f" Failed download {i}. "
            record.access_notes = (record.access_notes or "") + note

            # Inspect ZIPs for mechanism files.
            if filepath and zipfile.is_zipfile(filepath):
                extract_dir = si_dir / f"{safe_id}_extracted_{i}"
                extract_dir.mkdir(parents=True, exist_ok=True)
                found = self._extract_mechanism_from_zip(filepath, extract_dir)
                record.mechanism_files.extend(str(p) for p in found)
                logger.info(
                    "   [SI] %d mechanism files extracted from %s",
                    len(found), filepath.name,
                )
                if not found:
                    logger.info(
                        "   [SI] No mechanism files in %s (images/figures only)",
                        filepath.name,
                    )

        # Select the best mechanism across all extracted files.
        if record.mechanism_files:
            self._select_best_mechanism(record)

    def _extract_mechanism_from_zip(
        self, zip_path: Path, extract_dir: Path,
    ) -> list[Path]:
        """Recursively extract chemistry files from ZIP and nested ZIPs.

        Returns list of extracted mechanism file paths.
        """
        _CHEMKIN_EXT = {".inp", ".dat", ".ck"}
        _CANTERA_EXT = {".yaml", ".yml"}
        _CHEMKIN_KEYWORDS = {"ELEMENTS", "SPECIES", "REACTIONS", "THERMO"}
        found: list[Path] = []

        try:
            with zipfile.ZipFile(zip_path) as zf:
                for name in zf.namelist():
                    ext = Path(name).suffix.lower()

                    if ext == ".zip":
                        zf.extract(name, extract_dir)
                        nested_path = extract_dir / name
                        found.extend(
                            self._extract_mechanism_from_zip(nested_path, extract_dir)
                        )
                    elif ext in _CHEMKIN_EXT:
                        zf.extract(name, extract_dir)
                        found.append(extract_dir / name)
                        logger.info("   [SI] Found mechanism file: %s", name)
                    elif ext in _CANTERA_EXT:
                        zf.extract(name, extract_dir)
                        found.append(extract_dir / name)
                        logger.info("   [SI] Found Cantera file: %s", name)
                    elif ext == ".txt":
                        preview = zf.read(name)[:500].decode("utf-8", errors="ignore")
                        if any(kw in preview.upper() for kw in _CHEMKIN_KEYWORDS):
                            zf.extract(name, extract_dir)
                            found.append(extract_dir / name)
                            logger.info("   [SI] Found Chemkin-format TXT: %s", name)
        except zipfile.BadZipFile:
            logger.warning("   [SI] Bad ZIP file: %s", zip_path.name)

        return found

    @staticmethod
    def _select_best_mechanism(record: PaperRecord) -> None:
        """Set best_mechanism_path and mechanism_format based on priority.

        Priority: .yaml/.yml (Cantera) > .inp/.dat/.ck (Chemkin).
        Skip .cti (deprecated).
        """
        cantera_exts = {".yaml", ".yml"}
        chemkin_exts = {".inp", ".dat", ".ck"}

        for path_str in record.mechanism_files:
            ext = Path(path_str).suffix.lower()
            if ext in cantera_exts:
                record.best_mechanism_path = path_str
                record.mechanism_format = "cantera"
                return

        for path_str in record.mechanism_files:
            ext = Path(path_str).suffix.lower()
            if ext in chemkin_exts or ext == ".txt":
                record.best_mechanism_path = path_str
                record.mechanism_format = "chemkin"
                return

    def _download_paper(self, record: PaperRecord, papers_dir: Path) -> None:
        if not record.oa_url:
            return
        safe_id = record.id.replace("/", "_").replace(":", "_")
        fname = f"{safe_id}_paper.file"
        logger.info("   [PDF] Attempting to download paper for %s...", record.title[:40])
        filepath = download_file(record.oa_url, papers_dir, fname)
        note = f" Saved Paper to {filepath.name}. " if filepath else " Failed paper download. "
        record.access_notes = (record.access_notes or "") + note
