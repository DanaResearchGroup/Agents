import logging
from typing import Any
from .models import PaperRecord, SearchQuery, RegistryResult
from .clients.crossref_client import CrossrefClient
from .clients.openalex_client import OpenAlexClient
from .clients.europepmc_client import EuropePMCClient
from .scoring import calculate_relevance
from .tagging import assign_tags_and_priority
from .si_discovery import discover_si_links
from .io_utils import download_file
from pathlib import Path

logger = logging.getLogger(__name__)

class RegistryBuilder:
    def __init__(self, mailto: str = "your-email@example.com"):
        self.crossref = CrossrefClient(mailto=mailto)
        self.openalex = OpenAlexClient(mailto=mailto)
        self.epmc = EuropePMCClient(mailto=mailto)

    def build_registry(self, query: SearchQuery, download_si: bool = False, outdir: str = "outputs") -> RegistryResult:
        """
        Orchestrate the search, deduplication, enrichment, and scoring.
        """
        registry = {}  # doi -> PaperRecord
        raw_metadata = {} # doi -> {cr: ..., oa: ..., epmc: ...}

        # 1. Search Crossref
        logger.info(f"Searching Crossref for: {query.topic}")
        cr_items = self.crossref.search(query)
        for item in cr_items:
            record = self.crossref.normalize(item)
            doi = record.doi.lower() if record.doi else f"cr_{hash(record.title)}"
            registry[doi] = record
            raw_metadata[doi] = {"cr": item}

        # 2. Search OpenAlex and Deduplicate
        logger.info(f"Searching OpenAlex for: {query.topic}")
        oa_items = self.openalex.search(query)
        for item in oa_items:
            record = self.openalex.normalize(item)
            doi_key = record.doi.lower() if record.doi else f"oa_{hash(record.title)}"
            
            if doi_key in registry:
                # Enrich existing record
                existing = registry[doi_key]
                if not existing.abstract and record.abstract:
                    existing.abstract = record.abstract
                if not existing.oa_url and record.oa_url:
                    existing.oa_url = record.oa_url
                raw_metadata[doi_key]["oa"] = item
            else:
                registry[doi_key] = record
                raw_metadata[doi_key] = {"oa": item}

        # 3. SI Discovery and Score/Tag
        results = []
        si_dir = Path(outdir) / "si_files"
        
        for doi, record in registry.items():
            # Enrich with Europe PMC (especially for bridging preprints)
            epmc_item = None
            if record.doi:
                epmc_item = self.epmc.get_by_doi(record.doi)
            if not epmc_item and record.title:
                epmc_item = self.epmc.get_by_title(record.title)
            
            if epmc_item:
                epmc_rec = self.epmc.normalize(epmc_item)
                # Merge basic info if missing
                if not record.journal and epmc_rec.journal: record.journal = epmc_rec.journal
                if not record.abstract and epmc_rec.abstract: record.abstract = epmc_rec.abstract
                if epmc_rec.si_link_found:
                    record.si_link_found = True
                    for link in epmc_rec.si_links:
                        if link not in record.si_links: record.si_links.append(link)
                raw_metadata[doi]["epmc"] = epmc_item

            # 4. Deep SI Discovery (new)
            meta = raw_metadata.get(doi, {})
            discover_si_links(record, raw_crossref=meta.get("cr"), raw_openalex=meta.get("oa"), raw_epmc=meta.get("epmc"))
            
            # 5. Optional downloading
            if download_si and record.si_link_found:
                # Sanitize record ID for filesystem use
                safe_id = record.id.replace("/", "_").replace("\\", "_").replace(":", "_")
                for i, url in enumerate(record.si_links):
                    # Simple filename from URL or DOI
                    ext = "file"
                    if "." in url:
                        potential_ext = url.split(".")[-1].split("?")[0].split("#")[0]
                        if 2 <= len(potential_ext) <= 4:
                            ext = potential_ext
                    
                    fname = f"{safe_id}_si_{i}.{ext}"
                    # Filter out helper labels from URL before download
                    download_url = url.split(": ", 1)[-1] if ": " in url else url
                    
                    filepath = download_file(download_url, si_dir, fname)
                    if filepath:
                        record.access_notes = (record.access_notes or "") + f" Downloaded SI to {fname} from {download_url}. "
                    else:
                        if "europepmc" in download_url.lower() and "#sec5" in download_url.lower():
                            record.access_notes = (record.access_notes or "") + f" SI link {i} is a web supplemental section; see URL for manual review. "
                        else:
                            record.access_notes = (record.access_notes or "") + f" Failed to download SI link {i} ({download_url}). "

            # Score
            score, reasons = calculate_relevance(record, query.keywords)
            record.relevance_score = score
            record.relevance_reason = reasons
            
            # Tag and Priority
            assign_tags_and_priority(record)
            
            results.append(record)

        # Sort by relevance
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return RegistryResult(papers=results)
