import logging
import re
from typing import Any
from .models import PaperRecord

logger = logging.getLogger(__name__)

# Keywords that suggest a link or metadata refers to SI or mechanism files
SI_KEYWORDS = [
    "supplementary", "supporting information", "appendices", "appendix", 
    "dataset", "repository", "si material", "electronic supplementary"
]

MECH_KEYWORDS = [
    "mechanism", "chemkin", "cantera", "kinetic model", "ckin", "input file",
    "yaml", "cti", "xml"
]

def discover_si_links(record: PaperRecord, raw_crossref: dict[str, Any] = None, raw_openalex: dict[str, Any] = None, raw_epmc: dict[str, Any] = None):
    """
    Search through normalized record and raw metadata for SI/mechanism links.
    """
    links = set(record.si_links)
    mech_links = set(record.mechanism_links)

    # 1. Europe PMC Signals (very reliable for hasSuppl)
    if raw_epmc:
        if raw_epmc.get("hasSuppl") == "Y":
            record.si_link_found = True
            pmcid = raw_epmc.get("pmcid")
            pmid = raw_epmc.get("pmid")
            if pmcid:
                links.add(f"https://europepmc.org/articles/{pmcid}#sec5")
            elif pmid:
                links.add(f"https://europepmc.org/article/MED/{pmid}#sec5")

    # 2. DOI-based SI Detection (e.g., ACS uses .s001, .s002 for supplements)
    if record.doi:
        doi_lower = record.doi.lower()
        if re.search(r'\.s\d{3}$', doi_lower):
            record.si_link_found = True
            # The source URL itself is the SI link in this case
            if record.source_url:
                links.add(record.source_url)
            elif record.doi.startswith("10."):
                links.add(f"https://doi.org/{record.doi}")

    # 3. Search Crossref metadata
    if raw_crossref:
        # Check 'relation'
        relations = raw_crossref.get("relation", {})
        for rel_type, rel_list in relations.items():
            for item in rel_list:
                url = item.get("id")
                if url:
                    if any(kw in url.lower() for kw in SI_KEYWORDS):
                        links.add(url)
                    if any(kw in url.lower() for kw in MECH_KEYWORDS):
                        mech_links.add(url)
        
        # Check 'link' array (often contains direct resource URLs)
        for link_item in raw_crossref.get("link", []):
            url = link_item.get("URL")
            if url and any(kw in url.lower() for kw in SI_KEYWORDS + MECH_KEYWORDS):
                links.add(url)

        # Check 'assertion' for SI mentions
        for assertion in raw_crossref.get("assertion", []):
            label = assertion.get("label", "").lower()
            if any(kw in label for kw in SI_KEYWORDS):
                record.si_link_found = True

    # 4. Search OpenAlex metadata
    if raw_openalex:
        # Check all locations
        for loc in raw_openalex.get("locations", []):
            url = loc.get("landing_page_url") or loc.get("pdf_url")
            if url:
                low_url = url.lower()
                if any(kw in low_url for kw in SI_KEYWORDS):
                    links.add(url)
                if any(kw in low_url for kw in MECH_KEYWORDS):
                    mech_links.add(url)
        
        # Check OpenAlex 'supplementary_combined' field if available (future proofing)
        apc_list = raw_openalex.get("apc_list") or {}
        for url in apc_list.get("url", []):
             if any(kw in url.lower() for kw in SI_KEYWORDS):
                 links.add(url)

    # 5. Final Heuristics on text
    text = f"{record.title or ''} {record.abstract or ''}".lower()
    if any(kw in text for kw in SI_KEYWORDS):
        record.si_link_found = True
        
    if any(kw in text for kw in MECH_KEYWORDS):
        record.mechanism_link_found = True

    record.si_links = list(links)
    record.mechanism_links = list(mech_links)
    
    if record.si_links:
        record.si_link_found = True
    if record.mechanism_links:
        record.mechanism_link_found = True
