import requests
import logging
from typing import Any
from ..models import PaperRecord, SearchQuery

logger = logging.getLogger(__name__)

EUROPE_PMC_API_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
EUROPE_PMC_BASE_URL = "https://europepmc.org/api"

class EuropePMCClient:
    def __init__(self, mailto: str = "your-email@example.com"):
        self.mailto = mailto

    def search(self, query: str | SearchQuery, limit: int = 10) -> list[dict[str, Any]]:
        """
        Search for papers on Europe PMC.
        """
        if isinstance(query, SearchQuery):
            q_str = query.topic
            limit = query.max_results
        else:
            q_str = query

        params = {
            "query": q_str,
            "resultType": "core",
            "pageSize": limit,
            "format": "json"
        }

        try:
            response = requests.get(EUROPE_PMC_API_URL, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            return data.get("resultList", {}).get("result", [])
        except Exception as e:
            logger.error(f"Error searching Europe PMC: {e}")
            return []

    def get_by_doi(self, doi: str) -> dict[str, Any] or None:
        """
        Find a specific record by DOI.
        """
        results = self.search(f'DOI:"{doi}"', limit=1)
        return results[0] if results else None

    def get_by_title(self, title: str) -> dict[str, Any] or None:
        """
        Find a specific record by Title.
        """
        results = self.search(f'TITLE:"{title}"', limit=1)
        return results[0] if results else None

    def normalize(self, item: dict[str, Any]) -> PaperRecord:
        """
        Normalize Europe PMC item to PaperRecord.
        """
        doi = item.get("doi")
        pmid = item.get("pmid")
        pmcid = item.get("pmcid")
        title = item.get("title", "Untitled")
        
        authors = []
        author_list = item.get("authorList", {}).get("author", [])
        for author in author_list:
            name = author.get("fullName") or f"{author.get('firstName', '')} {author.get('lastName', '')}".strip()
            if name:
                authors.append(name)

        year = item.get("pubYear")
        journal = item.get("journalInfo", {}).get("journal", {}).get("title")
        
        source_url = f"https://europepmc.org/article/MED/{pmid}" if pmid else f"https://doi.org/{doi}" if doi else None
        
        abstract = item.get("abstractText", "")
        
        oa_status = "oa" if item.get("isOpenAccess") == "Y" else "non_oa"
        
        record = PaperRecord(
            id=f"epmc_{pmcid or pmid or doi or hash(title)}",
            title=title,
            authors=authors,
            year=int(year) if year and year.isdigit() else None,
            journal=journal,
            doi=doi,
            abstract=abstract,
            source_url=source_url,
            oa_status=oa_status
        )
        
        # Flags
        if item.get("hasSuppl") == "Y":
             record.si_link_found = True
             record.access_notes = (record.access_notes or "") + " Europe PMC flags supplementary material available. "
             if pmcid:
                 # Landing page link - clearly labeled
                 record.si_links.append(f"Europe PMC Supplemental Section: https://europepmc.org/articles/{pmcid}#sec5")
                 
                 # NEW: Fetch direct metadata to get "true" download links
                 direct_links = self.get_si_metadata(pmcid)
                 record.si_links.extend(direct_links)
        
        return record

    def get_si_metadata(self, pmcid: str) -> list[str]:
        """
        Fetch direct SI download links from Europe PMC's internal metadata API.
        """
        url = f"{EUROPE_PMC_BASE_URL}/fulltextRepo?pmcId={pmcid}&type=METADATA"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            links = []
            for file_item in data.get("files", []):
                if file_item.get("type") == "supplement":
                    filename = file_item.get("filename")
                    mime_type = file_item.get("mimeType")
                    if filename:
                        # Construct a URL that we've verified works with requests (doesn't 403)
                        working_url = (
                            f"{EUROPE_PMC_BASE_URL}/fulltextRepo?"
                            f"pmcId={pmcid}&type=FILE&fileName={filename}"
                        )
                        if mime_type:
                            working_url += f"&mimeType={mime_type}"
                        links.append(working_url)
            return links
        except Exception as e:
            logger.warning(f"Failed to fetch SI metadata for {pmcid}: {e}")
            return []
