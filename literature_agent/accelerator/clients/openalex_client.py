import requests
from typing import Any
import logging
from ..models import PaperRecord, SearchQuery

logger = logging.getLogger(__name__)

OPENALEX_API_URL = "https://api.openalex.org/works"

class OpenAlexClient:
    def __init__(self, mailto: str = "your-email@example.com"):
        self.mailto = mailto

    def search(self, query: str | SearchQuery, limit: int = 10) -> list[dict[str, Any]]:
        """
        Search for papers on OpenAlex.
        """
        if isinstance(query, SearchQuery):
            q_str = query.topic
            limit = query.max_results
        else:
            q_str = query

        params = {
            "search": q_str,
            "per_page": limit,
            "mailto": self.mailto
        }

        try:
            response = requests.get(OPENALEX_API_URL, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            return data.get("results", [])
        except Exception as e:
            logger.error(f"Error searching OpenAlex: {e}")
            return []

    def normalize(self, item: dict[str, Any]) -> PaperRecord:
        """
        Normalize OpenAlex item to PaperRecord.
        """
        doi_raw = item.get("doi")
        doi = doi_raw.replace("https://doi.org/", "") if doi_raw else None
        title = item.get("display_name", "Untitled")
        
        authors = []
        for authorship in item.get("authorships", []):
            name = authorship.get("author", {}).get("display_name")
            if name:
                authors.append(name)

        year = item.get("publication_year")
        
        primary_location = item.get("primary_location") or {}
        source = primary_location.get("source") or {}
        journal = source.get("display_name")
        
        source_url = item.get("ids", {}).get("mag") or item.get("id")
        
        # OpenAlex provides abstract in 'abstract_inverted_index' which is hard to parse simply
        # For now, we'll leave it empty or just store the DOI link
        abstract = ""

        oa_info = item.get("open_access", {})
        oa_status = "oa" if oa_info.get("is_oa") else "non_oa"
        oa_url = oa_info.get("oa_url")

        return PaperRecord(
            id=f"openalex_{item.get('id', hash(title))}",
            title=title,
            authors=authors,
            year=year,
            journal=journal,
            doi=doi,
            abstract=abstract,
            source_url=source_url,
            oa_status=oa_status,
            oa_url=oa_url
        )
