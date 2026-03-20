import requests
from typing import Any
import logging
from ..models import PaperRecord, SearchQuery

logger = logging.getLogger(__name__)

CROSSREF_API_URL = "https://api.crossref.org/works"

class CrossrefClient:
    def __init__(self, mailto: str = "your-email@example.com"):
        self.mailto = mailto

    def search(self, query: str | SearchQuery, limit: int = 10) -> list[dict[str, Any]]:
        """
        Search for papers on Crossref.
        """
        if isinstance(query, SearchQuery):
            q_str = query.topic
            limit = query.max_results
        else:
            q_str = query

        params = {
            "query": q_str,
            "rows": limit,
            "mailto": self.mailto
        }

        try:
            response = requests.get(CROSSREF_API_URL, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("items", [])
        except Exception as e:
            logger.error(f"Error searching Crossref: {e}")
            return []

    def normalize(self, item: dict[str, Any]) -> PaperRecord:
        """
        Normalize Crossref item to PaperRecord.
        """
        doi = item.get("DOI")
        title = item.get("title", ["Untitled"])[0]
        
        authors = []
        for author in item.get("author", []):
            name = author.get("family", "")
            if author.get("given"):
                name = f"{author.get('given')} {name}"
            if name:
                authors.append(name.strip())

        year = None
        issued = item.get("issued", {}).get("date-parts", [])
        if issued and issued[0]:
            year = issued[0][0]

        journal = item.get("container-title", [None])[0]
        source_url = item.get("URL")
        
        # Crossref usually doesn't provide abstracts in the main search
        abstract = item.get("abstract")

        return PaperRecord(
            id=f"crossref_{doi}" if doi else f"crossref_{hash(title)}",
            title=title,
            authors=authors,
            year=year,
            journal=journal,
            doi=doi,
            abstract=abstract,
            source_url=source_url,
            oa_status="unknown", 
        )
