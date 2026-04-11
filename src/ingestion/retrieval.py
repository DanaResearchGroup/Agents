import logging
import os
import requests
import time
import re
from typing import Any, List, Optional
from duckduckgo_search import DDGS
from src.schemas.ingestion import PaperRecord, SearchQuery

logger = logging.getLogger(__name__)

_DEFAULT_EMAIL = os.environ.get("CONTACT_EMAIL", "agent@example.com")

# --- Client 1: Crossref ---
S2_SEARCH_ENDPOINT = "https://api.semanticscholar.org/graph/v1/paper/search"

class CrossrefClient:
    def __init__(self, mailto: str = _DEFAULT_EMAIL):
        self.base_url = "https://api.crossref.org/works"
        self.mailto = mailto

    def search(self, query: SearchQuery, limit: int = None) -> list[dict]:
        limit = limit or query.max_results
        params = {
            "query": query.topic,
            "rows": min(limit, 50),
            "mailto": self.mailto
        }
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            return response.json().get("message", {}).get("items", []) or []
        except Exception as e:
            logger.error(f"Crossref Search Error: {e}")
            return []

    def search_by_doi(self, doi: str) -> PaperRecord | None:
        """Direct DOI lookup via Crossref /works/{doi} endpoint."""
        url = f"{self.base_url}/{doi}"
        try:
            resp = requests.get(
                url, params={"mailto": self.mailto}, timeout=10,
            )
            if resp.status_code != 200:
                return None
            item = resp.json().get("message")
            if not item:
                return None
            return self.normalize(item)
        except Exception as e:
            logger.error("Crossref DOI lookup error: %s", e)
            return None

    def normalize(self, item: dict[str, Any]) -> PaperRecord:
        doi = item.get("DOI")
        title = item.get("title", ["Untitled"])[0]
        authors = [f"{a.get('given', '')} {a.get('family', '')}".strip() for a in item.get("author", [])]
        year = item.get("published-print", item.get("issued", {})).get("date-parts", [[None]])[0][0]

        return PaperRecord(
            id=f"cr_{doi}" if doi else f"cr_{hash(title)}",
            title=title,
            authors=authors,
            year=year,
            journal=item.get("container-title", [None])[0],
            venue=item.get("container-title", [None])[0],
            doi=doi,
            abstract=item.get("abstract"),
            source_url=f"https://doi.org/{doi}" if doi else item.get("URL"),
            provenance="crossref"
        )

# --- Client 2: OpenAlex ---
class OpenAlexClient:
    def __init__(self, mailto: str = _DEFAULT_EMAIL):
        self.base_url = "https://api.openalex.org/works"
        self.mailto = mailto

    def get_by_doi(self, doi: str) -> PaperRecord | None:
        """Direct DOI lookup via OpenAlex filter API."""
        params = {
            "filter": f"doi:{doi}",
            "per_page": 1,
            "mailto": self.mailto,
        }
        try:
            resp = requests.get(
                self.base_url, params=params,
                headers={"Accept": "application/json"}, timeout=10,
            )
            if resp.status_code != 200:
                return None
            results = resp.json().get("results", [])
            if not results:
                return None
            return self.normalize(results[0])
        except Exception as e:
            logger.error("OpenAlex DOI lookup error: %s", e)
            return None

    def search(self, query: SearchQuery, limit: int = None) -> list[dict]:
        limit = limit or query.max_results
        # Boolean Mode: Wrap terms in quotes and join with AND for exact matching
        terms = [f'"{t.strip()}"' for t in query.topic.split() if t.strip()]
        boolean_query = " AND ".join(terms) if terms else query.topic

        params = {
            "search": boolean_query,
            "per_page": min(limit, 50),
            "mailto": self.mailto
        }
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            return response.json().get("results", []) or []
        except Exception as e:
            logger.error(f"OpenAlex Search Error: {e}")
            return []

    def get_works(self, work_ids: List[str]) -> List[dict[str, Any]]:
        if not work_ids: return []
        # Filter out full URIs to get short IDs
        short_ids = [wid.split("/")[-1] for wid in work_ids if wid]
        if not short_ids: return []

        # OpenAlex allows '|' for OR in filters
        filter_str = f"openalex:{'|'.join(short_ids[:50])}" # Limit to 50 for safety
        params = {"filter": filter_str, "mailto": self.mailto}
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            return response.json().get("results", []) or []
        except Exception as e:
            logger.error(f"OpenAlex get_works error: {e}")
            return []

    def normalize(self, item: dict[str, Any]) -> PaperRecord:
        doi = (item.get("doi") or "").replace("https://doi.org/", "")
        authors = [a.get("author", {}).get("display_name") for a in item.get("authorships", [])]

        # OA Status
        oa_info = item.get("open_access", {})
        oa_status = "oa" if oa_info.get("is_oa") else "non_oa"
        oa_url = oa_info.get("oa_url")

        return PaperRecord(
            id=item.get("id"),
            title=item.get("display_name", "Untitled"),
            authors=authors,
            year=item.get("publication_year"),
            journal=item.get("host_venue", {}).get("display_name"),
            venue=item.get("host_venue", {}).get("display_name"),
            doi=doi if doi else None,
            abstract=None, # OpenAlex abstract is inverted index, needs extra parsing
            source_url=item.get("doi") or item.get("id"),
            oa_status=oa_status,
            oa_url=oa_url,
            provenance="openalex",
            references=item.get("referenced_works", [])
        )

    def get_citations(self, openalex_id: str, limit: int = 10) -> list[dict]:
        """Find works that cite this OpenAlex ID."""
        params = {"filter": f"cites:{openalex_id}", "per_page": min(limit, 50)}
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            return response.json().get("results", []) or []
        except Exception as e:
            logger.error(f"OpenAlex Citation Error: {e}")
            return []

    def get_work_by_id(self, openalex_id: str) -> dict or None:
        """Fetch a single work by its OpenAlex ID."""
        try:
            response = requests.get(f"{self.base_url}/{openalex_id}", timeout=20)
            return response.json()
        except: return None

# --- Client 3: Europe PMC ---
class EuropePMCClient:
    def __init__(self, mailto: str = _DEFAULT_EMAIL):
        self.base_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
        self.mailto = mailto

    def search(self, query: SearchQuery, limit: int = None) -> list[dict]:
        limit = limit or query.max_results
        params = {
            "query": f'"{query.topic}"',
            "format": "json",
            "pageSize": min(limit, 50),
            "resultType": "core"
        }
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            return response.json().get("resultList", {}).get("result", []) or []
        except Exception as e:
            logger.error(f"EuropePMC Search Error: {e}")
            return []

    def get_by_doi(self, doi: str) -> dict or None:
        params = {"query": f'DOI:"{doi}"', "format": "json", "resultType": "core"}
        try:
            r = requests.get(self.base_url, params=params, timeout=20)
            res = r.json().get("resultList", {}).get("result", [])
            return res[0] if res else None
        except: return None

    def get_by_title(self, title: str) -> dict or None:
        params = {"query": f'TITLE:"{title}"', "format": "json", "resultType": "core"}
        try:
            r = requests.get(self.base_url, params=params, timeout=20)
            res = r.json().get("resultList", {}).get("result", [])
            return res[0] if res else None
        except: return None

    def normalize(self, item: dict[str, Any]) -> PaperRecord:
        doi = item.get("doi")
        pmcid = item.get("pmcid")
        authors = [a.get("fullName") for a in item.get("authorList", {}).get("author", []) if a.get("fullName")]

        # SI link discovery in EuropePMC
        si_links = []
        if pmcid:
            # We'll fetch actual links later using get_supplementary_links
            # This is a flag for RegistryBuilder to know more work is needed
            si_links.append(f"FETCH_REAL: {pmcid}")
        elif item.get("hasSupplementary") == "Y":
             si_links.append(f"EuropePMC mirror for {doi or item.get('id')}")

        return PaperRecord(
            id=item.get("id"),
            title=item.get("title", "Untitled"),
            authors=authors,
            year=item.get("pubYear"),
            journal=item.get("journalTitle"),
            venue=item.get("journalTitle"),
            doi=doi,
            abstract=item.get("abstractText"),
            source_url=f"https://europepmc.org/article/MED/{item.get('id')}",
            si_link_found=len(si_links) > 0,
            si_links=si_links,
            provenance="europepmc"
        )

    def get_supplementary_links(self, pmcid: str) -> list[str]:
        """Fetch actual supplementary file URLs for a given PMCID."""
        url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/supplementaryFiles"
        try:
            r = requests.get(url, params={"format": "json"}, timeout=20, stream=True)
            r.raise_for_status()

            content_type = r.headers.get("Content-Type", "").lower()
            # If the response itself is a ZIP/binary file, return the URL directly
            if "application/zip" in content_type or "application/octet-stream" in content_type:
                logger.info(f"   [EPMC] Detected bulk SI ZIP for {pmcid}")
                return [url]

            # Otherwise, try parsing as JSON
            try:
                data = r.json()
            except Exception:
                return []

            files_list = data.get("supplementaryFileList", {})
            if not files_list: return []

            files = files_list.get("supplementaryFile", [])
            if isinstance(files, dict): files = [files]

            links = []
            for f in files:
                u = f.get("fullTextUrl")
                if u:
                    if u.startswith("/"):
                        u = "https://europepmc.org" + u
                    links.append(u)
            return links
        except Exception as e:
            logger.warning(f"Failed to fetch SI for {pmcid}: {e}")
            return []

# --- Client 4: Semantic Scholar ---
class SemanticScholarClient:
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.headers = {"x-api-key": api_key} if api_key else {}

    def search(self, query: SearchQuery, limit: int = None) -> list[dict]:
        limit = limit or query.max_results
        params = {
            "query": query.topic,
            "limit": min(limit, 100),
            "fields": "paperId,title,authors,year,venue,externalIds,abstract,url,openAccessPdf"
        }

        # S2 Public API limit is ~1 req/sec. Increase sleep to 2.0s to avoid 429 bursts.
        time.sleep(2.0)

        for attempt in range(1): # Reduced to 1 attempt for fail-fast behavior
            try:
                response = requests.get(S2_SEARCH_ENDPOINT, params=params, headers=self.headers, timeout=30)
                if response.status_code == 429:
                    logger.warning(f"S2 Rate limited. Skipping S2 for this strategy to save time.")
                    return []
                response.raise_for_status()
                return response.json().get("data", []) or []
            except Exception as e:
                logger.error(f"S2 Search Error: {e}")
                return []
        return []

    def normalize(self, item: dict[str, Any]) -> PaperRecord:
        doi = item.get("externalIds", {}).get("DOI")
        authors = [a.get("name") for a in item.get("authors", []) if a.get("name")]
        oa_url = item.get("openAccessPdf", {}).get("url") if item.get("openAccessPdf") else None
        oa_status = "oa" if oa_url else "non_oa"

        return PaperRecord(
            id=item.get("paperId") or f"s2_{hash(item.get('title', ''))}",
            title=item.get("title", "Untitled"),
            authors=authors,
            year=item.get("year"),
            journal=item.get("venue"),
            venue=item.get("venue"),
            doi=doi,
            abstract=item.get("abstract"),
            source_url=item.get("url"),
            oa_status=oa_status,
            oa_url=oa_url,
            provenance="semantic_scholar"
        )

# --- Client 5: Web Search (Deep SI Discovery) ---
class WebSearchClient:
    def __init__(self):
        pass

    def find_si_links(self, title: str = None, doi: str = None) -> list[str]:
        si_links = []
        queries = []
        if doi:
            queries.append(f'"{doi}" supplementary data mechanism')
            queries.append(f'"{doi}" kinetics file')
        if title:
            clean_title = title[:150]
            queries.append(f'"{clean_title}" supplemental data mechanism')
            queries.append(f'"{clean_title}" kinetics supplementary information')

        with DDGS() as ddgs:
            for query in queries:
                try:
                    logger.info(f"Deep Search Query: {query}")
                    results = ddgs.text(query, max_results=5)
                    for r in results:
                        link = r.get("href") or ""
                        if any(kw in link.lower() for kw in ["supp", "si", "data", "appendix", "table", "mech"]) or link.endswith(".pdf"):
                            if link not in si_links: si_links.append(link)
                except Exception as e:
                    logger.warning(f"Web search failed for query '{query}': {e}")
                    continue
        return si_links
