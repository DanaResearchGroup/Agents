import logging
from duckduckgo_search import DDGS
from typing import List

logger = logging.getLogger(__name__)

class WebSearchClient:
    def __init__(self):
        pass

    def find_si_links(self, title: str, doi: str = None) -> List[str]:
        """
        Search the web for supplementary information links.
        """
        si_links = []
        # Construct search queries
        queries = []
        if doi:
            queries.append(f'"{doi}" supplementary information PDF')
        if title:
            # Clean title for better search
            clean_title = title[:150]
            queries.append(f'"{clean_title}" supplemental data mechanism')
            queries.append(f'"{clean_title}" kinetics supplementary information')

        with DDGS() as ddgs:
            for query in queries:
                try:
                    logger.info(f"Deep Search Query: {query}")
                    results = ddgs.text(query, max_results=5)
                    for r in results:
                        link = r.get("href", "")
                        # Heuristic filtering for SI-like links
                        if any(kw in link.lower() for kw in ["supp", "si", "data", "appendix", "table", "mech"]) or link.endswith(".pdf"):
                            if link not in si_links:
                                si_links.append(link)
                except Exception as e:
                    logger.warning(f"Web search failed for query '{query}': {e}")
                    continue
        
        return si_links
