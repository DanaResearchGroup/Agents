import argparse
import json
import logging
import sys
from pathlib import Path
from .models import SearchQuery
from .registry_builder import RegistryBuilder
from .io_utils import save_registry

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Paper Discovery and Triage Tool")
    parser.add_argument("--query", type=str, help="Free-text topic query or path to a JSON search object")
    parser.add_argument("--max-results", type=int, default=30, help="Maximum results to retrieve")
    parser.add_argument("--outdir", type=str, default="outputs/", help="Directory to save results")
    parser.add_argument("--mailto", type=str, default="your-email@example.com", help="Email for API polite usage")
    parser.add_argument("--download-si", action="store_true", help="Download discovered SI/mechanism files")

    args = parser.parse_args()

    if not args.query:
        logger.error("No query provided. Use --query 'your topic' or --query search_object.json")
        sys.exit(1)

    # 1. Parse Input
    search_query = None
    if args.query.endswith(".json") and Path(args.query).exists():
        try:
            with open(args.query, "r") as f:
                data = json.load(f)
                search_query = SearchQuery(**data)
        except Exception as e:
            logger.error(f"Failed to parse JSON query object: {e}")
            sys.exit(1)
    else:
        # Check if it's a JSON string
        try:
            data = json.loads(args.query)
            if isinstance(data, dict):
                search_query = SearchQuery(**data)
        except:
            # Treat as free-text
            search_query = SearchQuery(topic=args.query, max_results=args.max_results)

    if not search_query:
        logger.error("Could not construct a SearchQuery from input.")
        sys.exit(1)

    # 2. Run Pipeline
    builder = RegistryBuilder(mailto=args.mailto)
    logger.info(f"Starting discovery for: {search_query.topic}")
    
    result = builder.build_registry(search_query, download_si=args.download_si, outdir=args.outdir)
    
    # 3. Save Results
    save_registry(result, args.outdir)
    
    logger.info(f"Done. Found {len(result.papers)} papers.")
    logger.info(f"Registry saved to {args.outdir}")

if __name__ == "__main__":
    main()
