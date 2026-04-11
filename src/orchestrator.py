"""Orchestrator: top-level coordination for Path 1 and Path 2 pipelines."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import yaml

from src.agents.llm_client import LLMClient
from src.agents.paper_reader import read_paper
from src.ingestion.pdf_parser import parse_pdf
from src.ingestion.retrieval import CrossrefClient, OpenAlexClient
from src.pipelines.path1 import run_path1
from src.pipelines.path2 import run_path2
from src.report import ReportGenerator
from src.schemas.experimental import (
    ExperimentalDataset,
    PaperDocument,
    PaperSource,
    PaperSummary,
    RunConfig,
)
from src.ingestion.utils import download_file
from src.schemas.ingestion import PaperRecord, SearchQuery

logger = logging.getLogger(__name__)


def _default_confirm(records: list[PaperRecord]) -> int:
    """Interactive confirmation: print results and prompt the user to choose.

    Returns the 0-based index of the selected record.
    Raises ValueError on invalid input.
    """
    print("\n── Search Results ──")
    for i, rec in enumerate(records, 1):
        print(f"  [{i}] {rec.title}")
        if rec.doi:
            print(f"      DOI: {rec.doi}")
        if rec.year:
            print(f"      Year: {rec.year}")

    choice = input(f"\nSelect paper [1-{len(records)}]: ").strip()
    try:
        idx = int(choice) - 1
        if not 0 <= idx < len(records):
            raise ValueError
    except ValueError:
        raise ValueError(f"Invalid selection: {choice}")
    return idx


class Orchestrator:
    """Routes work through ingestion, Path 1, Path 2, and report generation.

    Initialised with a RunConfig that declares *what* to run.
    Call ``await orchestrator.run()`` to execute the full pipeline.
    """

    def __init__(
        self,
        config: RunConfig,
        confirm_fn: Callable[[list[PaperRecord]], int] | None = None,
    ) -> None:
        # Validate that required files exist
        if not config.original_model.exists():
            raise ValueError(
                f"Original model not found: {config.original_model}"
            )
        if not config.experimental_data.exists():
            raise ValueError(
                f"Experimental data not found: {config.experimental_data}"
            )

        self.config = config
        self._confirm_fn = confirm_fn or _default_confirm

        # Load LLM client
        self.llm_client = LLMClient(config_path=config.llm_config)

        # Load and validate experimental dataset
        raw = yaml.safe_load(config.experimental_data.read_text())
        self.dataset = ExperimentalDataset.model_validate(raw)

        # Populated during ingestion / paper reading.
        self.paper_document: PaperDocument | None = None
        self.paper_summary: PaperSummary | None = None

    # ── Paper ingestion ─────────────────────────────────────────────────

    async def ingest_paper(self, source: PaperSource) -> PaperRecord:
        """Retrieve or parse a paper based on the source mode.

        - doi:    look up metadata via OpenAlex
        - upload: parse local PDF, return a minimal PaperRecord
        - search: search OpenAlex, show top 3, prompt user to confirm
        """
        if source.mode == "doi":
            return self._ingest_doi(source.value)
        elif source.mode == "upload":
            return self._ingest_upload(source.value)
        elif source.mode == "search":
            return await self._ingest_search(source.value)
        else:
            raise ValueError(f"Unknown paper source mode: {source.mode}")

    def _ingest_doi(self, doi: str) -> PaperRecord:
        """Fetch paper metadata by DOI via direct lookup (OpenAlex, then Crossref)."""
        # Try OpenAlex direct DOI lookup first
        client = OpenAlexClient()
        paper = client.get_by_doi(doi)

        if paper is None:
            # Fallback: Crossref direct DOI lookup
            crossref = CrossrefClient()
            paper = crossref.search_by_doi(doi)

        if paper is None:
            raise ValueError(f"No paper found for DOI: {doi}")

        if paper.oa_url:
            safe_doi = doi.replace("/", "_").replace(":", "_")
            papers_dir = Path("data/papers")
            downloaded = download_file(
                paper.oa_url, papers_dir, f"{safe_doi}.pdf",
                use_browser_auth=self.config.browser_auth,
            )
            if downloaded:
                paper.pdf_path = str(downloaded.resolve())
                logger.info("Downloaded PDF to %s", paper.pdf_path)
            else:
                logger.warning(
                    "PDF download failed for DOI %s from %s",
                    doi, paper.oa_url,
                )
        else:
            logger.warning(
                "No open access PDF available for DOI %s — "
                "paper text extraction will be limited",
                doi,
            )

        return paper

    def _ingest_upload(self, file_path: str) -> PaperRecord:
        """Parse a local PDF and wrap it in a minimal PaperRecord."""
        path = Path(file_path)
        if not path.exists():
            raise ValueError(f"PDF file not found: {file_path}")

        document = parse_pdf(str(path))
        self.paper_document = document
        return PaperRecord(
            id=path.stem,
            title=document.title or path.name,
            provenance="local_upload",
            pdf_path=str(path.resolve()),
        )

    async def _ingest_search(self, query_text: str) -> PaperRecord:
        """Search for papers and let confirm_fn choose one."""
        client = OpenAlexClient()
        query = SearchQuery(topic=query_text, max_results=3)
        results = client.search(query, limit=3)
        if not results:
            raise ValueError(f"No papers found for query: {query_text}")

        records = [client.normalize(r) for r in results]
        idx = self._confirm_fn(records)
        return records[idx]

    # ── Pipeline stubs ──────────────────────────────────────────────────

    async def _run_path1(self, paper: PaperRecord) -> object:
        """Path 1: Literature Model Evaluation."""
        return await run_path1(
            paper=paper,
            original_model=self.config.original_model,
            experimental_data=self.dataset,
            llm_client=self.llm_client,
            literature_model=self.config.literature_model,
            species_aliases=self.config.species_aliases or None,
            literature_aliases=self.config.literature_aliases or None,
        )

    async def _run_path2(self, paper: PaperRecord, path1_results=None) -> object:
        """Path 2: Targeted Model Improvements."""
        return await run_path2(
            paper=paper,
            original_model=self.config.original_model,
            experimental_data=self.dataset,
            llm_client=self.llm_client,
            output_dir=self.config.output_dir,
            path1_results=path1_results,
            species_aliases=self.config.species_aliases or None,
        )

    # ── Report generation ───────────────────────────────────────────────

    def _generate_report(
        self,
        path1_results: object | None,
        path2_results: object | None,
    ) -> Path:
        """Write a structured YAML report to output_dir."""
        generator = ReportGenerator()
        return generator.generate(
            original_model=self.config.original_model,
            experimental_data=self.dataset,
            output_dir=self.config.output_dir,
            path1_results=path1_results,
            path2_results=path2_results,
        )

    # ── Main entry point ────────────────────────────────────────────────

    async def _read_paper(self, paper: PaperRecord) -> None:
        """Run PaperReaderAgent if we have a parsed PDF document."""
        if self.paper_document is None and paper.pdf_path:
            pdf = Path(paper.pdf_path)
            if pdf.exists():
                self.paper_document = parse_pdf(str(pdf))

        if self.paper_document is not None:
            self.paper_summary = await read_paper(
                paper=self.paper_document,
                config=self.llm_client.config,
            )

    async def run(self) -> Path:
        """Execute the full orchestrator pipeline and return the report path."""
        paper = await self.ingest_paper(self.config.paper_source)
        await self._read_paper(paper)

        path1_results = None
        path2_results = None

        if self.config.run_path1:
            path1_results = await self._run_path1(paper)

        if self.config.run_path2:
            path2_results = await self._run_path2(paper, path1_results)

        return self._generate_report(path1_results, path2_results)
