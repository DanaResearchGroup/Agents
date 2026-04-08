"""ConditionExtractionAgent: extract simulation conditions from paper text."""

from __future__ import annotations

import logging

from src.agents.llm_client import LLMClient
from src.schemas.experimental import (
    EvidenceSnippet,
    ExtractionResult,
    PaperDocument,
    SimConditions,
)

# Kinds that carry simulation-relevant numerical data.
_HIGH_SIGNAL_KINDS = frozenset({
    "temperature",
    "pressure",
    "composition",
    "experiment_family",
    "observable_type",
    "residence_time",
    "equivalence_ratio",
})

_MAX_SNIPPETS = 20
_TRUNCATED_SNIPPETS = 12
_WORD_BUDGET = 600

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a combustion chemistry expert extracting simulation conditions "
    "from a scientific paper. Return only conditions that are explicitly stated "
    "with numerical values. Do not infer or assume values that are not present "
    "in the text. Return valid JSON matching the ExtractionResult schema."
)


class ConditionExtractionAgent:
    """Extracts simulation-ready conditions from paper text via LLM."""

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    async def extract(
        self,
        paper: PaperDocument,
        evidence: list[EvidenceSnippet] | None = None,
    ) -> list[SimConditions]:
        """Extract simulation conditions from a parsed paper.

        When *evidence* is supplied (from the deterministic pipeline),
        it is formatted into a structured context block that anchors the
        LLM on pre-extracted values.  Falls back to raw paper text when
        evidence is ``None``.

        Returns an empty list (never raises) if no conditions are found.
        """
        if evidence:
            prompt = self._build_evidence_prompt(evidence, paper)
        else:
            text = paper.full_text()
            prompt = (
                "Extract all explicit experimental/simulation conditions from "
                "the following paper text. For each condition, identify the "
                "reactor type, temperature (K), pressure (atm), composition "
                "(mole fractions), observable type, and observable label.\n\n"
                f"Paper text:\n{text[:12000]}"
            )

        result: ExtractionResult = await self.llm_client.complete(
            prompt=prompt,
            system=SYSTEM_PROMPT,
            response_model=ExtractionResult,
            agent_name="condition_extraction",
        )

        if not result.conditions:
            logger.warning("No conditions extracted from paper: %s", paper.title)
            return []

        logger.info(
            "Extracted %d conditions (confidence=%.2f)",
            len(result.conditions),
            result.confidence,
        )
        return result.conditions

    # ── Evidence formatting helpers ──────────────────────────────────────

    def _build_evidence_prompt(
        self,
        evidence: list[EvidenceSnippet],
        paper: PaperDocument,
    ) -> str:
        """Build a prompt grounded on pre-extracted evidence snippets."""
        snippets = self._select_snippets(evidence, _MAX_SNIPPETS)
        context = self._format_evidence_block(snippets)

        # Enforce word budget — rebuild with fewer snippets if needed.
        if len(context.split()) > _WORD_BUDGET:
            snippets = self._select_snippets(evidence, _TRUNCATED_SNIPPETS)
            context = self._format_evidence_block(snippets)

        abstract = paper.abstract if hasattr(paper, "abstract") and paper.abstract else ""
        return (
            f"Evidence extracted from paper:\n{context}\n\n"
            f"Full abstract for context:\n{abstract}\n\n"
            "Using the evidence above, extract simulation conditions. "
            "Only extract conditions supported by the evidence. "
            "Do not infer values not present in the evidence."
        )

    @staticmethod
    def _select_snippets(
        evidence: list[EvidenceSnippet],
        limit: int,
    ) -> list[EvidenceSnippet]:
        """Filter to high-signal kinds, sort by confidence, take top *limit*."""
        filtered = [s for s in evidence if s.kind in _HIGH_SIGNAL_KINDS]
        filtered.sort(key=lambda s: s.confidence, reverse=True)
        return filtered[:limit]

    @staticmethod
    def _format_evidence_block(snippets: list[EvidenceSnippet]) -> str:
        """Render snippets into a structured text block for the prompt."""
        lines: list[str] = []
        for s in snippets:
            display = s.normalized_value if s.normalized_value is not None else s.value_text
            lines.append(
                f"[{s.kind.upper()}] {display} (confidence: {s.confidence:.2f})\n"
                f"  source: \"{s.source_text[:120]}\""
            )
        return "\n\n".join(lines)
