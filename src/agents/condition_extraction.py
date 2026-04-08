"""ConditionExtractionAgent: extract simulation conditions from paper text."""

import logging

from src.agents.llm_client import LLMClient
from src.schemas.experimental import (
    ExtractionResult,
    PaperDocument,
    SimConditions,
)

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

    async def extract(self, paper: PaperDocument) -> list[SimConditions]:
        """Extract simulation conditions from a parsed paper.

        Returns an empty list (never raises) if no conditions are found.
        """
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
