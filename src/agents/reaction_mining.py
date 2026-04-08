"""Reaction-mining agent: classifies figures and extracts kinetic data.

Replaces flux_diagram_agent/agent/brain.py. All LLM calls go through
LLMClient — no direct SDK imports.
"""

from __future__ import annotations

import logging

from src.agents.llm_client import LLMClient
from src.schemas.experimental import (
    FigureClassification,
    FluxInterpretation,
    SensitivityInterpretation,
)

logger = logging.getLogger(__name__)

AGENT_NAME = "reaction_mining"

# ── Prompts (carried over from flux_diagram_agent/agent/prompts.py) ──────

_CLASSIFICATION_SYSTEM = """
You are a vision-language assistant specializing in chemical kinetics and computational chemistry paper analysis.
Your task is to classify a scientific figure based on its image, caption, and context.

Classify the figure into ONE of the following categories:
- flux diagram (diagram showing reaction pathways with arrows, often with thickness representing flux)
- PES (potential energy surface / reaction profile showing relative energy levels)
- energy profile (similar to PES)
- molecular structure (3D or 2D geometry of a molecule or transition state)
- sensitivity analysis (bar charts or lists showing sensitivity coefficients)
- species profile (concentration/mole fraction vs time/distance/temperature plots)
- mechanism schematic (simplified representation of a chemical mechanism)
- other

Provide your response in JSON format:
{
  "label": "category_name",
  "confidence": 0.0 to 1.0,
  "reasoning": "brief explanation"
}
""".strip()

_CLASSIFICATION_USER = """
Figure ID: {figure_id}
Caption: {caption}
Context: {context_text}

Please categorize this figure.
""".strip()

_FLUX_SYSTEM = """
You are a chemical kinetics expert. You are given a figure that has been identified as a flux diagram (or reaction path diagram).
Your goal is to extract structured information about the pathways shown in the figure.

Extract:
1. Reaction system (e.g., Ethanol pyrolysis, Propane combustion, etc.)
2. Conditions (Temperature, Pressure, Equivalence ratio, Residence time, etc.)
3. Major species/nodes mentioned.
4. Dominant pathways (ORDERED BY SIGNIFICANCE: From species -> To species, importance).
5. Whether quantitative information (numbers or percentages on arrows) is present.
6. Usefulness assessment (how useful this diagram is for understanding the mechanism).
7. Uncertainty/Confidence.

Provide your response in JSON format following this schema:
{
  "system": "str",
  "conditions": {
    "temperature": "str or unknown",
    "pressure": "str or unknown",
    "equivalence_ratio": "str or unknown",
    "residence_time": "str or unknown"
  },
  "major_species": ["str"],
  "dominant_pathways": [
    {
      "rank": 1,
      "from": "str",
      "to": "str",
      "importance": "high/medium/low",
      "evidence": "str"
    }
  ],
  "quantitative_info": true/false,
  "usefulness": "high/medium/low",
  "use_cases": ["str"],
  "uncertainty": "str",
  "confidence": 0.0 to 1.0
}
""".strip()

_FLUX_USER = """
Figure ID: {figure_id}
Caption: {caption}
Context Text: {context_text}

Analyze the provided image of this flux diagram and extract the requested information.
""".strip()

_SENSITIVITY_SYSTEM = """
You are a chemical kinetics expert. You are given a figure that has been identified as a sensitivity analysis diagram (typically a bar chart).
Your goal is to extract structured information about which reactions have the most impact on a target property (e.g., ignition delay, flame speed).

Extract:
1. Target property (e.g., OH sensitivity, Ignition delay sensitivity).
2. Conditions (Temperature, Pressure, Equivalence ratio).
3. Top sensitive reactions (ORDERED BY ABSOLUTE SIGNIFICANCE: Reaction, sensitivity coefficient/value, impact direction).
4. Usefulness assessment.
5. Uncertainty/Confidence.

Provide your response in JSON format following this schema:
{
  "target_property": "str",
  "conditions": {
    "temperature": "str or unknown",
    "pressure": "str or unknown",
    "equivalence_ratio": "str or unknown"
  },
  "top_reactions": [
    {
      "rank": 1,
      "reaction": "str",
      "coefficient": "float or str",
      "impact": "positive/negative (e.g. promotes/inhibits reaction)"
    }
  ],
  "usefulness": "high/medium/low",
  "uncertainty": "str",
  "confidence": 0.0 to 1.0
}
""".strip()

_SENSITIVITY_USER = """
Figure ID: {figure_id}
Caption: {caption}
Context Text: {context_text}

Analyze the provided image of this sensitivity analysis diagram and extract the requested information.
""".strip()


# ── Agent ─────────────────────────────────────────────────────────────────


class ReactionMiningAgent:
    """Classifies scientific figures and extracts kinetic data via LLM vision.

    Replaces the provider-specific FigureClassifier, FluxInterpreter, and
    SensitivityInterpreter classes from flux_diagram_agent.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    async def classify_figure(
        self,
        image_b64: str | None,
        caption: str,
        *,
        figure_id: str = "Unknown",
        context_text: str = "",
    ) -> FigureClassification:
        """Classify a scientific figure into a category.

        Args:
            image_b64: Base64-encoded image, or None for text-only.
            caption: Figure caption text.
            figure_id: Identifier for logging.
            context_text: Surrounding text from the paper.
        """
        prompt = _CLASSIFICATION_USER.format(
            figure_id=figure_id, caption=caption, context_text=context_text,
        )
        images = [image_b64] if image_b64 else None
        try:
            result = await self.llm_client.complete(
                prompt=prompt,
                system=_CLASSIFICATION_SYSTEM,
                response_model=FigureClassification,
                agent_name=AGENT_NAME,
                images=images,
            )
            return result  # type: ignore[return-value]
        except Exception as e:
            logger.error("Error classifying figure %s: %s", figure_id, e)
            return FigureClassification(
                label="other", confidence=0.0,
                reasoning=f"Error during classification: {e}",
            )

    async def interpret_flux(
        self,
        image_b64: str | None,
        caption: str,
        *,
        figure_id: str = "Unknown",
        context_text: str = "",
    ) -> FluxInterpretation:
        """Extract structured pathway data from a flux diagram.

        Args:
            image_b64: Base64-encoded image, or None for text-only.
            caption: Figure caption text.
            figure_id: Identifier for logging.
            context_text: Surrounding text from the paper.
        """
        prompt = _FLUX_USER.format(
            figure_id=figure_id, caption=caption, context_text=context_text,
        )
        images = [image_b64] if image_b64 else None
        try:
            result = await self.llm_client.complete(
                prompt=prompt,
                system=_FLUX_SYSTEM,
                response_model=FluxInterpretation,
                agent_name=AGENT_NAME,
                images=images,
            )
            return result  # type: ignore[return-value]
        except Exception as e:
            logger.error("Error interpreting flux diagram %s: %s", figure_id, e)
            return FluxInterpretation(uncertainty=str(e))

    async def interpret_sensitivity(
        self,
        image_b64: str | None,
        caption: str,
        *,
        figure_id: str = "Unknown",
        context_text: str = "",
    ) -> SensitivityInterpretation:
        """Extract ranked reaction sensitivities from a sensitivity analysis figure.

        Args:
            image_b64: Base64-encoded image, or None for text-only.
            caption: Figure caption text.
            figure_id: Identifier for logging.
            context_text: Surrounding text from the paper.
        """
        prompt = _SENSITIVITY_USER.format(
            figure_id=figure_id, caption=caption, context_text=context_text,
        )
        images = [image_b64] if image_b64 else None
        try:
            result = await self.llm_client.complete(
                prompt=prompt,
                system=_SENSITIVITY_SYSTEM,
                response_model=SensitivityInterpretation,
                agent_name=AGENT_NAME,
                images=images,
            )
            return result  # type: ignore[return-value]
        except Exception as e:
            logger.error(
                "Error interpreting sensitivity analysis %s: %s", figure_id, e,
            )
            return SensitivityInterpretation(uncertainty=str(e))
