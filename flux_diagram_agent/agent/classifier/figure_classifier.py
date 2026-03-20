import json
import logging
from typing import Dict, Any, Optional
from agent.utils.llm_provider import LLMProvider
from agent.prompts.templates import CLASSIFICATION_SYSTEM_PROMPT, CLASSIFICATION_USER_PROMPT

logger = logging.getLogger(__name__)

class FigureClassifier:
    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def classify(self, figure: Dict[str, Any]) -> Dict[str, Any]:
        """Classifies a figure using DeepSeek Vision (or text-only fallback)."""
        prompt = CLASSIFICATION_USER_PROMPT.format(
            figure_id=figure.get("figure_id", "Unknown"),
            caption=figure.get("caption", ""),
            context_text=figure.get("context_text", "")
        )

        try:
            image_path = figure.get("page_image_path")
            if image_path:
                response_text = self.llm.vision_complete(
                    prompt=prompt,
                    image_path=image_path,
                    system_prompt=CLASSIFICATION_SYSTEM_PROMPT
                )
            else:
                response_text = self.llm.complete(
                    prompt=prompt,
                    system_prompt=CLASSIFICATION_SYSTEM_PROMPT
                )
            
            # Clean up response if needed (sometimes LLMs wrap in markdown code blocks)
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
                
            return json.loads(response_text)
        except Exception as e:
            logger.error(f"Error classifying figure {figure.get('figure_id')}: {e}")
            return {
                "label": "other",
                "confidence": 0.0,
                "reasoning": f"Error during classification: {str(e)}"
            }

def classify_figure(figure: Dict[str, Any], llm: LLMProvider) -> Dict[str, Any]:
    classifier = FigureClassifier(llm)
    return classifier.classify(figure)
