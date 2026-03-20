import json
import logging
from typing import Dict, Any
from agent.utils.llm_provider import LLMProvider
from agent.prompts.templates import FLUX_INTERPRETATION_SYSTEM_PROMPT, FLUX_INTERPRETATION_USER_PROMPT

logger = logging.getLogger(__name__)

class FluxInterpreter:
    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def interpret(self, figure: Dict[str, Any]) -> Dict[str, Any]:
        """Interprets a flux diagram using DeepSeek Vision."""
        prompt = FLUX_INTERPRETATION_USER_PROMPT.format(
            figure_id=figure.get("figure_id", "Unknown"),
            caption=figure.get("caption", ""),
            context_text=figure.get("context_text", "")
        )

        try:
            image_path = figure.get("page_image_path")
            if not image_path:
                raise ValueError("No image path provided for flux diagram interpretation.")

            response_text = self.llm.vision_complete(
                prompt=prompt,
                image_path=image_path,
                system_prompt=FLUX_INTERPRETATION_SYSTEM_PROMPT
            )
            
            # Clean up response
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
                
            return json.loads(response_text)
        except Exception as e:
            logger.error(f"Error interpreting flux diagram {figure.get('figure_id')}: {e}")
            return {
                "system": "unknown",
                "conditions": {
                    "temperature": "unknown",
                    "pressure": "unknown",
                    "equivalence_ratio": "unknown",
                    "residence_time": "unknown"
                },
                "major_species": [],
                "dominant_pathways": [],
                "quantitative_info": False,
                "usefulness": "unknown",
                "use_cases": [],
                "uncertainty": str(e),
                "confidence": 0.0
            }

def interpret_flux_diagram(figure: Dict[str, Any], llm: LLMProvider) -> Dict[str, Any]:
    interpreter = FluxInterpreter(llm)
    return interpreter.interpret(figure)
