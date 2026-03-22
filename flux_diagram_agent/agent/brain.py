import os
import json
import logging
import base64
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import openai
from dotenv import load_dotenv

from .prompts import (
    CLASSIFICATION_SYSTEM_PROMPT, CLASSIFICATION_USER_PROMPT,
    FLUX_INTERPRETATION_SYSTEM_PROMPT, FLUX_INTERPRETATION_USER_PROMPT,
    SENSITIVITY_INTERPRETATION_SYSTEM_PROMPT, SENSITIVITY_INTERPRETATION_USER_PROMPT
)

load_dotenv()
logger = logging.getLogger(__name__)

# --- LLM Provider ---

class LLMProvider(ABC):
    @abstractmethod
    def complete(self, prompt: str, system_prompt: str = "You are a helpful assistant.") -> str:
        pass

    @abstractmethod
    def vision_complete(self, prompt: str, image_path: str, system_prompt: str = "You are a helpful assistant.") -> str:
        pass

class OpenAICompatibleProvider(LLMProvider):
    """Base class for any provider that follows the OpenAI API format."""
    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key = api_key
        self.model = model
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)

    def complete(self, prompt: str, system_prompt: str = "You are a helpful assistant.") -> str:
        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        }
        if "reasoner" not in self.model:
            kwargs["response_format"] = {"type": "json_object"}
            
        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

    def vision_complete(self, prompt: str, image_path: str, system_prompt: str = "You are a helpful assistant.") -> str:
        if self.model in ["deepseek-chat", "deepseek-reasoner"]:
            logger.info(f"Model {self.model} does not support vision. Skipping vision attempt.")
            return self._text_fallback(prompt, system_prompt)

        try:
            def encode_image(img_path):
                with open(img_path, "rb") as image_file:
                    return base64.b64encode(image_file.read()).decode('utf-8')

            base64_image = encode_image(image_path)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                            },
                        ],
                    }
                ],
                max_tokens=2000,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.warning(f"Vision call for {self.model} failed ({e}), falling back to text-only.")
            return self._text_fallback(prompt, system_prompt)

    def _text_fallback(self, prompt: str, system_prompt: str) -> str:
        text_prompt = f"{prompt}\n\n[Note: Image was provided but could not be processed by the model. Please categorize/interpret based on the text context and caption provided above.]"
        return self.complete(text_prompt, system_prompt)

class DeepSeekProvider(OpenAICompatibleProvider):
    def __init__(self, api_key: Optional[str] = None, model: str = "deepseek-reasoner"):
        key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not key:
            raise ValueError("DEEPSEEK_API_KEY must be provided.")
        super().__init__(api_key=key, base_url="https://api.deepseek.com", model=model)

class GLMProvider(OpenAICompatibleProvider):
    def __init__(self, api_key: Optional[str] = None, model: str = "glm-4v-plus"):
        key = api_key or os.getenv("GLM_API_KEY")
        if not key:
            raise ValueError("GLM_API_KEY must be provided.")
        # Zhipu AI / GLM OpenAI-compatible endpoint
        super().__init__(api_key=key, base_url="https://open.bigmodel.cn/api/paas/v4/", model=model)

def get_llm_provider(provider_type: str = "deepseek", model: Optional[str] = None) -> LLMProvider:
    provider_type = provider_type.lower()
    if provider_type == "deepseek":
        return DeepSeekProvider(model=model or "deepseek-reasoner")
    elif provider_type == "glm":
        return GLMProvider(model=model or "glm-4v-plus")
    raise ValueError(f"Unsupported provider type: {provider_type}")

# --- AI Task Logic ---

class FigureClassifier:
    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def classify(self, figure: Dict[str, Any]) -> Dict[str, Any]:
        prompt = CLASSIFICATION_USER_PROMPT.format(
            figure_id=figure.get("figure_id", "Unknown"),
            caption=figure.get("caption", ""),
            context_text=figure.get("context_text", "")
        )
        try:
            image_path = figure.get("page_image_path")
            response_text = self.llm.vision_complete(prompt=prompt, image_path=image_path, system_prompt=CLASSIFICATION_SYSTEM_PROMPT) if image_path else self.llm.complete(prompt=prompt, system_prompt=CLASSIFICATION_SYSTEM_PROMPT)
            
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            return json.loads(response_text)
        except Exception as e:
            logger.error(f"Error classifying figure {figure.get('figure_id')}: {e}")
            return {"label": "other", "confidence": 0.0, "reasoning": f"Error during classification: {str(e)}"}

def classify_figure(figure: Dict[str, Any], llm: LLMProvider) -> Dict[str, Any]:
    return FigureClassifier(llm).classify(figure)

class FluxInterpreter:
    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def interpret(self, figure: Dict[str, Any]) -> Dict[str, Any]:
        prompt = FLUX_INTERPRETATION_USER_PROMPT.format(
            figure_id=figure.get("figure_id", "Unknown"),
            caption=figure.get("caption", ""),
            context_text=figure.get("context_text", "")
        )
        try:
            image_path = figure.get("page_image_path")
            if not image_path: raise ValueError("No image path provided for flux diagram interpretation.")
            response_text = self.llm.vision_complete(prompt=prompt, image_path=image_path, system_prompt=FLUX_INTERPRETATION_SYSTEM_PROMPT)
            
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            return json.loads(response_text)
        except Exception as e:
            logger.error(f"Error interpreting flux diagram {figure.get('figure_id')}: {e}")
            return {
                "system": "unknown", "conditions": {"temperature": "unknown", "pressure": "unknown", "equivalence_ratio": "unknown", "residence_time": "unknown"},
                "major_species": [], "dominant_pathways": [], "quantitative_info": False, "usefulness": "unknown", "use_cases": [], "uncertainty": str(e), "confidence": 0.0
            }

class SensitivityInterpreter:
    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def interpret(self, figure: Dict[str, Any]) -> Dict[str, Any]:
        prompt = SENSITIVITY_INTERPRETATION_USER_PROMPT.format(
            figure_id=figure.get("figure_id", "Unknown"),
            caption=figure.get("caption", ""),
            context_text=figure.get("context_text", "")
        )
        try:
            image_path = figure.get("page_image_path")
            if not image_path:
                raise ValueError("No image path provided for sensitivity analysis interpretation.")

            response_text = self.llm.vision_complete(
                prompt=prompt,
                image_path=image_path,
                system_prompt=SENSITIVITY_INTERPRETATION_SYSTEM_PROMPT
            )
            
            # Clean up response
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
                
            return json.loads(response_text)
        except Exception as e:
            logger.error(f"Error interpreting sensitivity analysis {figure.get('figure_id')}: {e}")
            return {
                "target_property": "unknown",
                "conditions": {"temperature": "unknown", "pressure": "unknown", "equivalence_ratio": "unknown"},
                "top_reactions": [],
                "usefulness": "unknown",
                "uncertainty": str(e),
                "confidence": 0.0
            }

def interpret_sensitivity_analysis(figure: Dict[str, Any], llm: LLMProvider) -> Dict[str, Any]:
    return SensitivityInterpreter(llm).interpret(figure)

def interpret_flux_diagram(figure: Dict[str, Any], llm: LLMProvider) -> Dict[str, Any]:
    return FluxInterpreter(llm).interpret(figure)
