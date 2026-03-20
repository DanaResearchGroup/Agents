import os
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import openai
from dotenv import load_dotenv

load_dotenv()

class LLMProvider(ABC):
    @abstractmethod
    def complete(self, prompt: str, system_prompt: str = "You are a helpful assistant.") -> str:
        pass

    @abstractmethod
    def vision_complete(self, prompt: str, image_path: str, system_prompt: str = "You are a helpful assistant.") -> str:
        pass

class DeepSeekProvider(LLMProvider):
    def __init__(self, api_key: Optional[str] = None, model: str = "deepseek-chat"):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY must be provided or set as an environment variable.")
        
        # DeepSeek often uses OpenAI-compatible API
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com" # Assuming this is the base URL
        )
        self.model = model

    def complete(self, prompt: str, system_prompt: str = "You are a helpful assistant.") -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content

    def vision_complete(self, prompt: str, image_path: str, system_prompt: str = "You are a helpful assistant.") -> str:
        """
        Tries to perform a vision-based completion. 
        If it fails (e.g. model doesn't support it), falls back to text-only using the prompt.
        """
        # Known text-only models in DeepSeek API
        if self.model in ["deepseek-chat", "deepseek-reasoner"]:
            logging.info(f"Model {self.model} does not support vision. Skipping vision attempt.")
            return self._text_fallback(prompt, system_prompt)

        import base64
        try:
            def encode_image(img_path):
                with open(img_path, "rb") as image_file:
                    return base64.b64encode(image_file.read()).decode('utf-8')

            base64_image = encode_image(image_path)

            # Attempt vision call - some proxies/endpoints might support this
            response = self.client.chat.completions.create(
                model=self.model, # Using the configured model
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                },
                            },
                        ],
                    }
                ],
                max_tokens=1000,
            )
            return response.choices[0].message.content
        except Exception as e:
            # Fallback to text-only if vision fails
            logging.warning(f"Vision call failed ({e}), falling back to text-only.")
            return self._text_fallback(prompt, system_prompt)

    def _text_fallback(self, prompt: str, system_prompt: str) -> str:
        text_prompt = f"{prompt}\n\n[Note: Image was provided but could not be processed by the model. Please categorize/interpret based on the text context and caption provided above.]"
        return self.complete(text_prompt, system_prompt)

def get_llm_provider(provider_type: str = "deepseek") -> LLMProvider:
    if provider_type == "deepseek":
        return DeepSeekProvider()
    else:
        raise ValueError(f"Unsupported provider type: {provider_type}")
