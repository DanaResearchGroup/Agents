"""Single point of LLM access. Wraps LiteLLM.

No other file in the codebase should import litellm, anthropic, or openai
directly. All LLM calls are routed through LLMClient.complete().
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

import litellm


class LLMConfig(BaseModel):
    """Mirrors config/llm_config.yaml."""
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-6"
    api_key: str | None = None
    base_url: str | None = None
    agent_overrides: dict[str, dict[str, str]] = Field(default_factory=dict)


class LLMClient:
    """Single point of LLM access. Wraps LiteLLM."""

    def __init__(self, config: LLMConfig | None = None, config_path: Path | None = None) -> None:
        if config is not None:
            self.config = config
        elif config_path is not None:
            raw = yaml.safe_load(config_path.read_text())
            self.config = LLMConfig.model_validate(raw)
        else:
            self.config = LLMConfig()

    def _resolve_model(self, agent_name: str | None) -> str:
        """Return the model string, applying per-agent override if configured."""
        if agent_name and agent_name in self.config.agent_overrides:
            return self.config.agent_overrides[agent_name].get(
                "model", self.config.model
            )
        return self.config.model

    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        response_model: type[BaseModel] | None = None,
        agent_name: str | None = None,
        images: list[str] | None = None,
    ) -> str | BaseModel:
        """Send a completion request via LiteLLM.

        Args:
            prompt: User message content.
            system: Optional system message.
            response_model: If provided, parse the response into this Pydantic model.
            agent_name: If provided, look up per-agent model override in config.
            images: Optional list of base64-encoded images to include in the
                user message (vision models only).

        Returns:
            Plain text string, or a validated Pydantic model instance if
            response_model was provided.
        """
        model = self._resolve_model(agent_name)

        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})

        if images:
            content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
            for img in images:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img}"},
                })
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": prompt})

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if self.config.api_key:
            kwargs["api_key"] = self.config.api_key
        if self.config.base_url:
            kwargs["api_base"] = self.config.base_url

        if response_model is not None:
            kwargs["response_format"] = {"type": "json_object"}

        response = await litellm.acompletion(**kwargs)
        text: str = response.choices[0].message.content

        if response_model is not None:
            return response_model.model_validate_json(text)

        return text
