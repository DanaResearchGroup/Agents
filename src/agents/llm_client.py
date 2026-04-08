"""Single point of LLM access. Wraps LiteLLM.

No other file in the codebase should import litellm, anthropic, or openai
directly. All LLM calls are routed through LLMClient.complete().
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

import litellm

log = logging.getLogger(__name__)


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

    @staticmethod
    def _build_example(model: type[BaseModel]) -> dict[str, Any]:
        """Build a concrete example dict covering every field of *model*."""
        _DEFAULTS: dict[type, Any] = {
            float: 0.5,
            str: "example",
            bool: True,
            int: 0,
            list: [],
        }
        example: dict[str, Any] = {}
        for name, field_info in model.model_fields.items():
            annotation = field_info.annotation
            # Unwrap Optional (Union[X, None]) to the inner type
            origin = getattr(annotation, "__origin__", None)
            if origin is type(int | str):  # types.UnionType (3.10+)
                args = [a for a in annotation.__args__ if a is not type(None)]
                annotation = args[0] if args else annotation
            example[name] = _DEFAULTS.get(annotation, "example")
        return example

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
        is_ollama = self.config.provider == "ollama"

        messages: list[dict[str, Any]] = []

        # ── Ollama structured-output fallback ────────────────────────────
        if response_model is not None and is_ollama:
            json_schema = json.dumps(
                response_model.model_json_schema(), indent=2,
            )
            example_json = json.dumps(
                self._build_example(response_model), indent=2,
            )
            schema_prompt = (
                "You must respond with valid JSON only. No explanation, no "
                "markdown, no code blocks. Raw JSON only.\n\n"
                f"Required schema:\n{json_schema}\n\n"
                "Example of a valid response (with placeholder values):\n"
                f"{example_json}\n\n"
                "Your response must include ALL fields shown in the example."
            )
            if system:
                system = f"{system}\n\n{schema_prompt}"
            else:
                system = schema_prompt

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

        if response_model is not None and not is_ollama:
            kwargs["response_format"] = {"type": "json_object"}

        # ── Call LLM (with retry for Ollama structured output) ───────────
        label = agent_name or "default"
        log.info(
            "LLM call | agent=%s | model=%s | structured=%s",
            label, model, response_model is not None,
        )

        max_attempts = 2 if (response_model is not None and is_ollama) else 1
        last_err: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                response = await litellm.acompletion(**kwargs)
            except Exception as exc:
                log.error("LLM call failed | agent=%s | error=%s", label, exc)
                raise
            text: str = response.choices[0].message.content

            if response_model is None:
                log.info(
                    "LLM response | agent=%s | tokens_approx=%d | type=text",
                    label, len(text) // 4,
                )
                return text

            log.debug("LLM raw response (attempt %d): %s", attempt, text)

            try:
                parsed = response_model.model_validate_json(text)
                log.info(
                    "LLM response | agent=%s | tokens_approx=%d | type=structured",
                    label, len(text) // 4,
                )
                return parsed
            except Exception as exc:
                last_err = exc
                if attempt < max_attempts:
                    log.warning(
                        "LLM parse failed, retrying | agent=%s | attempt=%d | raw=%s",
                        label, attempt, text[:100],
                    )
                    continue

        log.error("LLM call failed | agent=%s | error=%s", label, last_err)
        raise last_err  # type: ignore[misc]
