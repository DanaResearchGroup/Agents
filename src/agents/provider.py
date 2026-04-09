"""PydanticAI model factory. Maps LLMConfig to provider-specific model objects."""

from __future__ import annotations

import logging
from typing import Any

from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIModelProfile
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.groq import GroqProvider
from pydantic_ai.providers.openai import OpenAIProvider

from src.agents.llm_client import LLMConfig

log = logging.getLogger(__name__)


def _strip_null_content(messages: list[Any]) -> list[Any]:
    """Remove ``content`` key from any message dict where it is ``None``.

    Ollama's /v1 compatibility layer rejects ``"content": null`` with
    ``invalid message content type: <nil>``.  This is valid per the
    OpenAI spec, but Ollama's Go JSON unmarshaller treats missing and
    null differently: missing is fine, null triggers a type-switch
    error on the nil interface value.
    """
    cleaned = []
    for msg in messages:
        if isinstance(msg, dict) and msg.get("content") is None:
            msg = {k: v for k, v in msg.items() if k != "content"}
        cleaned.append(msg)
    return cleaned


class _OllamaModel(OpenAIChatModel):
    """OpenAI-compatible model with Ollama-specific fixes.

    Overrides ``_completions_create`` to sanitize messages at the last
    possible moment — after PydanticAI's ``_map_messages`` and right
    before the OpenAI SDK serializes the HTTP request.
    """

    async def _completions_create(
        self,
        messages: list[ModelMessage],
        stream: bool,
        model_settings: Any,
        model_request_parameters: Any,
    ) -> Any:
        # Let the parent build the openai_messages list via _map_messages
        # and assemble all other parameters, but intercept to sanitize.
        # We do this by temporarily patching self.client to capture & fix
        # the messages before they hit the wire.
        original_create = self.client.chat.completions.create

        async def patched_create(**kwargs: Any) -> Any:
            if "messages" in kwargs:
                kwargs["messages"] = _strip_null_content(kwargs["messages"])
            return await original_create(**kwargs)

        self.client.chat.completions.create = patched_create  # type: ignore[assignment]
        try:
            return await super()._completions_create(
                messages, stream, model_settings, model_request_parameters
            )
        finally:
            self.client.chat.completions.create = original_create  # type: ignore[assignment]

# Providers that use the OpenAI-compatible API with a custom base_url.
_OPENAI_COMPAT_DEFAULTS: dict[str, str] = {
    "deepseek": "https://api.deepseek.com/v1",
    "ollama": "http://localhost:11434/v1",
}


def make_model(
    config: LLMConfig,
    agent_name: str | None = None,
) -> AnthropicModel | OpenAIChatModel | GroqModel:
    """Build a PydanticAI model from LLMConfig.

    Supports:
      provider: anthropic  → AnthropicModel
      provider: openai     → OpenAIChatModel
      provider: groq       → GroqModel
      provider: deepseek   → OpenAIChatModel(base_url=https://api.deepseek.com/v1)
      provider: ollama     → OpenAIChatModel(base_url=http://localhost:11434/v1)

    Per-agent model override via agent_name:
      make_model(config, agent_name="condition_extraction")
      → uses config.agent_overrides["condition_extraction"]["model"] if present
    """
    model_name = config.model
    if agent_name and agent_name in config.agent_overrides:
        model_name = config.agent_overrides[agent_name].get("model", model_name)

    provider = config.provider.lower()
    base_url = config.base_url
    api_key = config.api_key or None  # treat empty string as None

    if provider == "anthropic":
        prov = AnthropicProvider(api_key=api_key, base_url=base_url)
        return AnthropicModel(model_name, provider=prov)

    if provider == "groq":
        prov = GroqProvider(api_key=api_key, base_url=base_url)
        return GroqModel(model_name, provider=prov)

    # openai, deepseek, ollama — all OpenAI-compatible
    if base_url is None:
        base_url = _OPENAI_COMPAT_DEFAULTS.get(provider)

    if provider == "ollama":
        api_key = api_key or "ollama"
        model_name = model_name.removeprefix("ollama/")
        if base_url and not base_url.rstrip("/").endswith("/v1"):
            base_url = base_url.rstrip("/") + "/v1"

        profile = OpenAIModelProfile(
            openai_supports_strict_tool_definition=False,
        )
        prov = OpenAIProvider(base_url=base_url, api_key=api_key)
        return _OllamaModel(model_name, provider=prov, profile=profile)

    prov = OpenAIProvider(base_url=base_url, api_key=api_key)
    return OpenAIChatModel(model_name, provider=prov)
