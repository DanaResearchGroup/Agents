"""PydanticAI model factory. Maps LLMConfig to provider-specific model objects."""

from __future__ import annotations

from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.groq import GroqProvider
from pydantic_ai.providers.openai import OpenAIProvider

from src.agents.llm_client import LLMConfig

# Providers that use the OpenAI-compatible API with a custom base_url.
_OPENAI_COMPAT_DEFAULTS: dict[str, str] = {
    "deepseek": "https://api.deepseek.com/v1",
    "ollama": "http://localhost:11434/v1",
}


def make_model(
    config: LLMConfig,
    agent_name: str | None = None,
) -> AnthropicModel | OpenAIModel | GroqModel:
    """Build a PydanticAI model from LLMConfig.

    Supports:
      provider: anthropic  → AnthropicModel
      provider: openai     → OpenAIModel
      provider: groq       → GroqModel
      provider: deepseek   → OpenAIModel(base_url=https://api.deepseek.com/v1)
      provider: ollama     → OpenAIModel(base_url=http://localhost:11434/v1)

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

    prov = OpenAIProvider(base_url=base_url, api_key=api_key)
    return OpenAIModel(model_name, provider=prov)
