"""Tests for src/agents/llm_client.py — mock litellm, test config loading and routing."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel

from src.agents.llm_client import LLMClient, LLMConfig


# ── Config tests ─────────────────────────────────────────────────────────


class TestLLMConfig:
    def test_defaults(self):
        cfg = LLMConfig()
        assert cfg.provider == "anthropic"
        assert cfg.model == "claude-sonnet-4-6"
        assert cfg.api_key is None
        assert cfg.agent_overrides == {}

    def test_from_dict(self):
        cfg = LLMConfig.model_validate({
            "provider": "openai",
            "model": "gpt-4o",
            "api_key": "sk-test",
            "agent_overrides": {
                "condition_extraction": {"model": "gpt-4o-mini"},
            },
        })
        assert cfg.provider == "openai"
        assert cfg.agent_overrides["condition_extraction"]["model"] == "gpt-4o-mini"


class TestLLMClientInit:
    def test_from_config_object(self):
        cfg = LLMConfig(provider="ollama", model="llama3", base_url="http://localhost:11434")
        client = LLMClient(config=cfg)
        assert client.config.provider == "ollama"

    def test_from_yaml_file(self, tmp_path: Path):
        yaml_content = (
            "provider: anthropic\n"
            "model: claude-haiku-4-5\n"
            "api_key: sk-test-key\n"
            "agent_overrides:\n"
            "  reaction_mining:\n"
            "    model: claude-sonnet-4-6\n"
        )
        config_file = tmp_path / "llm_config.yaml"
        config_file.write_text(yaml_content)

        client = LLMClient(config_path=config_file)
        assert client.config.model == "claude-haiku-4-5"
        assert client.config.api_key == "sk-test-key"
        assert client.config.agent_overrides["reaction_mining"]["model"] == "claude-sonnet-4-6"

    def test_default_config(self):
        client = LLMClient()
        assert client.config.model == "claude-sonnet-4-6"


# ── Model resolution tests ──────────────────────────────────────────────


class TestModelResolution:
    def test_default_model(self):
        client = LLMClient(config=LLMConfig(model="claude-sonnet-4-6"))
        assert client._resolve_model(None) == "claude-sonnet-4-6"

    def test_agent_override(self):
        cfg = LLMConfig(
            model="claude-sonnet-4-6",
            agent_overrides={"condition_extraction": {"model": "claude-haiku-4-5"}},
        )
        client = LLMClient(config=cfg)
        assert client._resolve_model("condition_extraction") == "claude-haiku-4-5"
        assert client._resolve_model("reaction_mining") == "claude-sonnet-4-6"

    def test_unknown_agent_uses_default(self):
        cfg = LLMConfig(model="claude-sonnet-4-6")
        client = LLMClient(config=cfg)
        assert client._resolve_model("nonexistent_agent") == "claude-sonnet-4-6"


# ── Completion tests (mocked) ───────────────────────────────────────────


def _mock_response(content: str):
    """Build a mock LiteLLM response object."""
    msg = type("Message", (), {"content": content})()
    choice = type("Choice", (), {"message": msg})()
    return type("Response", (), {"choices": [choice]})()


class TestComplete:
    @pytest.mark.asyncio
    async def test_plain_text(self):
        client = LLMClient(config=LLMConfig())
        mock_resp = _mock_response("Hello, world!")

        with patch("src.agents.llm_client.litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_resp
            result = await client.complete("Say hello")

        assert result == "Hello, world!"
        mock_llm.assert_called_once()
        call_kwargs = mock_llm.call_args.kwargs
        assert call_kwargs["model"] == "claude-sonnet-4-6"
        assert len(call_kwargs["messages"]) == 1
        assert call_kwargs["messages"][0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_with_system_prompt(self):
        client = LLMClient(config=LLMConfig())
        mock_resp = _mock_response("OK")

        with patch("src.agents.llm_client.litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_resp
            await client.complete("Do something", system="You are a chemist")

        call_kwargs = mock_llm.call_args.kwargs
        assert len(call_kwargs["messages"]) == 2
        assert call_kwargs["messages"][0]["role"] == "system"

    @pytest.mark.asyncio
    async def test_structured_output(self):
        class SpeciesResult(BaseModel):
            name: str
            fraction: float

        payload = json.dumps({"name": "NH3", "fraction": 0.01})
        client = LLMClient(config=LLMConfig())
        mock_resp = _mock_response(payload)

        with patch("src.agents.llm_client.litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_resp
            result = await client.complete(
                "Extract species", response_model=SpeciesResult
            )

        assert isinstance(result, SpeciesResult)
        assert result.name == "NH3"
        assert result.fraction == 0.01
        call_kwargs = mock_llm.call_args.kwargs
        assert call_kwargs["response_format"] == {"type": "json_object"}

    @pytest.mark.asyncio
    async def test_agent_override_routing(self):
        cfg = LLMConfig(
            model="claude-sonnet-4-6",
            agent_overrides={"condition_extraction": {"model": "claude-haiku-4-5"}},
        )
        client = LLMClient(config=cfg)
        mock_resp = _mock_response("extracted")

        with patch("src.agents.llm_client.litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_resp
            await client.complete("Extract conditions", agent_name="condition_extraction")

        call_kwargs = mock_llm.call_args.kwargs
        assert call_kwargs["model"] == "claude-haiku-4-5"

    @pytest.mark.asyncio
    async def test_api_key_and_base_url_passed(self):
        cfg = LLMConfig(
            provider="ollama",
            model="llama3",
            api_key="test-key",
            base_url="http://localhost:11434",
        )
        client = LLMClient(config=cfg)
        mock_resp = _mock_response("local response")

        with patch("src.agents.llm_client.litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_resp
            await client.complete("Test prompt")

        call_kwargs = mock_llm.call_args.kwargs
        assert call_kwargs["api_key"] == "test-key"
        assert call_kwargs["api_base"] == "http://localhost:11434"
