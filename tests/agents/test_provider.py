"""Tests for PydanticAI model factory."""

from unittest.mock import patch

import pytest

from src.agents.llm_client import LLMConfig
from src.agents.provider import _OllamaModel, make_model


@patch("src.agents.provider.AnthropicProvider")
@patch("src.agents.provider.AnthropicModel")
def test_anthropic_config(mock_model_cls, mock_prov_cls):
    cfg = LLMConfig(provider="anthropic", model="claude-sonnet-4-6", api_key="sk-test")
    result = make_model(cfg)

    mock_prov_cls.assert_called_once_with(api_key="sk-test", base_url=None)
    mock_model_cls.assert_called_once_with("claude-sonnet-4-6", provider=mock_prov_cls.return_value)
    assert result is mock_model_cls.return_value


@patch("src.agents.provider.OpenAIProvider")
@patch("src.agents.provider.OpenAIModel")
def test_openai_config(mock_model_cls, mock_prov_cls):
    cfg = LLMConfig(provider="openai", model="gpt-4o", api_key="sk-oai")
    result = make_model(cfg)

    mock_prov_cls.assert_called_once_with(base_url=None, api_key="sk-oai")
    mock_model_cls.assert_called_once_with("gpt-4o", provider=mock_prov_cls.return_value)
    assert result is mock_model_cls.return_value


@patch("src.agents.provider.OpenAIProvider")
@patch("src.agents.provider.OpenAIModel")
def test_deepseek_config(mock_model_cls, mock_prov_cls):
    cfg = LLMConfig(provider="deepseek", model="deepseek-chat", api_key="sk-ds")
    result = make_model(cfg)

    mock_prov_cls.assert_called_once_with(
        base_url="https://api.deepseek.com/v1", api_key="sk-ds"
    )
    mock_model_cls.assert_called_once_with("deepseek-chat", provider=mock_prov_cls.return_value)
    assert result is mock_model_cls.return_value


@patch("src.agents.provider.OpenAIProvider")
@patch("src.agents.provider._OllamaModel")
def test_ollama_config(mock_model_cls, mock_prov_cls):
    cfg = LLMConfig(provider="ollama", model="llama3.1:8b")
    result = make_model(cfg)

    mock_prov_cls.assert_called_once_with(
        base_url="http://localhost:11434/v1", api_key="ollama"
    )
    mock_model_cls.assert_called_once_with("llama3.1:8b", provider=mock_prov_cls.return_value)
    assert result is mock_model_cls.return_value


@patch("src.agents.provider.OpenAIProvider")
@patch("src.agents.provider._OllamaModel")
def test_ollama_preserves_explicit_api_key(mock_model_cls, mock_prov_cls):
    cfg = LLMConfig(provider="ollama", model="llama3.1:8b", api_key="custom-key")
    make_model(cfg)

    mock_prov_cls.assert_called_once_with(
        base_url="http://localhost:11434/v1", api_key="custom-key"
    )


@patch("src.agents.provider.GroqProvider")
@patch("src.agents.provider.GroqModel")
def test_groq_config(mock_model_cls, mock_prov_cls):
    cfg = LLMConfig(provider="groq", model="llama-3.3-70b-versatile", api_key="gsk-test")
    result = make_model(cfg)

    mock_prov_cls.assert_called_once_with(api_key="gsk-test", base_url=None)
    mock_model_cls.assert_called_once_with(
        "llama-3.3-70b-versatile", provider=mock_prov_cls.return_value
    )
    assert result is mock_model_cls.return_value


@patch("src.agents.provider.AnthropicProvider")
@patch("src.agents.provider.AnthropicModel")
def test_agent_override_selects_model(mock_model_cls, mock_prov_cls):
    cfg = LLMConfig(
        provider="anthropic",
        model="claude-sonnet-4-6",
        api_key="sk-test",
        agent_overrides={"paper_reader": {"model": "claude-haiku-4-5"}},
    )
    make_model(cfg, agent_name="paper_reader")

    mock_model_cls.assert_called_once_with("claude-haiku-4-5", provider=mock_prov_cls.return_value)


@patch("src.agents.provider.AnthropicProvider")
@patch("src.agents.provider.AnthropicModel")
def test_agent_override_missing_uses_default(mock_model_cls, mock_prov_cls):
    cfg = LLMConfig(
        provider="anthropic",
        model="claude-sonnet-4-6",
        api_key="sk-test",
        agent_overrides={"other_agent": {"model": "claude-haiku-4-5"}},
    )
    make_model(cfg, agent_name="paper_reader")

    mock_model_cls.assert_called_once_with("claude-sonnet-4-6", provider=mock_prov_cls.return_value)


@patch("src.agents.provider.OpenAIProvider")
@patch("src.agents.provider.OpenAIModel")
def test_deepseek_with_explicit_base_url(mock_model_cls, mock_prov_cls):
    cfg = LLMConfig(
        provider="deepseek",
        model="deepseek-chat",
        api_key="sk-ds",
        base_url="https://custom.deepseek.proxy/v1",
    )
    make_model(cfg)

    mock_prov_cls.assert_called_once_with(
        base_url="https://custom.deepseek.proxy/v1", api_key="sk-ds"
    )


@patch("src.agents.provider.OpenAIProvider")
@patch("src.agents.provider._OllamaModel")
def test_ollama_strips_prefix(mock_model_cls, mock_prov_cls):
    cfg = LLMConfig(provider="ollama", model="ollama/llama3.1:8b")
    make_model(cfg)

    mock_model_cls.assert_called_once_with("llama3.1:8b", provider=mock_prov_cls.return_value)


@patch("src.agents.provider.OpenAIProvider")
@patch("src.agents.provider._OllamaModel")
def test_ollama_appends_v1_when_missing(mock_model_cls, mock_prov_cls):
    cfg = LLMConfig(
        provider="ollama", model="llama3.1:8b",
        base_url="http://localhost:11434",
    )
    make_model(cfg)

    mock_prov_cls.assert_called_once_with(
        base_url="http://localhost:11434/v1", api_key="ollama"
    )


@patch("src.agents.provider.OpenAIProvider")
@patch("src.agents.provider._OllamaModel")
def test_ollama_preserves_v1_when_present(mock_model_cls, mock_prov_cls):
    cfg = LLMConfig(
        provider="ollama", model="llama3.1:8b",
        base_url="http://localhost:11434/v1",
    )
    make_model(cfg)

    mock_prov_cls.assert_called_once_with(
        base_url="http://localhost:11434/v1", api_key="ollama"
    )


@patch("src.agents.provider.AnthropicProvider")
@patch("src.agents.provider.AnthropicModel")
def test_empty_api_key_treated_as_none(mock_model_cls, mock_prov_cls):
    cfg = LLMConfig(provider="anthropic", model="claude-sonnet-4-6", api_key="")
    make_model(cfg)

    mock_prov_cls.assert_called_once_with(api_key=None, base_url=None)


def test_ollama_model_is_ollama_subclass():
    """Ollama provider returns _OllamaModel, not plain OpenAIModel."""
    with (
        patch("src.agents.provider.OpenAIProvider"),
        patch("src.agents.provider._OllamaModel") as mock_cls,
    ):
        cfg = LLMConfig(provider="ollama", model="qwen2.5:7b-instruct")
        make_model(cfg)
        mock_cls.assert_called_once()


@pytest.mark.asyncio
async def test_ollama_model_removes_null_content():
    """_OllamaModel removes content key when it is None."""
    from unittest.mock import AsyncMock, MagicMock

    model = MagicMock(spec=_OllamaModel)
    # Simulate parent _map_messages returning messages with None content
    parent_result = [
        {"role": "system", "content": "You are helpful."},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "1", "function": {"name": "get_evidence"}}]},
        {"role": "tool", "content": "Page 1: T=1200K", "tool_call_id": "1"},
    ]
    # Call the real _map_messages with a mocked super()
    with patch.object(
        _OllamaModel.__mro__[1], "_map_messages",
        new=AsyncMock(return_value=parent_result),
    ):
        result = await _OllamaModel._map_messages(model, messages=[], model_request_parameters=MagicMock())

    # Null content key removed entirely
    assert result[0]["content"] == "You are helpful."
    assert "content" not in result[1]
    assert result[1]["tool_calls"] == [{"id": "1", "function": {"name": "get_evidence"}}]
    assert result[2]["content"] == "Page 1: T=1200K"
