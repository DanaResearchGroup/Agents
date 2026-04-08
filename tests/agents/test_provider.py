"""Tests for PydanticAI model factory."""

from unittest.mock import patch

import pytest

from src.agents.llm_client import LLMConfig
from src.agents.provider import make_model


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
@patch("src.agents.provider.OpenAIModel")
def test_ollama_config(mock_model_cls, mock_prov_cls):
    cfg = LLMConfig(provider="ollama", model="llama3.1:8b")
    result = make_model(cfg)

    mock_prov_cls.assert_called_once_with(
        base_url="http://localhost:11434/v1", api_key="ollama"
    )
    mock_model_cls.assert_called_once_with("llama3.1:8b", provider=mock_prov_cls.return_value)
    assert result is mock_model_cls.return_value


@patch("src.agents.provider.OpenAIProvider")
@patch("src.agents.provider.OpenAIModel")
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


@patch("src.agents.provider.AnthropicProvider")
@patch("src.agents.provider.AnthropicModel")
def test_empty_api_key_treated_as_none(mock_model_cls, mock_prov_cls):
    cfg = LLMConfig(provider="anthropic", model="claude-sonnet-4-6", api_key="")
    make_model(cfg)

    mock_prov_cls.assert_called_once_with(api_key=None, base_url=None)
