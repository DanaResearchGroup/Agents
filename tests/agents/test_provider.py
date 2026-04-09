"""Tests for PydanticAI model factory."""

from unittest.mock import ANY, patch

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
@patch("src.agents.provider.OpenAIChatModel")
def test_openai_config(mock_model_cls, mock_prov_cls):
    cfg = LLMConfig(provider="openai", model="gpt-4o", api_key="sk-oai")
    result = make_model(cfg)

    mock_prov_cls.assert_called_once_with(base_url=None, api_key="sk-oai")
    mock_model_cls.assert_called_once_with("gpt-4o", provider=mock_prov_cls.return_value)
    assert result is mock_model_cls.return_value


@patch("src.agents.provider.OpenAIProvider")
@patch("src.agents.provider.OpenAIChatModel")
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
    mock_model_cls.assert_called_once_with("llama3.1:8b", provider=mock_prov_cls.return_value, profile=ANY)
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
@patch("src.agents.provider.OpenAIChatModel")
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

    mock_model_cls.assert_called_once_with("llama3.1:8b", provider=mock_prov_cls.return_value, profile=ANY)


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
    """Ollama provider returns _OllamaModel, not plain OpenAIChatModel."""
    with (
        patch("src.agents.provider.OpenAIProvider"),
        patch("src.agents.provider._OllamaModel") as mock_cls,
    ):
        cfg = LLMConfig(provider="ollama", model="qwen2.5:7b-instruct")
        make_model(cfg)
        mock_cls.assert_called_once()


def test_ollama_profile_disables_strict_tool_definitions():
    """Ollama profile sets openai_supports_strict_tool_definition=False."""
    from pydantic_ai.models.openai import OpenAIModelProfile

    with patch("src.agents.provider.OpenAIProvider"):
        cfg = LLMConfig(provider="ollama", model="qwen2.5:14b")
        model = make_model(cfg)

    profile = OpenAIModelProfile.from_profile(model.profile)
    assert profile.openai_supports_strict_tool_definition is False


def test_strip_null_content_removes_null():
    """_strip_null_content removes content key when it is None."""
    from src.agents.provider import _strip_null_content

    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "1", "function": {"name": "get_evidence"}}]},
        {"role": "tool", "content": "Page 1: T=1200K", "tool_call_id": "1"},
        {"role": "assistant", "content": None},
    ]
    result = _strip_null_content(messages)

    assert result[0]["content"] == "You are helpful."
    assert "content" not in result[1]
    assert result[1]["tool_calls"] == [{"id": "1", "function": {"name": "get_evidence"}}]
    assert result[2]["content"] == "Page 1: T=1200K"
    assert "content" not in result[3]


def test_strip_null_content_preserves_non_null():
    """_strip_null_content leaves messages with real content untouched."""
    from src.agents.provider import _strip_null_content

    messages = [
        {"role": "system", "content": "System prompt."},
        {"role": "assistant", "content": "I will help."},
        {"role": "user", "content": "Thanks."},
    ]
    result = _strip_null_content(messages)

    assert all("content" in msg for msg in result)
    assert result[0]["content"] == "System prompt."
    assert result[1]["content"] == "I will help."
    assert result[2]["content"] == "Thanks."


@pytest.mark.asyncio
async def test_ollama_completions_create_strips_null_content():
    """_completions_create sanitizes messages before the API call."""
    from unittest.mock import AsyncMock, MagicMock

    from pydantic_ai.models.openai import ModelRequestParameters

    captured_kwargs = {}

    async def fake_create(**kwargs):
        captured_kwargs.update(kwargs)
        resp = MagicMock()
        resp.model_dump.return_value = {
            "id": "x", "object": "chat.completion", "created": 0,
            "model": "test", "choices": [{
                "index": 0, "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }], "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        return resp

    prov = MagicMock()
    model = _OllamaModel.__new__(_OllamaModel)
    model._model_name = "test"
    model._provider = prov
    model._profile = None
    model.client = MagicMock()
    model.client.chat.completions.create = AsyncMock(side_effect=fake_create)

    params = ModelRequestParameters(
        output_mode="text", output_object=None, allow_text_output=True,
    )

    # Simulate _map_messages returning messages with null content
    mapped = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
        {"role": "tool", "content": "result", "tool_call_id": "1"},
    ]
    with patch.object(
        _OllamaModel.__mro__[1], "_map_messages",
        new=AsyncMock(return_value=mapped),
    ):
        await model._completions_create(
            messages=[], stream=False,
            model_settings={}, model_request_parameters=params,
        )

    sent_msgs = captured_kwargs["messages"]
    assert sent_msgs[0]["content"] == "hello"
    assert "content" not in sent_msgs[1]
    assert sent_msgs[2]["content"] == "result"
