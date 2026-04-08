"""Tests for ReactionMiningAgent — all LLM calls mocked."""

from __future__ import annotations

import pytest

from src.agents.llm_client import LLMClient, LLMConfig
from src.agents.reaction_mining import ReactionMiningAgent
from src.schemas.experimental import (
    FigureClassification,
    FluxInterpretation,
    SensitivityInterpretation,
)

FAKE_IMAGE_B64 = "iVBORw0KGgoAAAANSUhEUg=="  # tiny stub


@pytest.fixture
def client() -> LLMClient:
    return LLMClient(config=LLMConfig())


@pytest.fixture
def agent(client: LLMClient) -> ReactionMiningAgent:
    return ReactionMiningAgent(llm_client=client)


# ── classify_figure ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_classify_figure_with_image(agent: ReactionMiningAgent, monkeypatch):
    """Vision path: image_b64 is provided, images kwarg forwarded to LLM."""
    async def mock_complete(*, prompt, system, response_model, agent_name, images):
        assert images == [FAKE_IMAGE_B64]
        assert response_model is FigureClassification
        return FigureClassification(
            label="flux diagram", confidence=0.92, reasoning="arrows visible",
        )

    monkeypatch.setattr(agent.llm_client, "complete", mock_complete)

    result = await agent.classify_figure(
        image_b64=FAKE_IMAGE_B64,
        caption="Flux diagram of ethanol pyrolysis",
    )
    assert isinstance(result, FigureClassification)
    assert result.label == "flux diagram"
    assert result.confidence == pytest.approx(0.92)


@pytest.mark.asyncio
async def test_classify_figure_text_only(agent: ReactionMiningAgent, monkeypatch):
    """Text-only path: image_b64 is None, images kwarg is None."""
    async def mock_complete(*, prompt, system, response_model, agent_name, images):
        assert images is None
        return FigureClassification(
            label="sensitivity analysis", confidence=0.85, reasoning="bar chart",
        )

    monkeypatch.setattr(agent.llm_client, "complete", mock_complete)

    result = await agent.classify_figure(image_b64=None, caption="Sensitivity plot")
    assert result.label == "sensitivity analysis"


@pytest.mark.asyncio
async def test_classify_figure_error_fallback(agent: ReactionMiningAgent, monkeypatch):
    """On LLM error, returns a safe default FigureClassification."""
    async def mock_complete(**kwargs):
        raise RuntimeError("API unreachable")

    monkeypatch.setattr(agent.llm_client, "complete", mock_complete)

    result = await agent.classify_figure(image_b64=None, caption="broken")
    assert result.label == "other"
    assert result.confidence == 0.0
    assert "API unreachable" in result.reasoning


# ── interpret_flux ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_interpret_flux_with_image(agent: ReactionMiningAgent, monkeypatch):
    async def mock_complete(*, prompt, system, response_model, agent_name, images):
        assert images == [FAKE_IMAGE_B64]
        return FluxInterpretation(
            system="ethanol pyrolysis",
            major_species=["C2H5OH", "CH3CHO"],
            quantitative_info=True,
            usefulness="high",
            confidence=0.88,
        )

    monkeypatch.setattr(agent.llm_client, "complete", mock_complete)

    result = await agent.interpret_flux(
        image_b64=FAKE_IMAGE_B64,
        caption="Flux diagram at 1200K",
    )
    assert isinstance(result, FluxInterpretation)
    assert result.system == "ethanol pyrolysis"
    assert result.quantitative_info is True


@pytest.mark.asyncio
async def test_interpret_flux_text_only(agent: ReactionMiningAgent, monkeypatch):
    async def mock_complete(*, prompt, system, response_model, agent_name, images):
        assert images is None
        return FluxInterpretation(system="propane combustion", confidence=0.60)

    monkeypatch.setattr(agent.llm_client, "complete", mock_complete)

    result = await agent.interpret_flux(image_b64=None, caption="ROP analysis")
    assert result.system == "propane combustion"


@pytest.mark.asyncio
async def test_interpret_flux_error_fallback(agent: ReactionMiningAgent, monkeypatch):
    async def mock_complete(**kwargs):
        raise RuntimeError("timeout")

    monkeypatch.setattr(agent.llm_client, "complete", mock_complete)

    result = await agent.interpret_flux(image_b64=None, caption="broken")
    assert result.system == "unknown"
    assert "timeout" in result.uncertainty


# ── interpret_sensitivity ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_interpret_sensitivity_with_image(
    agent: ReactionMiningAgent, monkeypatch,
):
    async def mock_complete(*, prompt, system, response_model, agent_name, images):
        assert images == [FAKE_IMAGE_B64]
        return SensitivityInterpretation(
            target_property="ignition delay",
            usefulness="high",
            confidence=0.91,
        )

    monkeypatch.setattr(agent.llm_client, "complete", mock_complete)

    result = await agent.interpret_sensitivity(
        image_b64=FAKE_IMAGE_B64,
        caption="IDT sensitivity coefficients at 1300K",
    )
    assert isinstance(result, SensitivityInterpretation)
    assert result.target_property == "ignition delay"
    assert result.confidence == pytest.approx(0.91)


@pytest.mark.asyncio
async def test_interpret_sensitivity_text_only(
    agent: ReactionMiningAgent, monkeypatch,
):
    async def mock_complete(*, prompt, system, response_model, agent_name, images):
        assert images is None
        return SensitivityInterpretation(
            target_property="flame speed", confidence=0.70,
        )

    monkeypatch.setattr(agent.llm_client, "complete", mock_complete)

    result = await agent.interpret_sensitivity(
        image_b64=None, caption="Flame speed sensitivity",
    )
    assert result.target_property == "flame speed"


@pytest.mark.asyncio
async def test_interpret_sensitivity_error_fallback(
    agent: ReactionMiningAgent, monkeypatch,
):
    async def mock_complete(**kwargs):
        raise RuntimeError("bad gateway")

    monkeypatch.setattr(agent.llm_client, "complete", mock_complete)

    result = await agent.interpret_sensitivity(image_b64=None, caption="broken")
    assert result.target_property == "unknown"
    assert "bad gateway" in result.uncertainty
