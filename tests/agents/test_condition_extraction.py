"""Tests for ConditionExtractionAgent evidence-grounded extraction."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.condition_extraction import (
    ConditionExtractionAgent,
    _HIGH_SIGNAL_KINDS,
    _MAX_SNIPPETS,
    _TRUNCATED_SNIPPETS,
    _WORD_BUDGET,
)
from src.agents.llm_client import LLMClient, LLMConfig
from src.schemas.experimental import (
    EvidenceSnippet,
    ExtractionResult,
    PaperDocument,
    PageText,
    SimConditions,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_snippet(
    kind: str = "temperature",
    value_text: str = "1200 K",
    source_text: str = "Experiments were conducted at 1200 K",
    confidence: float = 0.90,
    normalized_value: str | None = None,
) -> EvidenceSnippet:
    return EvidenceSnippet(
        kind=kind,
        value_text=value_text,
        page_num=1,
        source_text=source_text,
        confidence=confidence,
        normalized_value=normalized_value,
    )


def _make_paper() -> PaperDocument:
    return PaperDocument(
        pdf_path="/tmp/test.pdf",
        title="Test Paper",
        pages=[PageText(page_num=1, text="shock tube T=1200K 1 atm H2/O2/Ar")],
        captions=[],
    )


@pytest.fixture()
def llm_client() -> LLMClient:
    config = LLMConfig(provider="anthropic", model="test-model")
    client = LLMClient(config)
    client.complete = AsyncMock(
        return_value=ExtractionResult(conditions=[], confidence=0.5)
    )
    return client


# ── Tests ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_extract_with_evidence(llm_client: LLMClient):
    """Evidence snippets appear formatted in the LLM prompt."""
    evidence = [
        _make_snippet("temperature", "1200-1600 K", "at temperatures between 1200 and 1600 K", 0.92),
        _make_snippet("pressure", "1 atm", "at atmospheric pressure", 0.88),
        _make_snippet("composition", "H2:0.02, O2:0.02, Ar:0.96", "stoichiometric H2/O2 in argon", 0.75),
    ]

    agent = ConditionExtractionAgent(llm_client)
    await agent.extract(_make_paper(), evidence=evidence)

    call_kwargs = llm_client.complete.call_args[1]
    prompt = call_kwargs["prompt"]

    # All three kinds should appear, uppercased
    assert "[TEMPERATURE]" in prompt
    assert "[PRESSURE]" in prompt
    assert "[COMPOSITION]" in prompt

    # Source texts should be included
    assert "at temperatures between 1200 and 1600 K" in prompt
    assert "at atmospheric pressure" in prompt
    assert "stoichiometric H2/O2 in argon" in prompt

    # Evidence-grounded instruction should be present
    assert "Only extract conditions supported by the evidence" in prompt


@pytest.mark.asyncio
async def test_extract_evidence_truncated_at_20(llm_client: LLMClient):
    """Only top 20 snippets (by confidence) are included."""
    evidence = [
        _make_snippet("temperature", f"{1000 + i} K", f"temp {i}", confidence=round(0.50 + i * 0.02, 2))
        for i in range(25)
    ]

    agent = ConditionExtractionAgent(llm_client)
    await agent.extract(_make_paper(), evidence=evidence)

    prompt = llm_client.complete.call_args[1]["prompt"]
    # Count [TEMPERATURE] occurrences — should be capped at 20
    assert prompt.count("[TEMPERATURE]") == _MAX_SNIPPETS


@pytest.mark.asyncio
async def test_extract_fallback_no_evidence(llm_client: LLMClient):
    """With evidence=None, the fallback prompt uses raw paper text."""
    agent = ConditionExtractionAgent(llm_client)
    paper = _make_paper()
    await agent.extract(paper, evidence=None)

    prompt = llm_client.complete.call_args[1]["prompt"]

    # Fallback prompt should contain the paper text preamble
    assert "Paper text:" in prompt
    # Should NOT contain evidence-grounded instructions
    assert "Evidence extracted from paper" not in prompt


@pytest.mark.asyncio
async def test_extract_word_budget_truncation(llm_client: LLMClient):
    """When evidence context exceeds 600 words, rebuild with 12 snippets."""
    # Each snippet source_text ~60 words → 20 snippets ≈ 1200+ words
    long_source = " ".join(["word"] * 60)
    evidence = [
        _make_snippet(
            "temperature",
            f"{1000 + i} K",
            long_source,
            confidence=round(0.99 - i * 0.01, 2),
        )
        for i in range(20)
    ]

    agent = ConditionExtractionAgent(llm_client)
    await agent.extract(_make_paper(), evidence=evidence)

    prompt = llm_client.complete.call_args[1]["prompt"]
    # Should have been truncated to 12
    assert prompt.count("[TEMPERATURE]") == _TRUNCATED_SNIPPETS


@pytest.mark.asyncio
async def test_extract_filters_low_signal_kinds(llm_client: LLMClient):
    """Snippets with non-high-signal kinds (e.g. 'mechanism') are excluded."""
    evidence = [
        _make_snippet("temperature", "1200 K", "at 1200 K", 0.95),
        _make_snippet("mechanism", "GRI-3.0", "using GRI-Mech 3.0", 0.99),
    ]

    agent = ConditionExtractionAgent(llm_client)
    await agent.extract(_make_paper(), evidence=evidence)

    prompt = llm_client.complete.call_args[1]["prompt"]
    assert "[TEMPERATURE]" in prompt
    assert "[MECHANISM]" not in prompt
