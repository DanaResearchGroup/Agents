"""Tests for ChemkinConverter — subprocess and LLM calls mocked."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.agents.llm_client import LLMClient, LLMConfig
from src.agents.conversion import ChemkinConverter
from src.schemas.experimental import ConversionResult


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture()
def client() -> LLMClient:
    return LLMClient(config=LLMConfig())


@pytest.fixture()
def converter(client: LLMClient) -> ChemkinConverter:
    return ChemkinConverter(llm_client=client)


# ── Success path ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_convert_success(converter: ChemkinConverter, monkeypatch, tmp_path):
    """T3 succeeds on first attempt."""
    chemkin = tmp_path / "mech.inp"
    chemkin.write_text("ELEMENTS H O END")
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    expected_output = output_dir / "mech.yaml"

    async def mock_run_t3(self, chemkin_path, output_path):
        output_path.write_text("phases:\n- name: gas\n")
        return (0, "", "")

    monkeypatch.setattr(ChemkinConverter, "_run_t3", mock_run_t3)

    result = await converter.convert(chemkin, output_dir)

    assert isinstance(result, ConversionResult)
    assert result.success is True
    assert result.output_path == expected_output
    assert result.attempts == 1
    assert result.errors == []


@pytest.mark.asyncio
async def test_convert_success_captures_stdout_warnings(
    converter: ChemkinConverter, monkeypatch, tmp_path
):
    """Stdout from T3 is captured as warnings even on success."""
    chemkin = tmp_path / "mech.inp"
    chemkin.write_text("ELEMENTS H O END")
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    async def mock_run_t3(self, chemkin_path, output_path):
        output_path.write_text("phases:\n- name: gas\n")
        return (0, "Warning: duplicate species ignored", "")

    monkeypatch.setattr(ChemkinConverter, "_run_t3", mock_run_t3)

    result = await converter.convert(chemkin, output_dir)

    assert result.success is True
    assert len(result.warnings) == 1
    assert "duplicate species" in result.warnings[0]


# ── Retry path ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_convert_retry_succeeds_on_second_attempt(
    converter: ChemkinConverter, monkeypatch, tmp_path
):
    """First T3 call fails, LLM diagnoses, second call succeeds."""
    chemkin = tmp_path / "mech.inp"
    chemkin.write_text("ELEMENTS H O END")
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    call_count = 0

    async def mock_run_t3(self, chemkin_path, output_path):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return (1, "", "Error: malformed thermo block")
        output_path.write_text("phases:\n- name: gas\n")
        return (0, "", "")

    async def mock_complete(*, prompt, system, agent_name, **kwargs):
        assert agent_name == "conversion"
        return "The thermo block has an incorrect temperature range."

    monkeypatch.setattr(ChemkinConverter, "_run_t3", mock_run_t3)
    monkeypatch.setattr(converter.llm_client, "complete", mock_complete)

    result = await converter.convert(chemkin, output_dir)

    assert result.success is True
    assert result.attempts == 2
    assert len(result.errors) == 1
    assert "malformed thermo block" in result.errors[0]
    assert any("LLM diagnosis" in w for w in result.warnings)


# ── Both attempts fail ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_convert_both_attempts_fail(
    converter: ChemkinConverter, monkeypatch, tmp_path
):
    """Both T3 calls fail — returns ConversionResult(success=False)."""
    chemkin = tmp_path / "mech.inp"
    chemkin.write_text("ELEMENTS H O END")
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    async def mock_run_t3(self, chemkin_path, output_path):
        return (1, "", "Error: fatal parse failure")

    async def mock_complete(*, prompt, system, agent_name, **kwargs):
        return "Unknown format issue."

    monkeypatch.setattr(ChemkinConverter, "_run_t3", mock_run_t3)
    monkeypatch.setattr(converter.llm_client, "complete", mock_complete)

    result = await converter.convert(chemkin, output_dir)

    assert result.success is False
    assert result.output_path is None
    assert result.attempts == 2
    assert len(result.errors) == 2


# ── Max attempts never exceeded ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_max_attempts_never_exceeded(
    converter: ChemkinConverter, monkeypatch, tmp_path
):
    """T3 is never called more than MAX_ATTEMPTS times."""
    chemkin = tmp_path / "mech.inp"
    chemkin.write_text("ELEMENTS H O END")
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    t3_calls = 0

    async def mock_run_t3(self, chemkin_path, output_path):
        nonlocal t3_calls
        t3_calls += 1
        return (1, "", "fail")

    async def mock_complete(*, prompt, system, agent_name, **kwargs):
        return "diagnosis"

    monkeypatch.setattr(ChemkinConverter, "_run_t3", mock_run_t3)
    monkeypatch.setattr(converter.llm_client, "complete", mock_complete)

    result = await converter.convert(chemkin, output_dir)

    assert t3_calls == 2
    assert result.attempts == 2
    assert result.success is False
