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


# ── extract_rates tests ──────────────────────────────────────────────────────


def test_extract_rates_standard_lines(converter: ChemkinConverter, tmp_path):
    """Parse standard Chemkin rate lines into normalised dict."""
    chemkin = tmp_path / "mech.inp"
    chemkin.write_text(
        "! mechanism\n"
        "H+O2<=>O+OH   1.04E+14  0.0  15310.0\n"
        "OH+H2<=>H2O+H   2.14E+08  1.52  3449.0\n"
    )
    rates = converter.extract_rates(chemkin)
    assert len(rates) == 2
    # Check normalised keys
    assert "h + o2 <=> o + oh" in rates
    assert "h2 + oh <=> h + h2o" in rates
    # Check values
    r = rates["h + o2 <=> o + oh"]
    assert r["A"] == pytest.approx(1.04e14)
    assert r["n"] == 0.0
    assert r["Ea"] == 15310.0


def test_extract_rates_skips_comments_and_malformed(converter: ChemkinConverter, tmp_path):
    """Comment lines (!) and malformed lines are skipped."""
    chemkin = tmp_path / "mech.inp"
    chemkin.write_text(
        "! This is a comment\n"
        "ELEMENTS H O N END\n"
        "SPECIES H O2 OH END\n"
        "REACTIONS\n"
        "H+O2<=>O+OH   1.04E+14  0.0  15310.0\n"
        "this is not a valid reaction line\n"
        "END\n"
    )
    rates = converter.extract_rates(chemkin)
    assert len(rates) == 1
    assert "h + o2 <=> o + oh" in rates


def test_extract_rates_empty_file(converter: ChemkinConverter, tmp_path):
    """Empty file or no parseable reactions returns empty dict."""
    chemkin = tmp_path / "empty.inp"
    chemkin.write_text("! nothing here\nELEMENTS H O END\n")
    rates = converter.extract_rates(chemkin)
    assert rates == {}


def test_extract_rates_plain_float_a(converter: ChemkinConverter, tmp_path):
    """A values in plain float format (not scientific notation) are parsed."""
    chemkin = tmp_path / "mech.inp"
    chemkin.write_text(
        "H+O2<=>O+OH   104000.0  0.0  15310.0\n"
    )
    rates = converter.extract_rates(chemkin)
    assert len(rates) == 1
    r = rates["h + o2 <=> o + oh"]
    assert r["A"] == pytest.approx(104000.0)
    assert r["n"] == 0.0
    assert r["Ea"] == 15310.0


def test_extract_rates_strips_annotations(converter: ChemkinConverter, tmp_path):
    """Trailing DUPLICATE, LOW, TROE, PLOG annotations are stripped."""
    chemkin = tmp_path / "mech.inp"
    chemkin.write_text(
        "H+O2<=>O+OH   1.04E+14  0.0  15310.0  DUPLICATE\n"
        "OH+H2<=>H2O+H   2.14E+08  1.52  3449.0  LOW\n"
        "NH2+NO<=>N2+H2O   2.60E+19  -2.37  0.0  TROE\n"
        "H+O2(+M)<=>HO2(+M)   4.65E+12  0.44  0.0  PLOG\n"
    )
    rates = converter.extract_rates(chemkin)
    assert len(rates) == 4
    assert rates["h + o2 <=> o + oh"]["A"] == pytest.approx(1.04e14)
    assert rates["h2 + oh <=> h + h2o"]["A"] == pytest.approx(2.14e8)
    assert rates["nh2 + no <=> h2o + n2"]["Ea"] == 0.0
