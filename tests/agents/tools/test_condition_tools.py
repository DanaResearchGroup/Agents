"""Tests for condition extraction tools."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from src.agents.tools.condition_tools import (
    ConditionDeps,
    get_evidence,
    get_executable_plans,
    get_paper_summary,
    validate_condition,
)
from src.schemas.experimental import (
    EvidenceSnippet,
    NormalizedComposition,
    NormalizedCondition,
    ObservableType,
    PageText,
    PaperDocument,
    PaperSummary,
    ReactorType,
    SimulationPlan,
)


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture()
def paper() -> PaperDocument:
    return PaperDocument(
        pdf_path="/tmp/paper.pdf",
        title="Test Paper",
        abstract="Study of NH3 oxidation in shock tube at 1200-2000 K.",
        pages=[
            PageText(page_num=1, text="Temperature range: 1200-2000 K, pressure: 1 atm"),
        ],
        captions=[],
    )


@pytest.fixture()
def deps(paper: PaperDocument) -> ConditionDeps:
    return ConditionDeps(
        paper=paper,
        model_species=["NH3(1)", "O2(6)", "Ar"],
    )


@pytest.fixture()
def deps_with_summary(paper: PaperDocument) -> ConditionDeps:
    return ConditionDeps(
        paper=paper,
        model_species=["NH3(1)", "O2(6)", "Ar"],
        paper_summary=PaperSummary(
            reactor_types=["shock_tube"],
            species_studied=["NH3", "O2"],
            temperature_range="1200-2000 K",
            pressure_range="1-10 atm",
            observable_types=["ignition_delay"],
            experimental_setup="Reflected shock tube",
        ),
    )


# ── get_evidence ────────────────────────────────────────────────────────────


def test_get_evidence_returns_formatted_snippets(deps: ConditionDeps):
    """get_evidence returns formatted lines for known evidence kind."""
    fake_evidence = [
        EvidenceSnippet(
            kind="temperature",
            value_text="1200 K",
            page_num=1,
            source_text="Experiments at 1200 K",
            confidence=0.9,
            normalized_value="1200",
        ),
        EvidenceSnippet(
            kind="temperature",
            value_text="2000 K",
            page_num=2,
            source_text="Up to 2000 K",
            confidence=0.7,
        ),
    ]

    with patch(
        "src.agents.tools.condition_tools.extract_evidence",
        return_value=fake_evidence,
    ):
        result = get_evidence(deps, "temperature")

    assert "Page 1 (conf=0.90): 1200" in result
    assert "Page 2 (conf=0.70): 2000 K" in result
    # Higher confidence first
    lines = result.strip().split("\n")
    assert lines[0].startswith("Page 1")


def test_get_evidence_no_matches(deps: ConditionDeps):
    """get_evidence returns message when no snippets of given kind."""
    with patch(
        "src.agents.tools.condition_tools.extract_evidence",
        return_value=[],
    ):
        result = get_evidence(deps, "residence_time")

    assert "No residence_time evidence" in result


def test_get_evidence_caches_extraction(deps: ConditionDeps):
    """Evidence is extracted once, then cached on deps."""
    fake_evidence = [
        EvidenceSnippet(
            kind="temperature",
            value_text="1200 K",
            page_num=1,
            source_text="T=1200K",
            confidence=0.9,
        ),
    ]

    with patch(
        "src.agents.tools.condition_tools.extract_evidence",
        return_value=fake_evidence,
    ) as mock_extract:
        get_evidence(deps, "temperature")
        get_evidence(deps, "pressure")

    # extract_evidence called only once
    mock_extract.assert_called_once()


# ── get_executable_plans ────────────────────────────────────────────────────


def _make_plan() -> SimulationPlan:
    return SimulationPlan(
        scenario_id="scenario_1",
        experiment_family="shock_tube",
        template_family="idt_const_uv",
        plan_status="executable",
        temperature=NormalizedCondition(
            raw_text="1200 K",
            kind="temperature",
            role="setpoint",
            is_range=False,
            min_value=1200.0,
            max_value=1200.0,
            unit="K",
            source_page=1,
        ),
        pressure=NormalizedCondition(
            raw_text="1 atm",
            kind="pressure",
            role="setpoint",
            is_range=False,
            min_value=1.0,
            max_value=1.0,
            unit="atm",
            source_page=1,
        ),
        composition=NormalizedComposition(
            species={"NH3": 0.01, "Ar": 0.99},
            balance_species=None,
            composition_complete=True,
            raw_mentions=["1% NH3 in Ar"],
            source_pages=[1],
        ),
        target_observables=["ignition_delay"],
        auto_generation_allowed=True,
    )


def test_get_executable_plans_formatted(deps: ConditionDeps):
    """Plans are formatted with T, P, composition, observable."""
    plans = [_make_plan()]

    with patch(
        "src.agents.tools.condition_tools.PaperExtractionPipeline",
        return_value=MagicMock(extract=MagicMock(return_value=plans)),
    ):
        result = get_executable_plans(deps)

    assert "Plan 1: shock_tube" in result
    assert "T=1200.0K" in result
    assert "P=1.0atm" in result
    assert "NH3=0.01" in result
    assert "observable=ignition_delay" in result


def test_get_executable_plans_empty(deps: ConditionDeps):
    """No plans returns a message."""
    with patch(
        "src.agents.tools.condition_tools.PaperExtractionPipeline",
        return_value=MagicMock(extract=MagicMock(return_value=[])),
    ):
        result = get_executable_plans(deps)

    assert result == "No executable simulation plans found."


# ── validate_condition ──────────────────────────────────────────────────────


def test_validate_condition_valid(deps: ConditionDeps):
    """Known species pass validation."""
    cond = {
        "reactor_type": "shock_tube",
        "T": 1200.0,
        "P": 1.0,
        "X": {"NH3(1)": 0.01, "Ar": 0.99},
        "observable_type": "idt",
    }
    result = validate_condition(deps, json.dumps(cond))
    assert result == "VALID"


def test_validate_condition_unknown_species(deps: ConditionDeps):
    """Unknown species returns error."""
    cond = {
        "reactor_type": "shock_tube",
        "T": 1200.0,
        "P": 1.0,
        "X": {"NH3": 0.01, "He": 0.99},
        "observable_type": "idt",
    }
    result = validate_condition(deps, json.dumps(cond))
    assert "Unknown species" in result
    assert "NH3" in result
    assert "He" in result


def test_validate_condition_bad_json(deps: ConditionDeps):
    """Invalid JSON returns error."""
    result = validate_condition(deps, "not json{")
    assert "Invalid JSON" in result


def test_validate_condition_bad_schema(deps: ConditionDeps):
    """Missing required fields returns schema error."""
    result = validate_condition(deps, json.dumps({"T": 1200}))
    assert "Schema validation error" in result


def test_validate_condition_bad_mole_fractions(deps: ConditionDeps):
    """Mole fractions outside ±0.01 of 1.0 returns error."""
    cond = {
        "reactor_type": "shock_tube",
        "T": 1200.0,
        "P": 1.0,
        "X": {"NH3(1)": 0.5, "Ar": 0.48},  # sums to 0.98, outside ±0.01
        "observable_type": "idt",
    }
    result = validate_condition(deps, json.dumps(cond))
    assert "sum to" in result
    assert "±0.01" in result


def test_validate_condition_mole_fractions_within_tolerance(deps: ConditionDeps):
    """Mole fractions within ±0.01 of 1.0 pass validation."""
    cond = {
        "reactor_type": "shock_tube",
        "T": 1200.0,
        "P": 1.0,
        "X": {"NH3(1)": 0.5, "Ar": 0.505},  # sums to 1.005, within ±0.01
        "observable_type": "idt",
    }
    result = validate_condition(deps, json.dumps(cond))
    assert result == "VALID"


def test_validate_condition_no_model_species():
    """Without model_species, unknown species check is skipped."""
    paper = PaperDocument(
        pdf_path="/tmp/p.pdf", title="T", pages=[], captions=[],
    )
    deps = ConditionDeps(paper=paper, model_species=[])
    cond = {
        "reactor_type": "shock_tube",
        "T": 1200.0,
        "P": 1.0,
        "X": {"NH3": 0.01, "Ar": 0.99},
        "observable_type": "idt",
    }
    result = validate_condition(deps, json.dumps(cond))
    assert result == "VALID"


# ── get_paper_summary ───────────────────────────────────────────────────────


def test_get_paper_summary_with_summary(deps_with_summary: ConditionDeps):
    """Returns formatted summary when PaperSummary is provided."""
    result = get_paper_summary(deps_with_summary)
    assert "shock_tube" in result
    assert "1200-2000 K" in result
    assert "ignition_delay" in result


def test_get_paper_summary_falls_back_to_abstract(deps: ConditionDeps):
    """Falls back to abstract when no PaperSummary."""
    result = get_paper_summary(deps)
    assert "NH3 oxidation" in result


def test_get_paper_summary_falls_back_to_first_page():
    """Falls back to first page text when no summary or abstract."""
    paper = PaperDocument(
        pdf_path="/tmp/p.pdf",
        title="No Abstract",
        pages=[PageText(page_num=1, text="Shock tube experiments were conducted")],
        captions=[],
    )
    deps = ConditionDeps(paper=paper)
    result = get_paper_summary(deps)
    assert "Shock tube experiments" in result


def test_get_paper_summary_no_content():
    """Returns message when nothing is available."""
    paper = PaperDocument(
        pdf_path="/tmp/p.pdf",
        title="Empty",
        pages=[],
        captions=[],
    )
    deps = ConditionDeps(paper=paper)
    result = get_paper_summary(deps)
    assert "No paper summary available." in result


# ── Non-empty string guarantees (Ollama nil protection) ─────────────────────


def test_get_evidence_returns_nonempty_on_exception(deps: ConditionDeps):
    """get_evidence returns a non-empty string even if extraction crashes."""
    with patch(
        "src.agents.tools.condition_tools.extract_evidence",
        side_effect=RuntimeError("parser crash"),
    ):
        result = get_evidence(deps, "temperature")

    assert isinstance(result, str)
    assert len(result) > 0
    assert "Tool error" in result


def test_get_executable_plans_returns_nonempty_on_exception(deps: ConditionDeps):
    """get_executable_plans returns a non-empty string even if pipeline crashes."""
    with patch(
        "src.agents.tools.condition_tools.PaperExtractionPipeline",
        side_effect=RuntimeError("pipeline init failed"),
    ):
        result = get_executable_plans(deps)

    assert isinstance(result, str)
    assert len(result) > 0
    assert "Tool error" in result


def test_validate_condition_returns_nonempty_on_exception(deps: ConditionDeps):
    """validate_condition returns a non-empty string on unexpected error."""
    # Pass valid JSON but patch SimConditions to crash
    cond = '{"reactor_type":"shock_tube","T":1200,"P":1,"X":{"NH3(1)":1},"observable_type":"idt"}'
    with patch(
        "src.agents.tools.condition_tools.SimConditions.model_validate",
        side_effect=RuntimeError("unexpected"),
    ):
        result = validate_condition(deps, cond)

    assert isinstance(result, str)
    assert len(result) > 0


def test_get_paper_summary_returns_nonempty_on_exception():
    """get_paper_summary returns a non-empty string on unexpected error."""
    # Create a mock deps that will raise when paper_summary is accessed
    mock_deps = MagicMock()
    mock_deps.paper_summary = MagicMock()
    mock_deps.paper_summary.reactor_types = ["shock_tube"]
    # Make species_studied raise to trigger the except branch
    type(mock_deps.paper_summary).species_studied = PropertyMock(
        side_effect=RuntimeError("boom"),
    )
    result = get_paper_summary(mock_deps)

    assert isinstance(result, str)
    assert len(result) > 0
    assert "Tool error" in result
