"""Tests for src.agents.utils."""

from src.agents.utils import extract_json_object, strip_thinking


def test_strips_think_block():
    text = "<think>Let me analyze this...</think>\n[{\"T\": 1200}]"
    assert strip_thinking(text) == '[{"T": 1200}]'


def test_strips_multiple_think_blocks():
    text = (
        "<think>First thought</think>\nSome text\n"
        "<think>Second thought</think>\nMore text"
    )
    result = strip_thinking(text)
    assert "<think>" not in result
    assert "Some text" in result
    assert "More text" in result


def test_strips_think_block_with_newline_after():
    text = "<think>reasoning here\nwith multiple lines</think>\n[{\"T\": 1200}]"
    assert strip_thinking(text) == '[{"T": 1200}]'


def test_returns_original_when_no_think_tags():
    text = '[{"reactor_type": "shock_tube", "T": 1200}]'
    assert strip_thinking(text) == text


def test_returns_original_when_cleaned_is_empty():
    text = "<think>only thinking, no actual content</think>"
    assert strip_thinking(text) == text


# ── extract_json_object ──────────────────────────────────────────────────────


def test_extract_json_object_clean():
    text = '{"reactor_types": ["shock_tube"], "temperature_range": "1200 K"}'
    assert extract_json_object(text) == text


def test_extract_json_object_from_markdown():
    text = '```json\n{"reactor_types": ["jsr"]}\n```'
    assert extract_json_object(text) == '{"reactor_types": ["jsr"]}'


def test_extract_json_object_from_preamble():
    text = 'Here is the summary:\n{"reactor_types": ["pfr"], "species_studied": ["NH3"]}'
    result = extract_json_object(text)
    assert result.startswith("{")
    assert '"pfr"' in result


def test_extract_json_object_returns_empty_when_no_object():
    text = "## Study Overview\nThis paper studies NH3 combustion."
    assert extract_json_object(text) == "{}"
