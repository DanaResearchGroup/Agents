"""Tests for src.agents.utils."""

from src.agents.utils import strip_thinking


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
