"""Shared utilities for agent modules."""

from __future__ import annotations

import re


def strip_thinking(text: str) -> str:
    """Remove qwen3 ``<think>...</think>`` blocks from model output.

    Some models (notably qwen3) include chain-of-thought reasoning
    inside ``<think>`` tags in the content field when tools are used.
    This strips those blocks so downstream parsing sees only the
    actual response content.

    Returns the original text unchanged if stripping would produce
    an empty string.
    """
    cleaned = re.sub(
        r"<think>.*?</think>\n?",
        "",
        text,
        flags=re.DOTALL,
    ).strip()
    return cleaned if cleaned else text
