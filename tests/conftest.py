"""Shared test fixtures — Cantera mock for tests that must not load real Cantera."""

from __future__ import annotations

import sys
import types
from unittest.mock import patch

import pytest


# ── Fake Cantera objects ─────────────────────────────────────────────────────


class _FakeReaction:
    """Minimal stand-in for cantera.Reaction with an .equation attribute."""

    def __init__(self, equation: str) -> None:
        self.equation = equation


class _FakeSolution:
    """Minimal stand-in for cantera.Solution."""

    _registry: dict[str, list[_FakeReaction]] = {}

    def __init__(self, path: str) -> None:
        if path not in self._registry:
            raise Exception(f"Cannot load: {path}")
        self._reactions = self._registry[path]

    def reactions(self) -> list[_FakeReaction]:
        return self._reactions


def register_mechanism(path: str, equations: list[str]) -> None:
    """Register a fake mechanism file so _FakeSolution can 'load' it."""
    _FakeSolution._registry[path] = [_FakeReaction(eq) for eq in equations]


def clear_registry() -> None:
    _FakeSolution._registry.clear()


def build_fake_cantera() -> types.ModuleType:
    """Build a fake cantera module with Solution = _FakeSolution."""
    fake = types.ModuleType("cantera")
    fake.Solution = _FakeSolution  # type: ignore[attr-defined]
    return fake


# ── Fixture ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def mock_cantera():
    """Patch sys.modules so ``import cantera`` resolves to a fake.

    Use this fixture in any test that must not load real Cantera.
    The patch is scoped per-test and cleaned up automatically.
    """
    fake = build_fake_cantera()
    with patch.dict(sys.modules, {"cantera": fake}):
        clear_registry()
        yield fake
        clear_registry()
