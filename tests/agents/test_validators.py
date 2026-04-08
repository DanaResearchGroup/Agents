"""Tests for ModelIsolationValidator — cantera.Solution() is mocked."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# ── Mock cantera before importing validators ─────────────────────────────────


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


def _register_mechanism(path: str, equations: list[str]) -> None:
    """Register a fake mechanism file so _FakeSolution can 'load' it."""
    _FakeSolution._registry[path] = [_FakeReaction(eq) for eq in equations]


def _clear_registry() -> None:
    _FakeSolution._registry.clear()


# Insert a fake cantera module into sys.modules BEFORE importing validators
_fake_ct = types.ModuleType("cantera")
_fake_ct.Solution = _FakeSolution  # type: ignore[attr-defined]
sys.modules["cantera"] = _fake_ct

from src.agents.validators import (  # noqa: E402
    ModelIsolationValidator,
    ModelIsolationViolation,
    _normalize_equation,
)


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clean_registry():
    """Clear fake mechanism registry between tests."""
    _clear_registry()
    yield
    _clear_registry()


@pytest.fixture()
def validator() -> ModelIsolationValidator:
    return ModelIsolationValidator()


# ── Normalization unit tests ────────────────────────────────────────────────


def test_normalize_basic():
    assert _normalize_equation("H + O2 <=> OH + O") == "h + o2 <=> o + oh"


def test_normalize_reorders_species():
    # Species on each side should be sorted alphabetically
    assert _normalize_equation("O2 + H <=> O + OH") == "h + o2 <=> o + oh"


def test_normalize_irreversible_arrow():
    # => is normalized to <=>
    assert _normalize_equation("A => B + C") == "a <=> b + c"


def test_normalize_equals_sign():
    # Plain = is normalized to <=>
    assert _normalize_equation("X + Y = Z") == "x + y <=> z"


# ── Path 1: isolation passes ───────────────────────────────────────────────


def test_isolation_passes_no_shared_reactions(validator: ModelIsolationValidator):
    """Two mechanisms with completely different reactions pass validation."""
    _register_mechanism("/original.yaml", [
        "H + O2 <=> OH + O",
        "H2 + OH <=> H2O + H",
    ])
    _register_mechanism("/literature.yaml", [
        "CH4 + O2 <=> CH3 + HO2",
        "C2H6 <=> 2 CH3",
    ])

    # Should not raise
    validator.validate_path1(
        literature_model=Path("/literature.yaml"),
        original_model=Path("/original.yaml"),
    )


# ── Path 1: isolation fails ────────────────────────────────────────────────


def test_isolation_raises_on_overlap(validator: ModelIsolationValidator):
    """Shared reactions trigger ModelIsolationViolation."""
    _register_mechanism("/original.yaml", [
        "H + O2 <=> OH + O",
        "H2 + OH <=> H2O + H",
    ])
    _register_mechanism("/literature.yaml", [
        "CH4 + O2 <=> CH3 + HO2",
        "O2 + H <=> O + OH",  # same as original, different species order
    ])

    with pytest.raises(ModelIsolationViolation) as exc_info:
        validator.validate_path1(
            literature_model=Path("/literature.yaml"),
            original_model=Path("/original.yaml"),
        )
    assert len(exc_info.value.overlapping) == 1
    assert "h + o2 <=> o + oh" in exc_info.value.overlapping


# ── Path 2: branch superset passes ─────────────────────────────────────────


def test_branch_passes_when_superset(validator: ModelIsolationValidator):
    """Branch has all original reactions plus extras — passes."""
    _register_mechanism("/original.yaml", [
        "H + O2 <=> OH + O",
        "H2 + OH <=> H2O + H",
    ])
    _register_mechanism("/branch.yaml", [
        "H + O2 <=> OH + O",
        "H2 + OH <=> H2O + H",
        "CH4 + O2 <=> CH3 + HO2",  # new candidate reaction
    ])

    # Should not raise
    validator.validate_path2_branch(
        branch_model=Path("/branch.yaml"),
        original_model=Path("/original.yaml"),
    )


# ── Path 2: branch missing reactions ───────────────────────────────────────


def test_branch_raises_when_missing_reaction(validator: ModelIsolationValidator):
    """Branch is missing an original reaction — raises ValueError."""
    _register_mechanism("/original.yaml", [
        "H + O2 <=> OH + O",
        "H2 + OH <=> H2O + H",
    ])
    _register_mechanism("/branch.yaml", [
        "H + O2 <=> OH + O",
        # H2 + OH <=> H2O + H is missing
        "CH4 + O2 <=> CH3 + HO2",
    ])

    with pytest.raises(ValueError, match="missing 1 original reaction"):
        validator.validate_path2_branch(
            branch_model=Path("/branch.yaml"),
            original_model=Path("/original.yaml"),
        )


# ── Bad file path ──────────────────────────────────────────────────────────


def test_bad_file_raises_valueerror(validator: ModelIsolationValidator):
    """Unregistered path (simulating bad file) raises ValueError."""
    _register_mechanism("/good.yaml", ["A <=> B"])

    with pytest.raises(ValueError, match="Failed to load mechanism"):
        validator.validate_path1(
            literature_model=Path("/nonexistent.yaml"),
            original_model=Path("/good.yaml"),
        )


# ── reaction_ids returns normalized set ─────────────────────────────────────


def test_reaction_ids_returns_normalized_set(validator: ModelIsolationValidator):
    _register_mechanism("/model.yaml", [
        "H + O2 <=> OH + O",
        "O2 + H <=> O + OH",  # duplicate after normalization
    ])

    ids = validator.reaction_ids(Path("/model.yaml"))
    assert ids == {"h + o2 <=> o + oh"}
