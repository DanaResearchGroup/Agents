"""Tests for ModelBranchingAgent — all Cantera calls mocked via conftest."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from src.agents.model_branching import ModelBranchingAgent
from src.schemas.experimental import CandidateReaction, ModelBranch


# ── Fixtures ─────────────────────────────────────────────────────────────────


def _make_model_yaml(path: Path, reactions: list[dict] | None = None) -> Path:
    """Write a minimal Cantera-style YAML mechanism file."""
    data = {
        "phases": [{"name": "gas", "species": ["H2", "O2", "H2O"]}],
        "reactions": reactions or [
            {"equation": "H2 + O2 => H2O", "rate-constant": {"A": 1e10, "b": 0, "Ea": 5000}},
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f)
    return path


@pytest.fixture()
def original_model(tmp_path: Path) -> Path:
    return _make_model_yaml(tmp_path / "original.yaml")


@pytest.fixture()
def output_dir(tmp_path: Path) -> Path:
    d = tmp_path / "branches"
    d.mkdir()
    return d


def _rxn(reaction_string: str, rate_params: dict | None = None, confidence: float = 0.8) -> CandidateReaction:
    return CandidateReaction(
        reaction_string=reaction_string,
        rate_params=rate_params,
        source_section="test",
        justification="test",
        confidence=confidence,
    )


# ── Tests ────────────────────────────────────────────────────────────────────


def test_create_branch_happy_path(original_model, output_dir):
    """Branch with valid rate_params creates a YAML file."""
    with patch("src.agents.model_branching.ModelIsolationValidator") as mock_v:
        mock_v.return_value.validate_path2_branch = MagicMock(return_value=None)
        agent = ModelBranchingAgent(original_model, output_dir)
        rxn = _rxn("OH + H2 => H2O + H", rate_params={"rate-constant": {"A": 2e10, "b": 0, "Ea": 3000}})
        branch = agent.create_branch([rxn], "branch_001")

    assert isinstance(branch, ModelBranch)
    assert branch.branch_id == "branch_001"
    assert branch.model_path.exists()
    assert len(branch.reactions_added) == 1

    # Verify the YAML contains the added reaction
    with open(branch.model_path) as f:
        data = yaml.safe_load(f)
    assert len(data["reactions"]) == 2  # original + new
    assert data["reactions"][-1]["equation"] == "OH + H2 => H2O + H"


def test_skip_reactions_without_rate_params(original_model, output_dir):
    """Reactions without rate_params are skipped with a warning."""
    rxn_no_params = _rxn("A => B", rate_params=None)
    rxn_with_params = _rxn("C => D", rate_params={"rate-constant": {"A": 1e8, "b": 0, "Ea": 1000}})

    with patch("src.agents.model_branching.ModelIsolationValidator") as mock_v:
        mock_v.return_value.validate_path2_branch = MagicMock(return_value=None)
        agent = ModelBranchingAgent(original_model, output_dir)
        branch = agent.create_branch([rxn_no_params, rxn_with_params], "branch_002")

    assert len(branch.reactions_added) == 1
    assert branch.reactions_added[0].reaction_string == "C => D"


def test_all_reactions_missing_rate_params_raises(original_model, output_dir):
    """ValueError when no reactions have rate_params."""
    rxn = _rxn("A => B", rate_params=None)

    with patch("src.agents.model_branching.ModelIsolationValidator") as mock_v:
        mock_v.return_value.validate_path2_branch = MagicMock(return_value=None)
        agent = ModelBranchingAgent(original_model, output_dir)
        with pytest.raises(ValueError, match="no reactions had rate_params"):
            agent.create_branch([rxn], "branch_003")


def test_branch_validation_failure_propagates(original_model, output_dir):
    """ValueError from validate_path2_branch must propagate."""
    rxn = _rxn("X => Y", rate_params={"rate-constant": {"A": 1e6, "b": 0, "Ea": 500}})

    with patch("src.agents.model_branching.ModelIsolationValidator") as mock_v:
        mock_v.return_value.validate_path2_branch = MagicMock(
            side_effect=ValueError("Branch is missing reactions"),
        )
        agent = ModelBranchingAgent(original_model, output_dir)
        with pytest.raises(ValueError, match="Branch is missing reactions"):
            agent.create_branch([rxn], "branch_004")


def test_output_dir_created_if_missing(original_model, tmp_path):
    """Output directory is created automatically if it doesn't exist."""
    new_dir = tmp_path / "deep" / "nested" / "branches"
    rxn = _rxn("E => F", rate_params={"rate-constant": {"A": 1e9, "b": 0, "Ea": 2000}})

    with patch("src.agents.model_branching.ModelIsolationValidator") as mock_v:
        mock_v.return_value.validate_path2_branch = MagicMock(return_value=None)
        agent = ModelBranchingAgent(original_model, new_dir)
        branch = agent.create_branch([rxn], "branch_005")

    assert new_dir.exists()
    assert branch.model_path.exists()
