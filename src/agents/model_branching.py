"""Model branching agent: creates modified model copies with added reactions."""

import logging
from pathlib import Path

import yaml

from src.agents.validators import ModelIsolationValidator
from src.schemas.experimental import CandidateReaction, ModelBranch

logger = logging.getLogger(__name__)


class ModelBranchingAgent:
    """Creates model branches by appending candidate reactions to the original.

    Pure file manipulation — no LLM calls.
    """

    def __init__(self, original_model: Path, output_dir: Path) -> None:
        self.original_model = original_model
        self.output_dir = output_dir

    def create_branch(
        self,
        reactions: list[CandidateReaction],
        branch_id: str,
    ) -> ModelBranch:
        """Create a branch model by appending reactions to the original.

        Args:
            reactions: Candidate reactions to add.
            branch_id: Unique identifier for this branch.

        Returns:
            ModelBranch with the path to the new model file.

        Raises:
            ValueError: If no reactions have rate_params, or if validation fails.
        """
        # Load original model YAML
        with open(self.original_model) as f:
            model_data = yaml.safe_load(f)

        if "reactions" not in model_data:
            model_data["reactions"] = []

        added: list[CandidateReaction] = []
        for rxn in reactions:
            if rxn.rate_params is None:
                logger.warning(
                    "Skipping reaction '%s' — no rate_params", rxn.reaction_string,
                )
                continue
            model_data["reactions"].append({
                "equation": rxn.reaction_string,
                **rxn.rate_params,
            })
            added.append(rxn)

        if not added:
            raise ValueError(
                f"Branch {branch_id}: no reactions had rate_params — nothing to add",
            )

        # Write branch model
        self.output_dir.mkdir(parents=True, exist_ok=True)
        branch_path = self.output_dir / f"{branch_id}.yaml"
        with open(branch_path, "w") as f:
            yaml.dump(model_data, f, default_flow_style=False, sort_keys=False)

        # Validate branch is superset of original
        validator = ModelIsolationValidator()
        validator.validate_path2_branch(branch_path, self.original_model)

        return ModelBranch(
            branch_id=branch_id,
            reactions_added=added,
            model_path=branch_path,
        )
