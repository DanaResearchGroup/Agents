"""Mean Absolute Error calculation for model evaluation."""

from __future__ import annotations


def compute_mae(predicted: list[float], observed: list[float]) -> float:
    """Compute Mean Absolute Error between predicted and observed values.

    Raises ValueError if lists are empty or different lengths.
    """
    if len(predicted) != len(observed):
        raise ValueError(
            f"Length mismatch: predicted={len(predicted)}, observed={len(observed)}"
        )
    if not predicted:
        raise ValueError("Cannot compute MAE on empty lists")

    return sum(abs(p - o) for p, o in zip(predicted, observed)) / len(predicted)


def compute_relative_mae(predicted: list[float], observed: list[float]) -> float:
    """Compute relative MAE: mean(|pred - obs| / |obs|).

    Skips data points where observed == 0.
    Raises ValueError if no valid points remain.
    """
    if len(predicted) != len(observed):
        raise ValueError(
            f"Length mismatch: predicted={len(predicted)}, observed={len(observed)}"
        )

    errors = []
    for p, o in zip(predicted, observed):
        if o != 0.0:
            errors.append(abs(p - o) / abs(o))

    if not errors:
        raise ValueError("No valid data points (all observed values are zero)")

    return sum(errors) / len(errors)
