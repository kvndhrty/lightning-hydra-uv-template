"""Utilities for metrics processing and extraction."""

from typing import Any, Union

import torch


def extract_callback_metrics(callback_metrics: dict[str, Any]) -> dict[str, float]:
    """Extract and convert metrics from trainer callback metrics.

    Args:
        callback_metrics: Dictionary of callback metrics from trainer.

    Returns:
        Dictionary with metrics converted to float values.
    """
    metrics = {}
    for key, value in callback_metrics.items():
        if isinstance(value, torch.Tensor):
            metrics[key] = value.item()
        else:
            metrics[key] = float(value) if isinstance(value, (int, float)) else value
    return metrics


def compute_gradient_norms(model: torch.nn.Module, per_layer: bool = False) -> dict[str, float]:
    """Compute gradient norms for a model.

    Args:
        model: PyTorch model to compute gradient norms for.
        per_layer: Whether to compute per-layer gradient norms.

    Returns:
        Dictionary with gradient norm values.
    """
    grad_norms = {}

    # Total gradient norm
    total_norm = 0.0
    param_count = 0

    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1

    if param_count > 0:
        grad_norms["total"] = total_norm**0.5
    else:
        grad_norms["total"] = 0.0

    # Per-layer gradient norms
    if per_layer:
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norms[f"layer_{name}"] = param.grad.data.norm(2).item()

    return grad_norms


def get_learning_rates(
    optimizers: Union[torch.optim.Optimizer, list[torch.optim.Optimizer]],
) -> dict[str, float]:
    """Extract learning rates from optimizers.

    Args:
        optimizers: Single optimizer or list of optimizers.

    Returns:
        Dictionary with learning rate values.
    """
    if not isinstance(optimizers, list):
        optimizers = [optimizers]

    lr_dict = {}
    for i, optimizer in enumerate(optimizers):
        for j, param_group in enumerate(optimizer.param_groups):
            if len(optimizers) == 1 and len(optimizer.param_groups) == 1:
                lr_dict["lr"] = param_group["lr"]
            else:
                lr_dict[f"optimizer_{i}_group_{j}_lr"] = param_group["lr"]

    return lr_dict


def aggregate_metrics(
    metrics_list: list[dict[str, float]], strategy: str = "mean"
) -> dict[str, float]:
    """Aggregate metrics across multiple dictionaries.

    Args:
        metrics_list: List of metric dictionaries to aggregate.
        strategy: Aggregation strategy ("mean", "sum", "max", "min").

    Returns:
        Dictionary with aggregated metrics.
    """
    if not metrics_list:
        return {}

    # Get all unique keys
    all_keys = set()
    for metrics in metrics_list:
        all_keys.update(metrics.keys())

    aggregated = {}
    for key in all_keys:
        values = [metrics.get(key, 0.0) for metrics in metrics_list if key in metrics]

        if not values:
            continue

        if strategy == "mean":
            aggregated[key] = sum(values) / len(values)
        elif strategy == "sum":
            aggregated[key] = sum(values)
        elif strategy == "max":
            aggregated[key] = max(values)
        elif strategy == "min":
            aggregated[key] = min(values)
        else:
            raise ValueError(f"Unknown aggregation strategy: {strategy}")

    return aggregated
