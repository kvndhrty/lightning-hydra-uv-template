"""Utilities for creating and formatting Rich tables."""

from typing import Any, Optional, Union

from rich.table import Table


def create_checkpoint_table(title: str = "Checkpoints") -> Table:
    """Create standardized checkpoint table.

    Args:
        title: Title for the table.

    Returns:
        Configured Rich Table for checkpoints.
    """
    table = Table(title=title)
    table.add_column("#", style="cyan", width=3)
    table.add_column("Checkpoint", style="green")
    table.add_column("Epoch", style="yellow")
    table.add_column("Created", style="blue")
    table.add_column("Metrics", style="magenta")
    return table


def create_metrics_table(title: str = "Metrics") -> Table:
    """Create standardized metrics table.

    Args:
        title: Title for the table.

    Returns:
        Configured Rich Table for metrics.
    """
    table = Table(title=title)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Epoch", style="yellow")
    return table


def create_config_table(title: str = "Configuration") -> Table:
    """Create standardized configuration table.

    Args:
        title: Title for the table.

    Returns:
        Configured Rich Table for configuration.
    """
    table = Table(title=title)
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Type", style="yellow")
    return table


def add_checkpoint_row(
    table: Table,
    index: int,
    checkpoint_name: str,
    epoch: Optional[int] = None,
    created: Optional[str] = None,
    metrics: Optional[dict[str, Any]] = None,
) -> None:
    """Add a row to a checkpoint table.

    Args:
        table: Rich Table to add row to.
        index: Row index number.
        checkpoint_name: Name of the checkpoint file.
        epoch: Training epoch number.
        created: Creation timestamp string.
        metrics: Dictionary of metrics to display.
    """
    epoch_str = str(epoch) if epoch is not None else "N/A"
    created_str = created if created is not None else "N/A"

    if metrics:
        # Format key metrics for display
        metrics_str = ", ".join(
            [
                f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                for k, v in list(metrics.items())[:3]  # Show first 3 metrics
            ]
        )
        if len(metrics) > 3:
            metrics_str += "..."
    else:
        metrics_str = "N/A"

    table.add_row(str(index), checkpoint_name, epoch_str, created_str, metrics_str)


def add_metrics_row(
    table: Table, metric_name: str, value: Union[float, int, str], epoch: Optional[int] = None
) -> None:
    """Add a row to a metrics table.

    Args:
        table: Rich Table to add row to.
        metric_name: Name of the metric.
        value: Metric value.
        epoch: Epoch when metric was recorded.
    """
    value_str = f"{value:.6f}" if isinstance(value, float) else str(value)

    epoch_str = str(epoch) if epoch is not None else "N/A"

    table.add_row(metric_name, value_str, epoch_str)


def add_config_row(
    table: Table, param_name: str, value: Any, param_type: Optional[str] = None
) -> None:
    """Add a row to a configuration table.

    Args:
        table: Rich Table to add row to.
        param_name: Name of the parameter.
        value: Parameter value.
        param_type: Type of the parameter.
    """
    value_str = str(value)
    if len(value_str) > 50:
        value_str = value_str[:47] + "..."

    type_str = param_type if param_type is not None else type(value).__name__

    table.add_row(param_name, value_str, type_str)


def format_metrics_for_display(metrics: dict[str, Any], max_metrics: int = 5) -> str:
    """Format metrics dictionary for compact display.

    Args:
        metrics: Dictionary of metrics.
        max_metrics: Maximum number of metrics to display.

    Returns:
        Formatted string representation of metrics.
    """
    if not metrics:
        return "No metrics"

    formatted_items = []
    for i, (key, value) in enumerate(metrics.items()):
        if i >= max_metrics:
            formatted_items.append("...")
            break

        if isinstance(value, float):
            formatted_items.append(f"{key}: {value:.4f}")
        else:
            formatted_items.append(f"{key}: {value}")

    return ", ".join(formatted_items)


def create_comparison_table(
    data: list[dict[str, Any]], columns: list[tuple[str, str]], title: str = "Comparison"
) -> Table:
    """Create a table for comparing multiple items.

    Args:
        data: List of dictionaries containing data to compare.
        columns: List of (column_name, dict_key) tuples defining table structure.
        title: Title for the table.

    Returns:
        Configured Rich Table with comparison data.
    """
    table = Table(title=title)

    # Add columns
    for col_name, _ in columns:
        table.add_column(col_name, style="cyan")

    # Add rows
    for item in data:
        row_values = []
        for _, dict_key in columns:
            value = item.get(dict_key, "N/A")
            if isinstance(value, float):
                row_values.append(f"{value:.4f}")
            else:
                row_values.append(str(value))
        table.add_row(*row_values)

    return table
