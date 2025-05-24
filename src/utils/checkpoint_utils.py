"""Utilities for checkpoint management and resuming."""
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from omegaconf import OmegaConf


class CheckpointInfo:
    """Container for checkpoint information."""

    def __init__(
        self,
        path: Path,
        run_dir: Path,
        epoch: Optional[int] = None,
        metric_value: Optional[float] = None,
        metric_name: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        config_path: Optional[Path] = None,
        overrides_path: Optional[Path] = None,
        metrics: Optional[dict[str, float]] = None,
    ):
        self.path = path
        self.run_dir = run_dir
        self.epoch = epoch
        self.metric_value = metric_value
        self.metric_name = metric_name
        self.timestamp = timestamp
        self.config_path = config_path
        self.overrides_path = overrides_path
        self.metrics = metrics or {}

    @property
    def name(self) -> str:
        """Short name for display."""
        return f"{self.run_dir.name}/{self.path.name}"

    @property
    def has_config(self) -> bool:
        """Check if config files exist."""
        return self.config_path and self.config_path.exists()


def scan_checkpoints(
    logs_dir: Path, task_name: str = "train", pattern: str = "*.ckpt"
) -> list[CheckpointInfo]:
    """Scan logs directory for checkpoints.

    Args:
        logs_dir: Root logs directory
        task_name: Task name (train, eval, etc.)
        pattern: Glob pattern for checkpoint files

    Returns:
        List of CheckpointInfo objects sorted by timestamp (newest first)
    """
    checkpoints = []

    # Search in both runs and multiruns
    for runs_type in ["runs", "multiruns"]:
        runs_dir = logs_dir / task_name / runs_type
        if not runs_dir.exists():
            continue

        # Find all checkpoint files
        for ckpt_path in runs_dir.rglob(pattern):
            run_dir = ckpt_path.parent.parent  # Go up from checkpoints/ to run dir

            # Parse timestamp from run directory name
            timestamp = None
            timestamp_match = re.search(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})", run_dir.name)
            if timestamp_match:
                timestamp = datetime.strptime(timestamp_match.group(1), "%Y-%m-%d_%H-%M-%S")

            # Parse epoch from filename
            epoch = None
            epoch_match = re.search(r"epoch_(\d+)", ckpt_path.stem)
            if epoch_match:
                epoch = int(epoch_match.group(1))

            # Find config files
            hydra_dir = run_dir / ".hydra"
            config_path = hydra_dir / "config.yaml" if hydra_dir.exists() else None
            overrides_path = hydra_dir / "overrides.yaml" if hydra_dir.exists() else None

            # Load metrics from JSON if available
            metrics = load_checkpoint_metrics(ckpt_path)

            # Extract key metric value if available
            metric_value = None
            metric_name = None
            if metrics:
                # Try common metric names
                for name in ["val/acc", "val/accuracy", "val_acc", "test/acc", "test/accuracy"]:
                    if name in metrics:
                        metric_value = metrics[name]
                        metric_name = name
                        break

            checkpoints.append(
                CheckpointInfo(
                    path=ckpt_path,
                    run_dir=run_dir,
                    epoch=epoch,
                    timestamp=timestamp,
                    config_path=config_path,
                    overrides_path=overrides_path,
                    metric_value=metric_value,
                    metric_name=metric_name,
                    metrics=metrics,
                )
            )

    # Sort by timestamp (newest first)
    checkpoints.sort(key=lambda x: x.timestamp or datetime.min, reverse=True)
    return checkpoints


def load_checkpoint_metrics(checkpoint_path: Path) -> dict[str, float]:
    """Load metrics associated with a checkpoint.

    Looks for metrics in:
    1. JSON file with same name as checkpoint
    2. metrics.json in the run's metrics directory
    3. CSV file in the run's csv directory

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Dictionary of metrics, empty if none found
    """
    metrics = {}

    # Try JSON file next to checkpoint
    json_path = checkpoint_path.with_suffix(".json")
    if json_path.exists():
        try:
            with open(json_path) as f:
                data = json.load(f)
                if "metrics" in data:
                    metrics.update(data["metrics"])
                else:
                    metrics.update(data)
        except Exception:
            pass

    # Try metrics directory
    run_dir = checkpoint_path.parent.parent
    metrics_dir = run_dir / "metrics"
    if metrics_dir.exists():
        # Try metrics.json
        metrics_json = metrics_dir / "metrics.json"
        if metrics_json.exists():
            try:
                with open(metrics_json) as f:
                    all_metrics = json.load(f)
                    # Get the last epoch's metrics
                    if isinstance(all_metrics, list) and all_metrics:
                        metrics.update(all_metrics[-1])
            except Exception:
                pass

        # Try metrics.csv
        metrics_csv = metrics_dir / "metrics.csv"
        if not metrics and metrics_csv.exists():
            try:
                df = pd.read_csv(metrics_csv)
                if not df.empty:
                    # Get last row as dict
                    last_row = df.iloc[-1].to_dict()
                    metrics.update(last_row)
            except Exception:
                pass

    # Try CSV logger output
    csv_dir = run_dir / "csv" / "version_0"
    if not metrics and csv_dir.exists():
        metrics_csv = csv_dir / "metrics.csv"
        if metrics_csv.exists():
            try:
                df = pd.read_csv(metrics_csv)
                if not df.empty:
                    # Get last row
                    last_row = df.iloc[-1].to_dict()
                    metrics.update(last_row)
            except Exception:
                pass

    return metrics


def get_latest_checkpoint(logs_dir: Path, task_name: str = "train") -> Optional[CheckpointInfo]:
    """Get the most recent checkpoint."""
    checkpoints = scan_checkpoints(logs_dir, task_name)
    return checkpoints[0] if checkpoints else None


def get_best_checkpoint(
    logs_dir: Path, task_name: str = "train", metric_name: str = "val/acc", mode: str = "max"
) -> Optional[CheckpointInfo]:
    """Get the best checkpoint based on metric.

    Note: This requires parsing CSV logs or checkpoint metadata,
    which is not implemented in this basic version.
    """
    # For now, look for files named "best.ckpt" or similar
    checkpoints = scan_checkpoints(logs_dir, task_name, pattern="*best*.ckpt")
    if checkpoints:
        return checkpoints[0]

    # Fallback to latest
    return get_latest_checkpoint(logs_dir, task_name)


def load_checkpoint_config(checkpoint: CheckpointInfo) -> dict:
    """Load the config associated with a checkpoint."""
    if not checkpoint.config_path or not checkpoint.config_path.exists():
        raise FileNotFoundError(f"Config not found for checkpoint: {checkpoint.path}")

    return OmegaConf.load(checkpoint.config_path)


def load_checkpoint_overrides(checkpoint: CheckpointInfo) -> list[str]:
    """Load the original command line overrides for a checkpoint."""
    if not checkpoint.overrides_path or not checkpoint.overrides_path.exists():
        return []

    overrides = OmegaConf.load(checkpoint.overrides_path)
    return list(overrides) if overrides else []


def get_resume_command(
    checkpoint: CheckpointInfo,
    script_path: str = "src/train.py",
    additional_overrides: Optional[list[str]] = None,
) -> tuple[str, list[str]]:
    """Generate command to resume from checkpoint.

    Returns:
        Tuple of (executable, arguments)
    """
    # Start with base command
    executable = "python"
    args = [script_path]

    # Add original overrides
    original_overrides = load_checkpoint_overrides(checkpoint)
    args.extend(original_overrides)

    # Add checkpoint path (this will override any existing ckpt_path)
    args.append(f"ckpt_path={checkpoint.path}")

    # Add any additional overrides
    if additional_overrides:
        args.extend(additional_overrides)

    return executable, args


def find_checkpoint_by_name(
    logs_dir: Path, name: str, task_name: str = "train"
) -> Optional[CheckpointInfo]:
    """Find a checkpoint by partial name match."""
    checkpoints = scan_checkpoints(logs_dir, task_name)

    # Try exact match first
    for ckpt in checkpoints:
        if name in str(ckpt.path):
            return ckpt

    # Try fuzzy match on run directory name
    for ckpt in checkpoints:
        if name in ckpt.run_dir.name:
            return ckpt

    return None
