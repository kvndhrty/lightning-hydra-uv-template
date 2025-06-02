"""Test utilities and helpers for reducing code duplication."""

import functools
from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import hydra
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from lightning import Trainer
from omegaconf import DictConfig, open_dict


def setup_hydra_config(
    config_name: str = "train", overrides: Optional[list[str]] = None
) -> DictConfig:
    """Setup Hydra configuration for testing.

    Args:
        config_name: Name of the config to load.
        overrides: List of config overrides.

    Returns:
        Loaded configuration.
    """
    if overrides is None:
        overrides = []

    # Clear any existing global Hydra instance
    GlobalHydra.instance().clear()

    # Get config directory
    config_dir = Path(__file__).parent.parent / "configs"

    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)

    return cfg


def configure_trainer_for_test(cfg: DictConfig, **kwargs) -> DictConfig:
    """Configure trainer with test-specific settings.

    Args:
        cfg: Configuration object.
        **kwargs: Trainer parameters to override.

    Returns:
        Modified configuration.
    """
    with open_dict(cfg):
        # Set common test defaults
        cfg.trainer.fast_dev_run = kwargs.get("fast_dev_run", True)
        cfg.trainer.accelerator = kwargs.get("accelerator", "cpu")
        cfg.trainer.devices = kwargs.get("devices", 1)
        cfg.trainer.logger = kwargs.get("logger", False)
        cfg.trainer.enable_checkpointing = kwargs.get("enable_checkpointing", False)

        # Apply any additional overrides
        for key, value in kwargs.items():
            if key not in [
                "fast_dev_run",
                "accelerator",
                "devices",
                "logger",
                "enable_checkpointing",
            ]:
                setattr(cfg.trainer, key, value)

    return cfg


def instantiate_config_components(cfg: DictConfig) -> dict[str, Any]:
    """Instantiate all major config components for testing.

    Args:
        cfg: Configuration object.

    Returns:
        Dictionary with instantiated components.
    """
    return {
        "data": hydra.utils.instantiate(cfg.data),
        "model": hydra.utils.instantiate(cfg.model),
        "trainer": hydra.utils.instantiate(cfg.trainer),
    }


def create_mock_run_structure(
    base_path: Path, run_name: str = "2024-01-20_15-30-45"
) -> tuple[Path, Path, Path]:
    """Create standard mock run directory structure for testing.

    Args:
        base_path: Base path for the run structure.
        run_name: Name of the run directory.

    Returns:
        Tuple of (run_dir, ckpt_dir, hydra_dir) paths.
    """
    run_dir = base_path / "logs" / "train" / "runs" / run_name
    ckpt_dir = run_dir / "checkpoints"
    hydra_dir = run_dir / ".hydra"

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    hydra_dir.mkdir(parents=True, exist_ok=True)

    return run_dir, ckpt_dir, hydra_dir


def create_mock_checkpoint(
    checkpoint_dir: Path,
    checkpoint_name: str = "epoch_005.ckpt",
    metrics: Optional[dict[str, float]] = None,
    create_config: bool = True,
) -> Path:
    """Create a mock checkpoint file with optional metrics and config.

    Args:
        checkpoint_dir: Directory to create checkpoint in.
        checkpoint_name: Name of the checkpoint file.
        metrics: Optional metrics to save alongside checkpoint.
        create_config: Whether to create Hydra config files.

    Returns:
        Path to the created checkpoint file.
    """
    # Create checkpoint file
    checkpoint_path = checkpoint_dir / checkpoint_name
    checkpoint_path.touch()

    # Create metrics files if provided
    if metrics:
        import json

        metrics_dir = checkpoint_dir.parent / "metrics"
        metrics_dir.mkdir(exist_ok=True)

        metrics_path = metrics_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump([metrics], f, indent=2)

    # Create Hydra config if requested
    if create_config:
        hydra_dir = checkpoint_dir.parent / ".hydra"
        hydra_dir.mkdir(exist_ok=True)

        config_path = hydra_dir / "config.yaml"
        config_content = """
defaults:
  - _self_
  - data: mnist
  - model: mnist
  - trainer: default

data:
  batch_size: 64

model:
  compile: false

trainer:
  max_epochs: 10
"""
        config_path.write_text(config_content)

    return checkpoint_path


def create_mock_trainer(**kwargs) -> MagicMock:
    """Create a mock trainer for testing callbacks.

    Args:
        **kwargs: Additional attributes to set on the trainer.

    Returns:
        Mock trainer object.
    """
    trainer = MagicMock(spec=Trainer)
    trainer.current_epoch = kwargs.get("current_epoch", 5)
    trainer.global_step = kwargs.get("global_step", 100)
    trainer.log_dir = kwargs.get("log_dir", "/tmp/logs")
    trainer.callback_metrics = kwargs.get(
        "callback_metrics",
        {"train/loss": 0.5, "train/acc": 0.85, "val/loss": 0.6, "val/acc": 0.82},
    )
    trainer.optimizers = kwargs.get("optimizers", [])

    # Add any additional attributes
    for key, value in kwargs.items():
        if key not in [
            "current_epoch",
            "global_step",
            "log_dir",
            "callback_metrics",
            "optimizers",
        ]:
            setattr(trainer, key, value)

    return trainer


def assert_file_exists_and_contains(
    file_path: Path, required_strings: list[str], file_description: str = "File"
) -> None:
    """Assert file exists and contains required strings.

    Args:
        file_path: Path to the file to check.
        required_strings: List of strings that must be in the file.
        file_description: Description of the file for error messages.
    """
    assert file_path.exists(), f"{file_description} not found: {file_path}"
    content = file_path.read_text()

    for string in required_strings:
        assert string in content, f"{file_description} should contain '{string}'"


def build_test_command(
    script_path: str,
    tmp_path: Path,
    experiment_type: Optional[str] = None,
    additional_args: Optional[list[str]] = None,
    overrides: Optional[list[str]] = None,
) -> list[str]:
    """Build standard test command with common options.

    Args:
        script_path: Path to the script to run.
        tmp_path: Temporary path for outputs.
        experiment_type: Optional experiment type to run.
        additional_args: Additional command line arguments.
        overrides: Hydra configuration overrides.

    Returns:
        Complete command list.
    """
    command = [script_path]

    # Add common test flags
    command.extend(
        [
            f"hydra.sweep.dir={tmp_path}",
            "++trainer.fast_dev_run=true",
            "++trainer.accelerator=cpu",
            "++trainer.devices=1",
            "++trainer.logger=false",
        ]
    )

    if experiment_type:
        command.append(f"experiment={experiment_type}")

    if additional_args:
        command.extend(additional_args)

    if overrides:
        command.extend(overrides)

    return command


def mock_cli_interaction():
    """Decorator for mocking CLI interactions.

    This decorator patches common CLI interaction methods used in testing.
    """

    def decorator(func):
        @patch("subprocess.run")
        @patch("rich.prompt.Prompt.ask")
        @patch("rich.prompt.IntPrompt.ask")
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator


class MockLightningModule:
    """Mock Lightning module for testing."""

    def __init__(self, num_params: int = 1000):
        """Initialize mock module.

        Args:
            num_params: Number of mock parameters.
        """
        self.num_params = num_params
        self.hparams = DictConfig({"model": {"compile": False}, "optimizer": {"lr": 0.001}})

    def parameters(self):
        """Mock parameters method."""
        import torch

        for _ in range(self.num_params):
            param = torch.randn(10, 10, requires_grad=True)
            # Add mock gradient
            param.grad = torch.randn_like(param)
            yield param

    def named_parameters(self):
        """Mock named_parameters method."""
        import torch

        for i in range(self.num_params):
            param = torch.randn(10, 10, requires_grad=True)
            param.grad = torch.randn_like(param)
            yield f"layer_{i}.weight", param


def skip_if_no_gpu():
    """Skip test if no GPU is available."""
    import pytest
    import torch

    return pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")


def skip_if_no_mps():
    """Skip test if MPS is not available."""
    import pytest
    import torch

    return pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
