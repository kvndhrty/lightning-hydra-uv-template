"""Test CLI functionality."""
from pathlib import Path
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from src.cli import main as cli_main
from src.utils.checkpoint_utils import CheckpointInfo, scan_checkpoints


def test_cli_help():
    """Test CLI help command."""
    # Test main help
    with pytest.raises(SystemExit) as excinfo:
        cli_main(["--help"])
    assert excinfo.value.code == 0


def test_cli_ckpt_help():
    """Test checkpoint subcommand help."""
    with pytest.raises(SystemExit) as excinfo:
        cli_main(["ckpt", "--help"])
    assert excinfo.value.code == 0


def test_checkpoint_info():
    """Test CheckpointInfo class."""
    ckpt = CheckpointInfo(
        path=Path("logs/train/runs/2024-01-20_15-30-45/checkpoints/epoch_042.ckpt"),
        run_dir=Path("logs/train/runs/2024-01-20_15-30-45"),
        epoch=42,
    )

    assert ckpt.name == "2024-01-20_15-30-45/epoch_042.ckpt"
    assert ckpt.epoch == 42
    assert not ckpt.has_config  # No config path set


def test_scan_checkpoints_empty(tmp_path):
    """Test scanning empty logs directory."""
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()

    checkpoints = scan_checkpoints(logs_dir)
    assert len(checkpoints) == 0


def test_scan_checkpoints_with_files(tmp_path):
    """Test scanning logs directory with checkpoints."""
    # Create mock logs structure
    run_dir = tmp_path / "logs" / "train" / "runs" / "2024-01-20_15-30-45"
    ckpt_dir = run_dir / "checkpoints"
    hydra_dir = run_dir / ".hydra"

    ckpt_dir.mkdir(parents=True)
    hydra_dir.mkdir(parents=True)

    # Create checkpoint files
    (ckpt_dir / "epoch_010.ckpt").touch()
    (ckpt_dir / "last.ckpt").touch()

    # Create config files
    config = {"model": {"lr": 0.001}, "data": {"batch_size": 32}}
    OmegaConf.save(config, hydra_dir / "config.yaml")
    OmegaConf.save(["model.lr=0.001", "data.batch_size=32"], hydra_dir / "overrides.yaml")

    # Scan checkpoints
    checkpoints = scan_checkpoints(tmp_path / "logs")

    assert len(checkpoints) == 2
    assert all(ckpt.has_config for ckpt in checkpoints)

    # Check epoch parsing
    epoch_ckpt = next(c for c in checkpoints if "epoch_010" in str(c.path))
    assert epoch_ckpt.epoch == 10


def test_get_resume_command(tmp_path):
    """Test resume command generation."""
    # Create mock checkpoint
    run_dir = tmp_path / "logs" / "train" / "runs" / "2024-01-20_15-30-45"
    ckpt_path = run_dir / "checkpoints" / "epoch_010.ckpt"
    hydra_dir = run_dir / ".hydra"

    hydra_dir.mkdir(parents=True)
    ckpt_path.parent.mkdir(parents=True)
    ckpt_path.touch()

    # Save overrides
    OmegaConf.save(["model.lr=0.001", "trainer.max_epochs=100"], hydra_dir / "overrides.yaml")

    ckpt = CheckpointInfo(
        path=ckpt_path,
        run_dir=run_dir,
        overrides_path=hydra_dir / "overrides.yaml",
    )

    from src.utils.checkpoint_utils import get_resume_command

    executable, args = get_resume_command(ckpt)

    assert executable == "python"
    assert args[0] == "src/train.py"
    assert "model.lr=0.001" in args
    assert "trainer.max_epochs=100" in args
    assert f"ckpt_path={ckpt_path}" in args


@patch("subprocess.run")
def test_cli_ckpt_list(mock_run, tmp_path, capsys):
    """Test checkpoint list command."""
    # Create mock checkpoints
    run_dir = tmp_path / "logs" / "train" / "runs" / "2024-01-20_15-30-45"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "epoch_010.ckpt").touch()

    # Run command (list doesn't exit)
    cli_main(["ckpt", "list", f"--logs-dir={tmp_path / 'logs'}"])

    # Check output contains checkpoint info
    captured = capsys.readouterr()
    assert "epoch_010.ckpt" in captured.out


@patch("subprocess.run")
@patch("rich.prompt.Prompt.ask")
@patch("rich.prompt.IntPrompt.ask")
def test_cli_ckpt_resume_dry_run(mock_int_prompt, mock_prompt, mock_run, tmp_path, capsys):
    """Test checkpoint resume command in dry-run mode."""
    # Create mock checkpoint with config
    run_dir = tmp_path / "logs" / "train" / "runs" / "2024-01-20_15-30-45"
    ckpt_path = run_dir / "checkpoints" / "epoch_010.ckpt"
    hydra_dir = run_dir / ".hydra"

    hydra_dir.mkdir(parents=True)
    ckpt_path.parent.mkdir(parents=True)
    ckpt_path.touch()

    # Create config files
    OmegaConf.save({"model": {"lr": 0.001}}, hydra_dir / "config.yaml")
    OmegaConf.save(["model.lr=0.001"], hydra_dir / "overrides.yaml")

    # Mock user selection
    mock_int_prompt.return_value = 1

    # Run command with dry-run
    cli_main(["ckpt", "resume", f"--logs-dir={tmp_path / 'logs'}", "--dry-run"])

    # Check output
    captured = capsys.readouterr()
    assert "Resume command:" in captured.out
    assert "python src/train.py" in captured.out
    assert "ckpt_path=" in captured.out
    assert "Dry run mode" in captured.out

    # Subprocess should not be called in dry-run
    mock_run.assert_not_called()


def test_cli_invalid_command():
    """Test invalid CLI command."""
    with pytest.raises(SystemExit) as excinfo:
        cli_main(["invalid"])
    assert excinfo.value.code == 2  # argparse exits with 2 for invalid arguments
