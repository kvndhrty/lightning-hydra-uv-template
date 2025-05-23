"""Test enhanced logging callbacks."""
import json
import time
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest
import torch
from lightning import Trainer

from src.utils.logging_callbacks import (
    CheckpointMetricsLogger,
    MetricsLogger,
    WandbEnhancedLogger,
)


@pytest.fixture
def mock_trainer():
    """Create mock trainer with necessary attributes."""
    trainer = MagicMock(spec=Trainer)
    trainer.current_epoch = 5
    trainer.global_step = 100
    trainer.log_dir = "/tmp/logs"
    trainer.callback_metrics = {
        "train/loss": torch.tensor(0.5),
        "train/acc": torch.tensor(0.85),
        "val/loss": torch.tensor(0.6),
        "val/acc": torch.tensor(0.82),
    }
    
    # Mock optimizers
    optimizer = MagicMock()
    optimizer.param_groups = [{"lr": 0.001}, {"lr": 0.0001}]
    trainer.optimizers = [optimizer]
    
    return trainer


@pytest.fixture
def mock_module():
    """Create mock Lightning module."""
    module = MagicMock()
    
    # Mock parameters with gradients
    param1 = torch.nn.Parameter(torch.randn(10, 10))
    param1.grad = torch.randn(10, 10)
    param2 = torch.nn.Parameter(torch.randn(5, 5))
    param2.grad = torch.randn(5, 5)
    
    module.parameters.return_value = [param1, param2]
    module.named_parameters.return_value = [("layer1.weight", param1), ("layer2.weight", param2)]
    
    return module


class TestMetricsLogger:
    """Test MetricsLogger callback."""
    
    def test_init(self):
        """Test initialization."""
        logger = MetricsLogger(save_json=True, save_csv=False)
        assert logger.save_json is True
        assert logger.save_csv is False
        assert logger.metrics_history == []
    
    def test_setup(self, tmp_path, mock_trainer, mock_module):
        """Test setup creates directories."""
        logger = MetricsLogger(save_dir=tmp_path / "metrics")
        logger.setup(mock_trainer, mock_module, "fit")
        
        assert logger.save_dir.exists()
        assert logger.save_dir == tmp_path / "metrics"
    
    def test_epoch_timing(self, mock_trainer, mock_module):
        """Test epoch timing tracking."""
        logger = MetricsLogger(log_epoch_time=True)
        
        # Simulate epoch
        logger.on_train_epoch_start(mock_trainer, mock_module)
        time.sleep(0.1)  # Simulate some training time
        logger.on_train_epoch_end(mock_trainer, mock_module)
        
        assert len(logger.metrics_history) == 1
        assert "epoch_time" in logger.metrics_history[0]
        assert logger.metrics_history[0]["epoch_time"] >= 0.1
    
    def test_metric_logging(self, mock_trainer, mock_module):
        """Test metric logging."""
        logger = MetricsLogger()
        logger.on_train_epoch_end(mock_trainer, mock_module)
        
        metrics = logger.metrics_history[0]
        assert metrics["epoch"] == 5
        assert metrics["global_step"] == 100
        assert metrics["train/loss"] == 0.5
        assert metrics["val/acc"] == 0.82
        assert "timestamp" in metrics
    
    def test_learning_rate_logging(self, mock_trainer, mock_module):
        """Test learning rate logging."""
        logger = MetricsLogger(log_lr=True)
        logger.on_train_epoch_end(mock_trainer, mock_module)
        
        metrics = logger.metrics_history[0]
        assert metrics["lr/optimizer_0_group_0"] == 0.001
        assert metrics["lr/optimizer_0_group_1"] == 0.0001
    
    def test_gradient_norm_logging(self, mock_trainer, mock_module):
        """Test gradient norm computation."""
        logger = MetricsLogger(log_grad_norm=True)
        logger.on_train_epoch_end(mock_trainer, mock_module)
        
        metrics = logger.metrics_history[0]
        assert "grad_norm/total" in metrics
        assert "grad_norm/layer1.weight" in metrics
        assert "grad_norm/layer2.weight" in metrics
        assert metrics["grad_norm/total"] > 0
    
    def test_save_json(self, tmp_path, mock_trainer, mock_module):
        """Test JSON saving."""
        logger = MetricsLogger(save_dir=tmp_path, save_json=True, save_csv=False)
        logger.setup(mock_trainer, mock_module, "fit")
        logger.on_train_epoch_end(mock_trainer, mock_module)
        
        json_path = tmp_path / "metrics.json"
        assert json_path.exists()
        
        with open(json_path) as f:
            data = json.load(f)
        
        assert len(data) == 1
        assert data[0]["epoch"] == 5
    
    def test_save_csv(self, tmp_path, mock_trainer, mock_module):
        """Test CSV saving."""
        logger = MetricsLogger(save_dir=tmp_path, save_json=False, save_csv=True)
        logger.setup(mock_trainer, mock_module, "fit")
        logger.on_train_epoch_end(mock_trainer, mock_module)
        
        csv_path = tmp_path / "metrics.csv"
        assert csv_path.exists()
        
        df = pd.read_csv(csv_path)
        assert len(df) == 1
        assert df.iloc[0]["epoch"] == 5
    
    def test_test_results(self, tmp_path, mock_trainer, mock_module):
        """Test saving test results."""
        mock_trainer.callback_metrics = {
            "test/loss": torch.tensor(0.4),
            "test/acc": torch.tensor(0.9),
        }
        
        logger = MetricsLogger(save_dir=tmp_path)
        logger.setup(mock_trainer, mock_module, "test")
        logger.on_test_end(mock_trainer, mock_module)
        
        # Check JSON
        json_path = tmp_path / "test_results.json"
        assert json_path.exists()
        
        with open(json_path) as f:
            data = json.load(f)
        assert data["test/acc"] == 0.9
        assert data["stage"] == "test"


class TestCheckpointMetricsLogger:
    """Test CheckpointMetricsLogger callback."""
    
    def test_init(self):
        """Test initialization."""
        logger = CheckpointMetricsLogger()
        assert logger.checkpoint_dir is None
    
    def test_setup_finds_checkpoint_dir(self, mock_trainer, mock_module, tmp_path):
        """Test finding checkpoint directory from callbacks."""
        # Add mock checkpoint callback
        ckpt_callback = MagicMock()
        ckpt_callback.dirpath = str(tmp_path / "checkpoints")
        mock_trainer.callbacks = [ckpt_callback]
        
        logger = CheckpointMetricsLogger()
        logger.setup(mock_trainer, mock_module, "fit")
        
        assert logger.checkpoint_dir == tmp_path / "checkpoints"
    
    def test_checkpoint_metadata(self, mock_trainer, mock_module):
        """Test checkpoint metadata creation."""
        logger = CheckpointMetricsLogger()
        logger.checkpoint_dir = Path("/tmp/checkpoints")
        
        checkpoint = {}
        logger.on_save_checkpoint(mock_trainer, mock_module, checkpoint)
        
        assert hasattr(logger, "_pending_metadata")
        metadata = logger._pending_metadata
        assert metadata["epoch"] == 5
        assert metadata["global_step"] == 100
        assert "metrics" in metadata
        assert abs(metadata["metrics"]["val/acc"] - 0.82) < 0.01


class TestWandbEnhancedLogger:
    """Test WandbEnhancedLogger callback."""
    
    def test_init(self):
        """Test initialization."""
        logger = WandbEnhancedLogger(log_gradients=False)
        assert logger.log_gradients is False
        assert logger._wandb_logger is None
    
    @patch("wandb.watch")
    def test_setup_watches_model(self, mock_watch, mock_trainer, mock_module):
        """Test model watching setup."""
        # Mock W&B logger
        wandb_logger = MagicMock()
        wandb_logger.experiment.config = {}
        mock_trainer.loggers = [wandb_logger]
        
        logger = WandbEnhancedLogger(watch_model=True)
        logger.setup(mock_trainer, mock_module, "fit")
        
        mock_watch.assert_called_once_with(mock_module, log="all")
    
    @patch("wandb.log")
    def test_gradient_logging(self, mock_wandb_log, mock_trainer, mock_module):
        """Test gradient norm logging to W&B."""
        # Mock W&B logger
        wandb_logger = MagicMock()
        wandb_logger.experiment.config = {}
        mock_trainer.loggers = [wandb_logger]
        
        logger = WandbEnhancedLogger(log_gradients=True)
        logger._wandb_logger = wandb_logger
        logger.on_train_epoch_end(mock_trainer, mock_module)
        
        # Check gradient norm was logged
        calls = mock_wandb_log.call_args_list
        assert any("gradients/total_norm" in call[0][0] for call in calls)
    
    @patch("wandb.log")
    def test_learning_rate_logging(self, mock_wandb_log, mock_trainer, mock_module):
        """Test learning rate logging to W&B."""
        # Mock W&B logger
        wandb_logger = MagicMock()
        wandb_logger.experiment.config = {}
        mock_trainer.loggers = [wandb_logger]
        
        logger = WandbEnhancedLogger()
        logger._wandb_logger = wandb_logger
        logger.on_train_epoch_end(mock_trainer, mock_module)
        
        # Check LR was logged
        calls = mock_wandb_log.call_args_list
        assert any("lr/optimizer_0_group_0" in call[0][0] for call in calls)