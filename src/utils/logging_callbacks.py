"""Custom logging callbacks for enhanced metric tracking."""
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from lightning import Callback, LightningModule, Trainer
from omegaconf import OmegaConf


class MetricsLogger(Callback):
    """Logs metrics to JSON and CSV files alongside checkpoints.
    
    This callback creates structured logs that can be easily parsed by CLI tools.
    """
    
    def __init__(
        self,
        save_dir: Optional[Path] = None,
        save_json: bool = True,
        save_csv: bool = True,
        log_grad_norm: bool = True,
        log_lr: bool = True,
        log_epoch_time: bool = True,
    ):
        """Initialize metrics logger.
        
        Args:
            save_dir: Directory to save logs. If None, uses trainer's log_dir
            save_json: Whether to save JSON logs
            save_csv: Whether to save CSV logs
            log_grad_norm: Whether to log gradient norms
            log_lr: Whether to log learning rates
            log_epoch_time: Whether to log epoch timing
        """
        self.save_dir = save_dir
        self.save_json = save_json
        self.save_csv = save_csv
        self.log_grad_norm = log_grad_norm
        self.log_lr = log_lr
        self.log_epoch_time = log_epoch_time
        
        self.metrics_history: List[Dict[str, Any]] = []
        self.epoch_start_time: Optional[float] = None
    
    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Setup logging paths."""
        if self.save_dir is None:
            self.save_dir = Path(trainer.log_dir) / "metrics"
        else:
            self.save_dir = Path(self.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save initial config if available
        config_path = self.save_dir / "config.yaml"
        if hasattr(pl_module, "hparams") and isinstance(pl_module.hparams, (dict, OmegaConf)):
            try:
                OmegaConf.save(pl_module.hparams, config_path)
            except Exception:
                # Skip config saving if it fails
                pass
    
    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Record epoch start time."""
        self.epoch_start_time = time.time()
    
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Log metrics at end of epoch."""
        metrics = {
            "epoch": trainer.current_epoch,
            "global_step": trainer.global_step,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        # Add epoch timing
        if self.log_epoch_time and self.epoch_start_time is not None:
            metrics["epoch_time"] = time.time() - self.epoch_start_time
        
        # Add all logged metrics
        for key, value in trainer.callback_metrics.items():
            if isinstance(value, torch.Tensor):
                metrics[key] = value.item()
            else:
                metrics[key] = value
        
        # Add learning rates
        if self.log_lr:
            for i, optimizer in enumerate(trainer.optimizers):
                for j, param_group in enumerate(optimizer.param_groups):
                    metrics[f"lr/optimizer_{i}_group_{j}"] = param_group["lr"]
        
        # Add gradient norms
        if self.log_grad_norm:
            grad_norm_dict = self._compute_grad_norm(pl_module)
            metrics.update(grad_norm_dict)
        
        # Store metrics
        self.metrics_history.append(metrics)
        
        # Save to files
        self._save_metrics()
    
    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Log test metrics."""
        test_metrics = {
            "epoch": trainer.current_epoch,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "stage": "test",
        }
        
        # Add test metrics
        for key, value in trainer.callback_metrics.items():
            if "test" in key:
                if isinstance(value, torch.Tensor):
                    test_metrics[key] = value.item()
                else:
                    test_metrics[key] = value
        
        # Save test results separately
        if self.save_json:
            test_path = self.save_dir / "test_results.json"
            with open(test_path, "w") as f:
                json.dump(test_metrics, f, indent=2)
        
        if self.save_csv:
            test_path = self.save_dir / "test_results.csv"
            pd.DataFrame([test_metrics]).to_csv(test_path, index=False)
    
    def _compute_grad_norm(self, pl_module: LightningModule) -> Dict[str, float]:
        """Compute gradient norms for different parameter groups."""
        grad_norms = {}
        
        # Total gradient norm
        total_norm = 0.0
        for p in pl_module.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        grad_norms["grad_norm/total"] = total_norm
        
        # Per-layer gradient norms (optional, for debugging)
        for name, param in pl_module.named_parameters():
            if param.grad is not None:
                grad_norms[f"grad_norm/{name}"] = param.grad.data.norm(2).item()
        
        return grad_norms
    
    def _save_metrics(self) -> None:
        """Save metrics to JSON and CSV files."""
        if not self.metrics_history:
            return
        
        if self.save_json:
            json_path = self.save_dir / "metrics.json"
            with open(json_path, "w") as f:
                json.dump(self.metrics_history, f, indent=2)
        
        if self.save_csv:
            csv_path = self.save_dir / "metrics.csv"
            df = pd.DataFrame(self.metrics_history)
            df.to_csv(csv_path, index=False)


class CheckpointMetricsLogger(Callback):
    """Logs checkpoint-specific metrics for easy access by CLI tools."""
    
    def __init__(self, checkpoint_dir: Optional[Path] = None):
        """Initialize checkpoint metrics logger.
        
        Args:
            checkpoint_dir: Directory where checkpoints are saved
        """
        self.checkpoint_dir = checkpoint_dir
    
    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Setup checkpoint directory."""
        if self.checkpoint_dir is None:
            # Find checkpoint callback to get its directory
            for callback in trainer.callbacks:
                if hasattr(callback, "dirpath"):
                    self.checkpoint_dir = Path(callback.dirpath)
                    break
    
    def on_save_checkpoint(
        self, trainer: Trainer, pl_module: LightningModule, checkpoint: Dict[str, Any]
    ) -> None:
        """Save metrics when checkpoint is saved."""
        if self.checkpoint_dir is None:
            return
        
        # Create checkpoint metadata
        metadata = {
            "epoch": trainer.current_epoch,
            "global_step": trainer.global_step,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": {},
        }
        
        # Add all current metrics
        for key, value in trainer.callback_metrics.items():
            if isinstance(value, torch.Tensor):
                metadata["metrics"][key] = value.item()
            else:
                metadata["metrics"][key] = value
        
        # Save metadata next to checkpoint
        # The checkpoint filename will be available in the next callback
        self._pending_metadata = metadata
    
    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Ensure final metrics are saved."""
        if self.checkpoint_dir is None:
            return
        
        # Create summary of all checkpoints
        summary = []
        for ckpt_file in self.checkpoint_dir.glob("*.ckpt"):
            # Try to find associated metrics
            metrics_file = ckpt_file.with_suffix(".json")
            if metrics_file.exists():
                with open(metrics_file) as f:
                    data = json.load(f)
                    data["checkpoint"] = ckpt_file.name
                    summary.append(data)
        
        # Save summary
        if summary:
            summary_path = self.checkpoint_dir / "checkpoint_summary.json"
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)


class WandbEnhancedLogger(Callback):
    """Enhanced W&B logging with additional features."""
    
    def __init__(
        self,
        log_gradients: bool = True,
        log_model_architecture: bool = True,
        log_best_model: bool = True,
        watch_model: bool = True,
    ):
        """Initialize enhanced W&B logger.
        
        Args:
            log_gradients: Whether to log gradient histograms
            log_model_architecture: Whether to log model architecture
            log_best_model: Whether to upload best model as artifact
            watch_model: Whether to watch model gradients
        """
        self.log_gradients = log_gradients
        self.log_model_architecture = log_model_architecture
        self.log_best_model = log_best_model
        self.watch_model = watch_model
        self._wandb_logger = None
    
    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Setup W&B logger."""
        # Find W&B logger
        for logger in trainer.loggers:
            if hasattr(logger, "experiment") and hasattr(logger.experiment, "config"):
                self._wandb_logger = logger
                break
        
        if self._wandb_logger is None:
            return
        
        # Watch model for gradients
        if self.watch_model and stage == "fit":
            try:
                import wandb
                wandb.watch(pl_module, log="all" if self.log_gradients else "parameters")
            except ImportError:
                pass
        
        # Log model architecture
        if self.log_model_architecture:
            self._log_model_summary(pl_module)
    
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Log additional metrics to W&B."""
        if self._wandb_logger is None:
            return
        
        try:
            import wandb
        except ImportError:
            return
        
        # Log learning rate
        for i, optimizer in enumerate(trainer.optimizers):
            for j, param_group in enumerate(optimizer.param_groups):
                wandb.log({f"lr/optimizer_{i}_group_{j}": param_group["lr"]})
        
        # Log gradient norms
        if self.log_gradients:
            total_norm = 0.0
            for p in pl_module.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            wandb.log({"gradients/total_norm": total_norm})
    
    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Save best model as W&B artifact."""
        if self._wandb_logger is None or not self.log_best_model:
            return
        
        try:
            import wandb
        except ImportError:
            return
        
        # Find best checkpoint
        best_model_path = trainer.checkpoint_callback.best_model_path
        if best_model_path and Path(best_model_path).exists():
            artifact = wandb.Artifact(
                name=f"model-{wandb.run.id}",
                type="model",
                description=f"Best model from run {wandb.run.name}",
                metadata={
                    "best_metric": trainer.checkpoint_callback.best_model_score.item()
                    if hasattr(trainer.checkpoint_callback, "best_model_score")
                    else None,
                    "monitor": trainer.checkpoint_callback.monitor,
                }
            )
            artifact.add_file(best_model_path)
            wandb.log_artifact(artifact)
    
    def _log_model_summary(self, pl_module: LightningModule) -> None:
        """Log model architecture summary."""
        try:
            import wandb
            from torchinfo import summary
        except ImportError:
            return
        
        try:
            # Get model summary
            model_summary = summary(
                pl_module,
                input_size=(1, 1, 28, 28),  # Adjust based on your model
                verbose=0,
                col_names=["input_size", "output_size", "num_params"],
            )
            
            # Log as text
            wandb.run.summary["model_architecture"] = str(model_summary)
        except Exception:
            # Fallback to simple parameter count
            total_params = sum(p.numel() for p in pl_module.parameters())
            trainable_params = sum(p.numel() for p in pl_module.parameters() if p.requires_grad)
            wandb.run.summary["total_parameters"] = total_params
            wandb.run.summary["trainable_parameters"] = trainable_params