"""Base Lightning Module with common functionality."""

from typing import Any, Optional

import torch
from lightning import LightningModule
from torchmetrics import Accuracy, MaxMetric, MeanMetric, MetricCollection

from src.utils.metrics_utils import compute_gradient_norms, get_learning_rates


class BaseLitModule(LightningModule):
    """Base Lightning Module with common functionality for classification tasks.

    This base class provides:
    - Common metric setup for train/val/test phases
    - Gradient norm tracking
    - Learning rate logging
    - Standardized metric logging patterns
    - Model compilation support
    """

    def __init__(
        self,
        compile: bool = False,
        track_grad_norm: bool = True,
        log_learning_rates: bool = True,
    ) -> None:
        """Initialize base Lightning module.

        Args:
            compile: Whether to compile the model using torch.compile().
            track_grad_norm: Whether to track gradient norms during training.
            log_learning_rates: Whether to log learning rates during training.
        """
        super().__init__()

        # Save hyperparameters
        self.save_hyperparameters(logger=False)

        # Training settings
        self.compile = compile
        self.track_grad_norm = track_grad_norm
        self.log_learning_rates = log_learning_rates

        # Placeholders for metrics (set up in setup_metrics)
        self.train_metrics: Optional[MetricCollection] = None
        self.val_metrics: Optional[MetricCollection] = None
        self.test_metrics: Optional[MetricCollection] = None
        self.val_acc_best: Optional[MaxMetric] = None

    def setup_metrics(
        self,
        num_classes: int,
        task: str = "multiclass",
        additional_metrics: Optional[dict[str, Any]] = None,
    ) -> None:
        """Setup metrics for all training phases.

        Args:
            num_classes: Number of classes for classification.
            task: Type of classification task ("multiclass", "binary", etc.).
            additional_metrics: Additional metrics to include beyond acc and loss.
        """
        # Base metrics
        metric_dict = {
            "acc": Accuracy(task=task, num_classes=num_classes),
            "loss": MeanMetric(),
        }

        # Add any additional metrics
        if additional_metrics:
            metric_dict.update(additional_metrics)

        # Create metric collections for each phase
        self.train_metrics = MetricCollection(metric_dict.copy(), prefix="train/")
        self.val_metrics = MetricCollection(metric_dict.copy(), prefix="val/")
        self.test_metrics = MetricCollection(metric_dict.copy(), prefix="test/")

        # Best validation accuracy tracker
        self.val_acc_best = MaxMetric()

    def log_metrics(
        self,
        metrics: MetricCollection,
        loss: torch.Tensor,
        preds: torch.Tensor,
        targets: torch.Tensor,
        on_step: bool = False,
        on_epoch: bool = True,
        prog_bar: bool = True,
    ) -> None:
        """Log metrics for a given phase.

        Args:
            metrics: MetricCollection to update and log.
            loss: Loss tensor.
            preds: Model predictions.
            targets: Ground truth targets.
            on_step: Whether to log on each step.
            on_epoch: Whether to log on each epoch.
            prog_bar: Whether to show in progress bar.
        """
        # Update metrics
        metrics["loss"](loss)
        metrics["acc"](preds, targets)

        # Log all metrics
        self.log_dict(metrics, on_step=on_step, on_epoch=on_epoch, prog_bar=prog_bar)

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step.

        Args:
            batch: Batch containing inputs and targets.
            batch_idx: Index of the current batch.

        Returns:
            Loss tensor for backpropagation.
        """
        loss, preds, targets = self.model_step(batch)

        # Log training metrics
        if self.train_metrics is not None:
            self.log_metrics(self.train_metrics, loss, preds, targets)

        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step.

        Args:
            batch: Batch containing inputs and targets.
            batch_idx: Index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # Log validation metrics
        if self.val_metrics is not None:
            self.log_metrics(self.val_metrics, loss, preds, targets)

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step.

        Args:
            batch: Batch containing inputs and targets.
            batch_idx: Index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # Log test metrics
        if self.test_metrics is not None:
            self.log_metrics(self.test_metrics, loss, preds, targets)

    def on_validation_epoch_end(self) -> None:
        """Called at the end of validation epoch."""
        if self.val_metrics is not None and self.val_acc_best is not None:
            # Update best validation accuracy
            acc = self.val_metrics["val/acc"].compute()
            self.val_acc_best(acc)
            self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """Called before optimizer step.

        Args:
            optimizer: The optimizer being stepped.
        """
        # Log gradient norms
        if self.track_grad_norm:
            grad_norms = compute_gradient_norms(self, per_layer=False)
            for key, value in grad_norms.items():
                self.log(f"grad_norm/{key}", value, on_step=True, on_epoch=False)

        # Log learning rates
        if self.log_learning_rates:
            lr_dict = get_learning_rates(optimizer)
            for key, value in lr_dict.items():
                self.log(f"lr/{key}", value, on_step=True, on_epoch=False)

    def configure_model(self) -> None:
        """Configure the model (called after model setup)."""
        if self.compile and hasattr(torch, "compile"):
            self.net = torch.compile(self.net)

    def model_step(
        self, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model forward pass.

        This method should be implemented by subclasses to define the specific
        forward pass logic for their model.

        Args:
            batch: Batch containing inputs and targets.

        Returns:
            Tuple of (loss, predictions, targets).
        """
        raise NotImplementedError("Subclasses must implement model_step method")

    def configure_optimizers(self) -> Any:
        """Configure optimizers and learning rate schedulers.

        This method should be implemented by subclasses to define their
        specific optimizer and scheduler configuration.

        Returns:
            Optimizer configuration (optimizer, scheduler, etc.).
        """
        raise NotImplementedError("Subclasses must implement configure_optimizers method")
