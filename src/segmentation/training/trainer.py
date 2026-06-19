"""
Route Resilience — Phase 2
src/segmentation/training/trainer.py

Modular, production-grade Trainer class supporting AMP, gradient accumulation,
dynamic device allocation, DDP readiness, and robust checkpoint/metrics logging.
"""

from __future__ import annotations

import os
import time
import yaml
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

from src.segmentation.evaluation.metrics import compute_batch_metrics

logger = logging.getLogger(__name__)


class Trainer:
    """
    Modular Trainer for training segmentation models. Designed to be fully compatible
    with local and Kaggle runs, and extensible to Distributed Data Parallel (DDP).
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        val_dataset: Dataset,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.val_dataset = val_dataset
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config or {}

        # Core training hyperparameters
        cfg_train = self.config.get("training", {})
        self.epochs = cfg_train.get("epochs", 15)
        self.grad_clip_norm = cfg_train.get("grad_clip_norm", 1.0)
        self.early_stopping_patience = cfg_train.get("early_stopping_patience", 5)
        self.device = torch.device(cfg_train.get("device", "cuda") if torch.cuda.is_available() else "cpu")
        self.accumulation_steps = cfg_train.get("accumulation_steps", 1)  # Grad accumulation support

        # Output paths
        cfg_paths = self.config.get("paths", {})
        self.output_dir = cfg_paths.get("output_dir", "/kaggle/working/output")

        # Initialize directories and DDP process states
        self.is_master = True  # Can be mapped to rank == 0 in DDP
        self._setup_experiment_dir()

        # State tracking
        self.start_epoch = 1
        self.best_metric = 0.0
        self.best_metric_name = "val_dice"
        self.epochs_no_improve = 0
        self.history: Dict[str, List[float]] = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "val_dice": [],
            "val_iou": [],
            "val_precision": [],
            "val_recall": [],
            "learning_rate": [],
        }

        # PyTorch AMP scaler
        self.scaler = GradScaler(enabled=(self.device.type == "cuda"))

        # Qualitative visualization settings
        self.num_val_samples = min(3, len(val_dataset))

        # Push model to device
        self.model.to(self.device)

    def _setup_experiment_dir(self) -> None:
        """Create structured output folders dynamically (output/experiment_001, etc.)."""
        if not self.is_master:
            return

        os.makedirs(self.output_dir, exist_ok=True)

        # Get existing experiment count
        exps = [
            d for d in os.listdir(self.output_dir)
            if d.startswith("experiment_") and os.path.isdir(os.path.join(self.output_dir, d))
        ]

        exp_num = 1
        if exps:
            nums = []
            for exp in exps:
                try:
                    nums.append(int(exp.split("_")[1]))
                except ValueError:
                    pass
            if nums:
                exp_num = max(nums) + 1

        exp_name = f"experiment_{exp_num:03d}"
        self.experiment_dir = os.path.join(self.output_dir, exp_name)

        self.model_dir = os.path.join(self.experiment_dir, "models")
        self.plot_dir = os.path.join(self.experiment_dir, "plots")
        self.log_dir = os.path.join(self.experiment_dir, "logs")
        self.pred_dir = os.path.join(self.experiment_dir, "predictions")

        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.pred_dir, exist_ok=True)

        # Save config copy
        with open(os.path.join(self.experiment_dir, "config.yaml"), "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)

        logger.info(f"Experiment folder initialized: {self.experiment_dir}")

    def train_epoch(self) -> float:
        """Run single training epoch (returns average loss)."""
        self.model.train()
        total_loss = 0.0
        self.optimizer.zero_grad()

        for step, (images, masks) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            # AMP Autocast
            with autocast(enabled=(self.device.type == "cuda")):
                logits = self.model(images)
                loss = self.criterion(logits, masks)
                # Scale loss according to accumulation steps
                loss = loss / self.accumulation_steps

            self.scaler.scale(loss).backward()

            # Perform optimizer step after accumulating enough gradients
            if (step + 1) % self.accumulation_steps == 0 or (step + 1) == len(self.train_loader):
                if self.grad_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.accumulation_steps

        return total_loss / len(self.train_loader)

    def validate_epoch(self) -> Tuple[float, Dict[str, float]]:
        """Run validation loop, returning average loss and metrics dict."""
        self.model.eval()
        total_loss = 0.0

        accumulated_metrics = {
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "dice": 0.0,
            "iou": 0.0
        }

        with torch.no_grad():
            for images, masks in self.val_loader:
                images = images.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)

                with autocast(enabled=(self.device.type == "cuda")):
                    logits = self.model(images)
                    loss = self.criterion(logits, masks)

                total_loss += loss.item()

                # Calculate metrics for the batch
                batch_metrics = compute_batch_metrics(logits, masks)
                for k, v in batch_metrics.items():
                    accumulated_metrics[k] += v

        # Normalize metrics over loader steps
        num_steps = len(self.val_loader)
        avg_loss = total_loss / num_steps
        avg_metrics = {k: v / num_steps for k, v in accumulated_metrics.items()}

        return avg_loss, avg_metrics

    def fit(self) -> None:
        """Run the complete training and validation cycle."""
        logger.info(f"Starting training on {self.device}...")
        start_time = time.time()

        for epoch in range(self.start_epoch, self.epochs + 1):
            epoch_start = time.time()

            # Train
            train_loss = self.train_epoch()

            # Validate
            val_loss, val_metrics = self.validate_epoch()

            # LR Scheduler step
            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Progress details
            epoch_duration = time.time() - epoch_start
            eta_seconds = epoch_duration * (self.epochs - epoch)
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))

            # Logging metrics
            logger.info(
                f"Epoch {epoch:02d}/{self.epochs:02d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Dice: {val_metrics['dice']:.4f} | "
                f"IoU: {val_metrics['iou']:.4f} | "
                f"LR: {current_lr:.6f} | "
                f"ETA: {eta_str}"
            )

            # Record history
            if self.is_master:
                self.history["epoch"].append(epoch)
                self.history["train_loss"].append(train_loss)
                self.history["val_loss"].append(val_loss)
                self.history["val_dice"].append(val_metrics["dice"])
                self.history["val_iou"].append(val_metrics["iou"])
                self.history["val_precision"].append(val_metrics["precision"])
                self.history["val_recall"].append(val_metrics["recall"])
                self.history["learning_rate"].append(current_lr)

                self._save_history_csv()
                self._save_qualitative_predictions(epoch)

            # Check improvements and save checkpoints
            val_dice = val_metrics["dice"]
            is_best = val_dice > self.best_metric

            if is_best:
                self.best_metric = val_dice
                self.epochs_no_improve = 0
                logger.info(f"✓ New best metric achieved ({self.best_metric_name}: {self.best_metric:.4f}). Saving best model...")
                if self.is_master:
                    self._save_checkpoint(epoch, is_best=True)
            else:
                self.epochs_no_improve += 1

            if self.is_master:
                self._save_checkpoint(epoch, is_best=False)

            # Early Stopping
            if self.epochs_no_improve >= self.early_stopping_patience:
                logger.info(f"Early stopping triggered! No improvement in {self.early_stopping_patience} epochs.")
                break

        total_duration = time.time() - start_time
        logger.info(f"Training completed in {total_duration/60:.2f} minutes.")

    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save training states, best weights, and standalone optimizer/scheduler states."""
        checkpoint = {
            "epoch": epoch,
            "best_metric": self.best_metric,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state": self.scaler.state_dict() if self.scaler else None,
        }

        # Save standard last training checkpoint
        last_ckpt_path = os.path.join(self.model_dir, "last_checkpoint.pth")
        torch.save(checkpoint, last_ckpt_path)

        # Save standalone optimizer and scheduler states
        torch.save(self.optimizer.state_dict(), os.path.join(self.model_dir, "optimizer_state.pth"))
        if self.scheduler:
            torch.save(self.scheduler.state_dict(), os.path.join(self.model_dir, "scheduler_state.pth"))

        # Save config_used.yaml
        with open(os.path.join(self.model_dir, "config_used.yaml"), "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)

        # If it achieved a new best metric, save the model state weights separately
        if is_best:
            best_model_path = os.path.join(self.model_dir, "best_model.pth")
            torch.save(self.model.state_dict(), best_model_path)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Resume training loop state from checkpoint file."""
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])

        if self.scheduler and checkpoint.get("scheduler_state") is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])

        if checkpoint.get("scaler_state") is not None:
            self.scaler.load_state_dict(checkpoint["scaler_state"])

        self.start_epoch = checkpoint["epoch"] + 1
        self.best_metric = checkpoint["best_metric"]

        # Restore history if logs exist in the same run
        history_file = os.path.join(self.log_dir, "training_history.csv")
        if os.path.exists(history_file):
            try:
                df = pd.read_csv(history_file)
                self.history = df.to_dict(orient="list")
                logger.info(f"Restored training history containing {len(df)} records.")
            except Exception as exc:
                logger.warning(f"Failed to restore history file: {exc}")

        logger.info(f"Resumed from epoch {self.start_epoch} (Best metric: {self.best_metric:.4f}).")

    def _save_history_csv(self) -> None:
        """Log history records as CSV."""
        df = pd.DataFrame(self.history)
        history_path = os.path.join(self.log_dir, "training_history.csv")
        df.to_csv(history_path, index=False)

    def _save_qualitative_predictions(self, epoch: int) -> None:
        """Save a comparison grid of image, target, and prediction after epoch validation."""
        self.model.eval()

        fig, axes = plt.subplots(self.num_val_samples, 3, figsize=(12, 4 * self.num_val_samples))
        fig.patch.set_facecolor("#0e1117")

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        with torch.no_grad():
            for idx in range(self.num_val_samples):
                image, mask = self.val_dataset[idx]
                image_dev = image.unsqueeze(0).to(self.device)

                with autocast(enabled=(self.device.type == "cuda")):
                    logits = self.model(image_dev)
                    probs = torch.sigmoid(logits)
                    pred = (probs > 0.5).float().squeeze(0).squeeze(0).cpu().numpy()

                # Denormalize image for correct RGB display
                img_np = image.permute(1, 2, 0).numpy()
                img_np = np.clip((img_np * std + mean) * 255.0, 0, 255).astype(np.uint8)

                mask_np = mask.squeeze(0).numpy()

                # Visualizing row elements
                row_ax = axes[idx] if self.num_val_samples > 1 else axes

                row_ax[0].imshow(img_np)
                row_ax[0].set_title("Original Image", color="white", fontsize=10)
                row_ax[0].axis("off")
                row_ax[0].set_facecolor("#0e1117")

                row_ax[1].imshow(mask_np, cmap="gray")
                row_ax[1].set_title("Ground Truth Mask", color="white", fontsize=10)
                row_ax[1].axis("off")
                row_ax[1].set_facecolor("#0e1117")

                row_ax[2].imshow(pred, cmap="gray")
                row_ax[2].set_title(f"Prediction (Epoch {epoch})", color="white", fontsize=10)
                row_ax[2].axis("off")
                row_ax[2].set_facecolor("#0e1117")

        plt.tight_layout()
        plot_path = os.path.join(self.pred_dir, f"validation_grid_epoch_{epoch:03d}.png")
        plt.savefig(plot_path, dpi=120, bbox_inches="tight", facecolor="#0e1117")
        plt.close()
