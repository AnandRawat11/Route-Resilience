"""
Route Resilience — Phase 2
src/segmentation/evaluation/metrics.py

Evaluation metrics for binary segmentation, operating directly on PyTorch tensors.
"""

from __future__ import annotations

from typing import Dict
import torch


def compute_batch_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Computes binary segmentation metrics: Dice (F1), IoU, Precision, and Recall
    for a batch of predictions and targets.

    Args:
        logits: Model output raw predictions (shape: B, C, H, W or B, H, W)
        targets: Ground truth binary mask (same shape, values 0.0 or 1.0)
        threshold: Probability threshold for classification

    Returns:
        Dictionary of calculated metrics as floats.
    """
    # Apply sigmoid and threshold to get binary predictions
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    # Flatten for 1D calculation
    preds = preds.view(-1)
    targets = targets.view(-1)

    tp = (preds * targets).sum()
    fp = (preds * (1.0 - targets)).sum()
    fn = ((1.0 - preds) * targets).sum()

    eps = 1e-7

    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f1 = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)

    return {
        "precision": precision.item(),
        "recall": recall.item(),
        "f1_score": f1.item(),
        "dice": f1.item(),  # Dice is mathematically equivalent to binary F1
        "iou": iou.item()
    }
