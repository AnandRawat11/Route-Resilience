"""
Route Resilience — Phase 2
src/segmentation/losses/losses.py

Loss functions for binary segmentation, including Dice, Jaccard, and Combined Loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """
    Dice loss designed for binary segmentation. Operates on raw logits.
    """
    def __init__(self, eps: float = 1e-7) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)

        # Flatten tensors for 1D calculation
        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        union = probs.sum() + targets.sum()

        dice = (2.0 * intersection + self.eps) / (union + self.eps)
        return 1.0 - dice


class JaccardLoss(nn.Module):
    """
    Jaccard (IoU) loss designed for binary segmentation. Operates on raw logits.
    """
    def __init__(self, eps: float = 1e-7) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)

        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        total = probs.sum() + targets.sum()
        union = total - intersection

        jaccard = (intersection + self.eps) / (union + self.eps)
        return 1.0 - jaccard


class CombinedLoss(nn.Module):
    """
    Weighted combination of Binary Cross Entropy, Dice, and Jaccard losses.
    """
    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        jaccard_weight: float = 0.0
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.jaccard_weight = jaccard_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.jaccard = JaccardLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        if self.bce_weight > 0:
            loss += self.bce_weight * self.bce(logits, targets)
        if self.dice_weight > 0:
            loss += self.dice_weight * self.dice(logits, targets)
        if self.jaccard_weight > 0:
            loss += self.jaccard_weight * self.jaccard(logits, targets)
        return loss
