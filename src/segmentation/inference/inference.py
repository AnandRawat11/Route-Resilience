"""
Route Resilience — Phase 2
src/segmentation/inference/inference.py

Inference utilities for single-image and batch predictions, supporting thresholding,
visual overlay overlays, and export for Phase 3 topological graph reconstruction.
"""

from __future__ import annotations

import os
import glob
import logging
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

logger = logging.getLogger(__name__)


def visualize_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.4,
    color: Tuple[int, int, int] = (0, 255, 0)
) -> np.ndarray:
    """
    Overlay a binary road mask onto the original RGB image.

    Args:
        image: RGB image numpy array (H, W, 3).
        mask: Grayscale binary mask numpy array (H, W), values 0 or 255.
        alpha: Overlay transparency.
        color: RGB tuple for overlay color.

    Returns:
        RGB image array with overlaid mask.
    """
    overlay = image.copy()
    mask_bool = mask > 127
    overlay[mask_bool] = color
    return cv2.addWeighted(image, 1.0 - alpha, overlay, alpha, 0)


def predict_image(
    model: nn.Module,
    image_path: str,
    device: torch.device,
    threshold: float = 0.5,
    transform: Optional[Any] = None
) -> np.ndarray:
    """
    Predict a binary road mask for a single satellite image.

    Args:
        model: Trained segmentation model.
        image_path: Path to the input image file.
        device: Torch device (cuda/cpu).
        threshold: Probability threshold for road class.
        transform: Optional Albumentations transform.

    Returns:
        Binary mask numpy array (H, W) with values 0 (background) and 255 (road).
    """
    model.eval()

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Read image (BGR to RGB)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if transform is not None:
        augmented = transform(image=image)
        input_tensor = augmented["image"]
    else:
        # Default fallback standardization & normalization (ImageNet stats)
        t = image.transpose(2, 0, 1) / 255.0
        t = torch.from_numpy(t).float()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        input_tensor = (t - mean) / std

    input_batch = input_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_batch)
        probs = torch.sigmoid(logits)
        prob_mask = probs.squeeze(0).squeeze(0).cpu().numpy()

    binary_mask = (prob_mask > threshold).astype(np.uint8) * 255
    return binary_mask


def predict_folder(
    model: nn.Module,
    input_dir: str,
    output_dir: str,
    device: torch.device,
    threshold: float = 0.5,
    transform: Optional[Any] = None,
    batch_size: int = 8,
    save_overlays: bool = False
) -> None:
    """
    Batch prediction for all satellite images in a folder.
    Saves predicted binary road masks (and optional transparent overlays).

    Args:
        model: Trained segmentation model.
        input_dir: Folder containing source satellite images.
        output_dir: Target folder to save road masks (0 or 255 values).
        device: Torch device (cuda/cpu).
        threshold: Probability threshold.
        transform: Albumentations transform.
        batch_size: Inference batch size.
        save_overlays: If true, overlays predicted roads in green on original images.
    """
    os.makedirs(output_dir, exist_ok=True)
    if save_overlays:
        overlay_dir = os.path.join(output_dir, "overlays")
        os.makedirs(overlay_dir, exist_ok=True)

    # Gather image files
    image_extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff"]
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
        image_paths.extend(glob.glob(os.path.join(input_dir, f"*{ext.upper()}")))

    if not image_paths:
        logger.warning(f"No images found in folder: {input_dir}")
        return

    logger.info(f"Running batch inference on {len(image_paths)} images, batch size {batch_size}...")
    model.eval()

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = []
        batch_tensors = []

        for path in batch_paths:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            batch_images.append(img)

            if transform is not None:
                augmented = transform(image=img)
                batch_tensors.append(augmented["image"])
            else:
                t = img.transpose(2, 0, 1) / 255.0
                t = torch.from_numpy(t).float()
                t = (t - mean) / std
                batch_tensors.append(t)

        input_batch = torch.stack(batch_tensors).to(device)

        with torch.no_grad():
            logits = model(input_batch)
            probs = torch.sigmoid(logits)
            prob_masks = probs.squeeze(1).cpu().numpy()  # shape (B, H, W)

        for idx, (img_path, prob_mask) in enumerate(zip(batch_paths, prob_masks)):
            binary_mask = (prob_mask > threshold).astype(np.uint8) * 255
            basename = os.path.basename(img_path)
            stem, _ = os.path.splitext(basename)

            # Save binary road mask
            mask_save_path = os.path.join(output_dir, f"{stem}_mask.png")
            cv2.imwrite(mask_save_path, binary_mask)

            # Save overlay image (for developer inspection)
            if save_overlays:
                img_rgb = batch_images[idx]
                overlay = visualize_overlay(img_rgb, binary_mask)
                overlay_path = os.path.join(overlay_dir, f"{stem}_overlay.png")
                cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    logger.info(f"Batch prediction completed. Masks saved to: {output_dir}")
