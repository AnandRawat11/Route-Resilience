"""
Route Resilience — Phase 1
src/standardization.py

Step 2: Image Standardization (In-Memory Streaming Mode)

Phase 1 Policy
--------------
  skip_save: true  →  No files are ever written to data/processed/.
                       Standardization is a pure in-memory transform
                       called directly by the tiler for each image.

  No _float32.npy files are generated.

Public API
----------
  standardize_image(img, config)  →  standardized uint8 RGB ndarray
  run_standardization(config, records)  →  generates preview only, returns []
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.core.io     import ensure_dir, load_image, load_mask, save_json, extract_geospatial_metadata
from src.core.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────
#  In-memory standardizer (pure utility — no I/O side effects)
# ─────────────────────────────────────────────────────────────

def standardize_image(
    img: np.ndarray,
    config: Dict[str, Any],
) -> np.ndarray:
    """
    Standardize a satellite image in memory.

    Performs:
        1. Ensure 3-channel uint8 RGB
        2. Optional resize (disabled by default)

    Args:
        img:    (H, W, C) image array loaded from disk.
        config: Full pipeline config dict.

    Returns:
        (H, W, 3) uint8 RGB ndarray — ready for tiling.
        The input array is NOT modified.
    """
    sc = config.get("standardization", {})
    resize_enabled: bool = sc.get("resize_enabled", False)
    max_dimension: int = sc.get("max_dimension", 4096)

    out = _ensure_rgb(img)
    if resize_enabled:
        out, _ = _maybe_resize(out, max_dimension)
    return out


def standardize_mask(
    mask: np.ndarray,
    config: Dict[str, Any],
    source_format: str = "grayscale",
) -> np.ndarray:
    """
    Standardize a road mask in memory.

    Supports:
        'grayscale'  — values > 127 → 255, rest → 0
        'deepglobe'  — RGB white (>127 on any channel) → 255 road, else 0

    Returns:
        (H, W) uint8 binary mask, values 0 or 255.
    """
    if source_format == "deepglobe":
        # DeepGlobe: RGB mask, white=road
        if mask.ndim == 3:
            binary = np.any(mask > 127, axis=2).astype(np.uint8) * 255
        else:
            binary = np.where(mask > 127, 255, 0).astype(np.uint8)
    else:
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) if mask.shape[2] == 3 else mask[:, :, 0]
        binary = np.where(mask > 127, 255, 0).astype(np.uint8)
    return binary


# ─────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────

def _ensure_rgb(img: np.ndarray) -> np.ndarray:
    """Return a 3-channel uint8 RGB copy of the image."""
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.shape[2] == 1:
        img = np.concatenate([img, img, img], axis=-1)
    elif img.shape[2] > 3:
        img = img[:, :, :3]
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    return img


def _maybe_resize(
    img: np.ndarray,
    max_dimension: int,
    interpolation: int = cv2.INTER_LANCZOS4,
) -> Tuple[np.ndarray, float]:
    """Resize image only if it exceeds max_dimension on either axis."""
    h, w = img.shape[:2]
    if max(h, w) <= max_dimension:
        return img, 1.0
    scale = max_dimension / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=interpolation), scale


# ─────────────────────────────────────────────────────────────
#  Comparison visualization (sample only — no dataset copies)
# ─────────────────────────────────────────────────────────────

class ImageStandardizer:
    """
    Thin wrapper around the in-memory standardization functions.

    In Phase 1 (skip_save=True), this class generates a visual comparison
    grid from a small sample of images and returns immediately.
    No files are written to data/processed/.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        sc = config.get("standardization", {})
        self.skip_save: bool = sc.get("skip_save", True)
        paths = config.get("paths", {})
        self.out_viz = ensure_dir(paths.get("visualizations", "outputs/visualizations"))
        logger.info(
            f"ImageStandardizer initialised (skip_save={self.skip_save}). "
            "No files will be written to data/processed/."
        )

    def process_all(self, records: List[Any]) -> List[Dict[str, Any]]:
        """
        In skip_save mode: generate a visual sample and return an empty list.

        The tiler will call standardize_image() directly for each image.
        """
        if self.skip_save:
            logger.info(
                "Standardization running in streaming mode (skip_save=True). "
                "No intermediate files created."
            )
            self.generate_comparison_grid(records, n_samples=6)
            return []

        # Legacy path (skip_save=False) — kept for non-Phase-1 use
        processed = []
        total = len(records)
        out_images = ensure_dir(
            self.config.get("paths", {}).get("processed_images", "data/processed/images")
        )
        out_masks = ensure_dir(
            self.config.get("paths", {}).get("processed_masks", "data/processed/masks")
        )
        out_meta = ensure_dir(
            self.config.get("paths", {}).get("processed_metadata", "data/processed/metadata")
        )
        from src.core.io import save_image
        for i, rec in enumerate(records):
            logger.debug(f"Standardizing [{i+1}/{total}]: {rec.image_id}")
            try:
                img = load_image(Path(rec.image_path))
                img = standardize_image(img, self.config)
                out_img = out_images / (rec.image_id + ".png")
                save_image(img, out_img)
                result = {
                    "image_id": rec.image_id,
                    "source_dataset": rec.source_dataset,
                    "processed_image": str(out_img),
                    "width": img.shape[1],
                    "height": img.shape[0],
                }
                if rec.has_mask and rec.mask_path:
                    from src.core.io import load_mask as _load_mask
                    mask = _load_mask(Path(rec.mask_path))
                    out_msk = out_masks / (rec.image_id + "_mask.png")
                    save_image(mask, out_msk, is_mask=True)
                    result["processed_mask"] = str(out_msk)
                processed.append(result)
            except Exception as exc:
                logger.error(f"Standardization failed for {rec.image_id}: {exc}")
        return processed

    def generate_comparison_grid(
        self,
        records: List[Any],
        n_samples: int = 6,
    ) -> None:
        """
        Generate a before/after comparison grid PNG (sample only).

        Saves to outputs/visualizations/standardization_comparison.png.
        """
        sample_records = [r for r in records if r.has_mask][:n_samples]
        if not sample_records:
            logger.warning("No masked records available for comparison grid.")
            return

        n = len(sample_records)
        fig, axes = plt.subplots(n, 3, figsize=(18, 5 * n))
        if n == 1:
            axes = axes[np.newaxis, :]

        fig.suptitle(
            "Standardization Preview — Original / Processed / Mask",
            fontsize=14, fontweight="bold", color="white", y=1.01,
        )
        fig.patch.set_facecolor("#0e1117")

        for row, rec in enumerate(sample_records):
            try:
                raw_img = load_image(Path(rec.image_path))
                proc_img = standardize_image(raw_img, self.config)
                raw_mask = None
                if rec.has_mask and rec.mask_path:
                    raw_mask = cv2.imread(str(rec.mask_path), cv2.IMREAD_GRAYSCALE)
                    if raw_mask is not None:
                        raw_mask = np.where(raw_mask > 127, 255, 0).astype(np.uint8)

                for ax in axes[row]:
                    ax.set_facecolor("#0e1117")
                    ax.axis("off")

                axes[row, 0].imshow(raw_img)
                axes[row, 0].set_title(f"{rec.image_id}\nRaw", color="white", fontsize=8)

                axes[row, 1].imshow(proc_img)
                axes[row, 1].set_title("Standardized (in-memory)", color="white", fontsize=8)

                if raw_mask is not None:
                    axes[row, 2].imshow(raw_mask, cmap="Greens", vmin=0, vmax=255)
                    axes[row, 2].set_title("Ground Truth Mask", color="white", fontsize=8)

                del raw_img, proc_img, raw_mask
            except Exception as exc:
                logger.debug(f"Comparison grid error for {rec.image_id}: {exc}")

        plt.tight_layout(pad=0.5)
        out_path = self.out_viz / "standardization_comparison.png"
        plt.savefig(str(out_path), dpi=120, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close()
        logger.info(f"Standardization comparison grid saved → {out_path}")


# ─────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────

def run_standardization(
    config: Dict[str, Any],
    records: List[Any],
) -> List[Dict[str, Any]]:
    """
    Main standardization entry point.

    In Phase 1 (skip_save=True):
        Generates a visual comparison grid only. Returns [].
        The tiler calls standardize_image() inline per image.

    Args:
        config:  Full pipeline config.
        records: List of ImageRecord objects from ingestion step.

    Returns:
        List of processed file dicts (empty in skip_save mode).
    """
    standardizer = ImageStandardizer(config)
    processed = standardizer.process_all(records)
    logger.info(
        "Standardization complete. "
        f"{'(streaming mode — no disk writes)' if not processed else f'{len(processed)} images processed.'}"
    )
    return processed
