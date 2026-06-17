"""
Route Resilience — Phase 1
src/standardization.py

Step 2: Image Standardization
  - Convert GeoTIFF / multi-band rasters to RGB uint8
  - Preserve and save geospatial metadata
  - Normalize pixel values to [0,1] float32 for model consumption
  - Resize images if configured
  - Generate before/after comparison visualizations
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.core.io     import ensure_dir, load_image, load_mask, save_image, save_json, extract_geospatial_metadata
from src.core.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────
#  ImageStandardizer
# ─────────────────────────────────────────────────────────────

class ImageStandardizer:
    """
    Standardizes satellite images and masks for the training pipeline.

    Outputs:
        data/processed/images/   — uint8 RGB PNGs
        data/processed/masks/    — uint8 binary PNGs (0 or 255)
        data/processed/metadata/ — per-image JSON with geospatial metadata
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        sc = config.get("standardization", {})
        self.percentile_low: float = sc.get("percentile_low", 2)
        self.percentile_high: float = sc.get("percentile_high", 98)
        self.resize_enabled: bool = sc.get("resize_enabled", False)
        self.max_dimension: int = sc.get("max_dimension", 4096)
        self.normalize_float: bool = sc.get("normalize_float", True)

        paths = config.get("paths", {})
        self.out_images = ensure_dir(paths.get("processed_images", "data/processed/images"))
        self.out_masks = ensure_dir(paths.get("processed_masks", "data/processed/masks"))
        self.out_meta = ensure_dir(paths.get("processed_metadata", "data/processed/metadata"))
        self.out_viz = ensure_dir(
            paths.get("visualizations", "outputs/visualizations")
        )

        logger.info("ImageStandardizer initialised.")

    # ── Public API ────────────────────────────────────────────

    def process_all(
        self, records: List[Any]
    ) -> List[Dict[str, Any]]:
        """
        Process all ImageRecords.

        Args:
            records: List of ingestion.ImageRecord objects.

        Returns:
            List of dicts with processed paths and metadata.
        """
        processed = []
        total = len(records)
        for i, rec in enumerate(records):
            logger.debug(f"Standardizing [{i+1}/{total}]: {rec.image_id}")
            result = self._process_record(rec)
            if result is not None:
                processed.append(result)
        logger.info(f"Standardized {len(processed)}/{total} images successfully.")
        return processed

    def generate_comparison_grid(
        self,
        records: List[Any],
        n_samples: int = 6,
    ) -> None:
        """
        Generate a before/after comparison grid PNG.

        Saves to outputs/visualizations/standardization_comparison.png
        """
        from src.data.ingestion import ImageRecord  # local import to avoid circular

        sample_records = [r for r in records if r.has_mask][:n_samples]
        if not sample_records:
            logger.warning("No records with masks available for comparison grid.")
            return

        n = len(sample_records)
        fig, axes = plt.subplots(n, 4, figsize=(22, 5 * n))
        if n == 1:
            axes = axes[np.newaxis, :]

        fig.suptitle(
            "Standardization Pipeline — Before / After Comparison",
            fontsize=16, fontweight="bold", color="white", y=1.01
        )
        fig.patch.set_facecolor("#0e1117")

        for row, rec in enumerate(sample_records):
            proc_img_path = self.out_images / (rec.image_id + ".png")
            proc_msk_path = self.out_masks / (rec.image_id + "_mask.png")

            # Load raw
            try:
                raw_img = load_image(Path(rec.image_path))
            except Exception:
                continue

            # Load processed
            proc_img = None
            if proc_img_path.exists():
                try:
                    proc_img = load_image(proc_img_path)
                except Exception:
                    pass

            # Load mask
            raw_mask = None
            if rec.has_mask and rec.mask_path:
                try:
                    raw_mask = load_mask(Path(rec.mask_path))
                except Exception:
                    pass

            _plot_comparison_row(axes[row], raw_img, proc_img, raw_mask, rec.image_id)

        plt.tight_layout(pad=0.5)
        out_path = self.out_viz / "standardization_comparison.png"
        plt.savefig(str(out_path), dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close()
        logger.info(f"Comparison grid saved → {out_path}")

    # ── Internal helpers ──────────────────────────────────────

    def _process_record(self, rec: Any) -> Optional[Dict[str, Any]]:
        """Process a single ImageRecord."""
        try:
            img_path = Path(rec.image_path)

            # 1. Load raw image
            img = load_image(img_path)

            # 2. Stretch to uint8 if needed (already handled in load_image,
            #    but we also enforce 3-channel)
            img = self._ensure_rgb(img)

            # 3. Resize if enabled and image exceeds max_dimension
            img, scale = self._maybe_resize(img, interpolation=cv2.INTER_LANCZOS4)

            # 4. Save uint8 RGB PNG
            out_img_path = self.out_images / (rec.image_id + ".png")
            save_image(img, out_img_path)

            # 5. Process mask if available
            mask_out_path = None
            if rec.has_mask and rec.mask_path:
                mask = load_mask(Path(rec.mask_path))
                if scale != 1.0:
                    new_h = int(mask.shape[0] * scale)
                    new_w = int(mask.shape[1] * scale)
                    mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                mask_out_path = self.out_masks / (rec.image_id + "_mask.png")
                save_image(mask, mask_out_path, is_mask=True)

            # 6. Save float32 normalized version (for PyTorch DataLoader)
            if self.normalize_float:
                img_float = img.astype(np.float32) / 255.0
                float_path = self.out_images / (rec.image_id + "_float32.npy")
                np.save(str(float_path), img_float)

            # 7. Preserve geospatial metadata
            geo = extract_geospatial_metadata(img_path)
            geo["processed_image_path"] = str(out_img_path)
            geo["processed_mask_path"] = str(mask_out_path) if mask_out_path else None
            geo["scale_applied"] = scale
            geo["standardized_width"] = img.shape[1]
            geo["standardized_height"] = img.shape[0]
            meta_path = self.out_meta / (rec.image_id + "_geo.json")
            save_json(geo, meta_path)

            return {
                "image_id": rec.image_id,
                "source_dataset": rec.source_dataset,
                "processed_image": str(out_img_path),
                "processed_mask": str(mask_out_path) if mask_out_path else None,
                "geo_metadata": str(meta_path),
                "width": img.shape[1],
                "height": img.shape[0],
                "scale_applied": scale,
                "is_georeferenced": geo.get("is_georeferenced", False),
            }

        except Exception as exc:
            logger.error(f"Standardization failed for {rec.image_id}: {exc}")
            return None

    def _ensure_rgb(self, img: np.ndarray) -> np.ndarray:
        """Ensure the image is exactly 3-channel uint8 RGB."""
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
        self, img: np.ndarray, interpolation: int = cv2.INTER_LANCZOS4
    ) -> Tuple[np.ndarray, float]:
        """Resize image if resize_enabled and dimension exceeds max_dimension."""
        if not self.resize_enabled:
            return img, 1.0
        h, w = img.shape[:2]
        if max(h, w) <= self.max_dimension:
            return img, 1.0
        scale = self.max_dimension / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
        logger.debug(f"Resized {w}×{h} → {new_w}×{new_h} (scale={scale:.3f})")
        return img, scale


# ─────────────────────────────────────────────────────────────
#  Visualization helpers
# ─────────────────────────────────────────────────────────────

def _plot_comparison_row(
    axes: np.ndarray,
    raw_img: np.ndarray,
    proc_img: Optional[np.ndarray],
    mask: Optional[np.ndarray],
    title: str,
) -> None:
    """Fill one row of the comparison grid."""
    dark_bg = "#0e1117"

    titles = ["Raw Image", "Processed (RGB)", "Ground Truth Mask", "Overlay"]
    for ax in axes:
        ax.set_facecolor(dark_bg)
        ax.axis("off")

    # Raw
    axes[0].imshow(raw_img)
    axes[0].set_title(f"{title}\n{titles[0]}", color="white", fontsize=9)

    # Processed
    if proc_img is not None:
        axes[1].imshow(proc_img)
        axes[1].set_title(titles[1], color="white", fontsize=9)
    else:
        axes[1].text(0.5, 0.5, "Not processed", ha="center", va="center",
                     color="gray", transform=axes[1].transAxes)

    # Mask
    if mask is not None:
        axes[2].imshow(mask, cmap="Greens", vmin=0, vmax=255)
        axes[2].set_title(titles[2], color="white", fontsize=9)

    # Overlay: processed image + mask
    if proc_img is not None and mask is not None:
        overlay = proc_img.copy()
        road_px = mask > 0
        overlay[road_px, 0] = np.clip(overlay[road_px, 0] * 0.5 + 127, 0, 255).astype(np.uint8)
        overlay[road_px, 1] = np.clip(overlay[road_px, 1] * 0.5, 0, 255).astype(np.uint8)
        overlay[road_px, 2] = np.clip(overlay[road_px, 2] * 0.5, 0, 255).astype(np.uint8)
        axes[3].imshow(overlay)
        axes[3].set_title(titles[3], color="white", fontsize=9)


# ─────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────

def run_standardization(
    config: Dict[str, Any],
    records: List[Any],
) -> List[Dict[str, Any]]:
    """
    Main standardization entry point.

    Args:
        config:  Full pipeline config.
        records: List of ImageRecord objects from ingestion step.

    Returns:
        List of dicts describing processed files.
    """
    standardizer = ImageStandardizer(config)
    processed = standardizer.process_all(records)

    # Generate comparison grid (sample up to 6 pairs)
    standardizer.generate_comparison_grid(records, n_samples=6)

    logger.info(
        f"Standardization complete: {len(processed)} images processed."
    )
    return processed
