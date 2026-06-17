"""
Route Resilience — Phase 1
src/augmentation.py

Step 5: Data Augmentation
  - Albumentations pipeline (spatial + pixel transforms)
  - Applied consistently to image and mask pairs
  - Serializable pipeline configuration
  - Preview grid generation
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import albumentations as A
from albumentations.core.composition import Compose

from src.core.io     import ensure_dir, load_image, load_mask, save_json
from src.core.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────
#  AugmentationPipeline
# ─────────────────────────────────────────────────────────────

class AugmentationPipeline:
    """
    Albumentations-based augmentation pipeline for satellite road imagery.

    Spatial transforms are applied to both image and mask simultaneously.
    Pixel transforms are applied to the image only.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        ac = config.get("augmentation", {})
        self.enabled: bool = ac.get("enabled", True)
        self.preview_count: int = ac.get("preview_count", 8)
        self.pipeline_cfg: Dict = ac.get("pipeline", {})

        paths = config.get("paths", {})
        self.out_viz = ensure_dir(paths.get("visualizations", "outputs/visualizations"))
        self.report_dir = ensure_dir(paths.get("reports", "outputs/reports"))

        self.transform: Compose = self._build_pipeline()
        self._save_pipeline_config()

        logger.info(f"AugmentationPipeline built ({len(self.transform.transforms)} transforms).")

    # ── Public API ────────────────────────────────────────────

    def augment(
        self, image: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply augmentation to a single image/mask pair.

        Args:
            image: (H, W, 3) uint8 RGB image.
            mask:  (H, W) uint8 binary mask (0 or 255).

        Returns:
            (augmented_image, augmented_mask) with same dtypes and shapes.
        """
        if not self.enabled:
            return image, mask

        result = self.transform(image=image, mask=mask)
        return result["image"], result["mask"]

    def generate_preview(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        image_id: str = "sample",
    ) -> None:
        """
        Generate an augmentation preview grid.

        Shows original + N augmented variants side-by-side.
        Saves to outputs/visualizations/augmentation_preview_{image_id}.png
        """
        n = self.preview_count
        ncols = 4
        nrows = (n + 1 + ncols - 1) // ncols  # +1 for original

        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
        fig.patch.set_facecolor("#0e1117")
        axes = axes.flatten()

        # Original
        self._render_pair(axes[0], image, mask, "Original")

        # Augmented variants
        for i in range(1, n + 1):
            aug_img, aug_mask = self.augment(image.copy(), mask.copy())
            self._render_pair(axes[i], aug_img, aug_mask, f"Augmented #{i}")

        # Hide unused axes
        for j in range(n + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(
            f"Augmentation Preview — {image_id}",
            color="white", fontsize=14, fontweight="bold"
        )
        plt.tight_layout(pad=0.5)
        out = self.out_viz / f"augmentation_preview_{image_id}.png"
        plt.savefig(str(out), dpi=150, bbox_inches="tight", facecolor="#0e1117")
        plt.close()
        logger.info(f"Augmentation preview saved → {out}")

    def generate_previews_from_split(
        self,
        tile_infos: List[Any],
        n_samples: int = 2,
    ) -> None:
        """Generate augmentation previews from a list of tile paths."""
        samples = tile_infos[:n_samples]
        for tile in samples:
            try:
                img = load_image(Path(tile.image_tile_path))
                mask_path = tile.mask_tile_path
                mask = load_mask(Path(mask_path)) if mask_path and Path(mask_path).exists() else np.zeros(
                    img.shape[:2], dtype=np.uint8
                )
                self.generate_preview(img, mask, image_id=tile.tile_id[:30])
            except Exception as exc:
                logger.warning(f"Could not generate preview for {tile.tile_id}: {exc}")

    # ── Pipeline construction ─────────────────────────────────

    def _build_pipeline(self) -> Compose:
        """Build Albumentations Compose pipeline from config."""
        pc = self.pipeline_cfg

        def p(key: str, default: float = 0.5) -> float:
            return pc.get(key, {}).get("p", default)

        transforms: List[Any] = []

        # Spatial transforms (applied to image + mask)
        if "horizontal_flip" in pc:
            transforms.append(A.HorizontalFlip(p=p("horizontal_flip")))

        if "vertical_flip" in pc:
            transforms.append(A.VerticalFlip(p=p("vertical_flip")))

        if "random_rotate_90" in pc:
            transforms.append(A.RandomRotate90(p=p("random_rotate_90")))

        if "shift_scale_rotate" in pc:
            ssr = pc["shift_scale_rotate"]
            transforms.append(A.ShiftScaleRotate(
                shift_limit=ssr.get("shift_limit", 0.0625),
                scale_limit=ssr.get("scale_limit", 0.2),
                rotate_limit=ssr.get("rotate_limit", 45),
                border_mode=cv2.BORDER_REFLECT_101,
                p=ssr.get("p", 0.5),
            ))

        # Pixel transforms (image only, wrapped in A.Lambda or direct)
        if "random_brightness_contrast" in pc:
            rbc = pc["random_brightness_contrast"]
            transforms.append(A.RandomBrightnessContrast(
                brightness_limit=rbc.get("brightness_limit", 0.3),
                contrast_limit=rbc.get("contrast_limit", 0.3),
                p=rbc.get("p", 0.5),
            ))

        if "gauss_noise" in pc:
            gn = pc["gauss_noise"]
            var_limit = gn.get("var_limit", [10, 50])
            if isinstance(var_limit, list):
                var_limit = tuple(var_limit)
            transforms.append(A.GaussNoise(
                var_limit=var_limit,
                p=gn.get("p", 0.3),
            ))

        if "blur" in pc:
            bl = pc["blur"]
            transforms.append(A.Blur(
                blur_limit=bl.get("blur_limit", 3),
                p=bl.get("p", 0.3),
            ))

        if "clahe" in pc:
            cl = pc["clahe"]
            transforms.append(A.CLAHE(
                clip_limit=cl.get("clip_limit", 4.0),
                p=cl.get("p", 0.3),
            ))

        return A.Compose(transforms)

    def _save_pipeline_config(self) -> None:
        """Save the serialized Albumentations pipeline to JSON."""
        try:
            serialized = A.to_dict(self.transform)
            save_json(serialized, self.report_dir / "augmentation_pipeline.json")
            logger.debug("Augmentation pipeline config saved.")
        except Exception as exc:
            logger.warning(f"Could not serialize augmentation pipeline: {exc}")

    # ── Visualization helpers ─────────────────────────────────

    @staticmethod
    def _render_pair(
        ax: Any,
        image: np.ndarray,
        mask: np.ndarray,
        title: str,
    ) -> None:
        """Render image with road overlay onto a single axes."""
        ax.set_facecolor("#0e1117")
        ax.axis("off")

        display = image.copy()
        if mask is not None and mask.max() > 0:
            road_px = mask > 0
            # Red-tinted overlay on road pixels
            display[road_px, 0] = np.clip(
                display[road_px, 0].astype(np.float32) * 0.5 + 127, 0, 255
            ).astype(np.uint8)
            display[road_px, 1] = (display[road_px, 1].astype(np.float32) * 0.5).astype(np.uint8)
            display[road_px, 2] = (display[road_px, 2].astype(np.float32) * 0.5).astype(np.uint8)

        ax.imshow(display)
        ax.set_title(title, color="white", fontsize=8, pad=3)


# ─────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────

def run_augmentation(
    config: Dict[str, Any],
    train_tiles: List[Any],
) -> AugmentationPipeline:
    """
    Build augmentation pipeline and generate preview images.

    Note: The pipeline is returned for use during training (DataLoader).
    Augmentation is NOT applied to the saved files — it runs on-the-fly
    during model training for maximum data efficiency.

    Args:
        config:      Full pipeline config.
        train_tiles: List of TileInfo from the training split.

    Returns:
        Configured AugmentationPipeline instance.
    """
    pipeline = AugmentationPipeline(config)
    pipeline.generate_previews_from_split(train_tiles, n_samples=2)
    logger.info("Augmentation pipeline ready (applied on-the-fly during training).")
    return pipeline
