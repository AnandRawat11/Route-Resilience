"""
Route Resilience — Phase 1
src/visualization/plots.py

Step 8: Pipeline Visualization
  - Raw image + ground truth mask
  - Before/after preprocessing comparison
  - Tile grid overlay
  - Augmentation preview grid
  - Occlusion comparison (original | occluded | diff)
  - Full pipeline side-by-side summary
  - All outputs saved as high-resolution PNG
  - Compatible with notebook and script usage
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.core.io     import ensure_dir, load_image, load_mask
from src.core.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────
#  PipelineVisualizer
# ─────────────────────────────────────────────────────────────

class PipelineVisualizer:
    """
    Centralized visualization engine for the Phase 1 pipeline.

    All methods save their output to outputs/visualizations/ and
    return the saved file path.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        vc = config.get("visualization", {})
        self.dpi: int = vc.get("dpi", 150)
        self.save_format: str = vc.get("save_format", "png")
        paths = config.get("paths", {})
        self.out_viz = ensure_dir(paths.get("visualizations", "outputs/visualizations"))
        self.dark_bg = "#0e1117"
        logger.info("PipelineVisualizer initialised.")

    # ── 1. Raw + Ground Truth ─────────────────────────────────

    def plot_sample(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        title: str = "Sample",
        filename: Optional[str] = None,
    ) -> Path:
        """
        Plot raw image alongside its ground-truth road mask.

        Returns path to saved file.
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.patch.set_facecolor(self.dark_bg)

        axes[0].imshow(image)
        axes[0].set_title("Satellite Image", color="white", fontsize=11)

        axes[1].imshow(mask, cmap="Greens", vmin=0, vmax=255)
        axes[1].set_title("Ground Truth Mask", color="white", fontsize=11)

        # Overlay
        overlay = _road_overlay(image, mask)
        axes[2].imshow(overlay)
        axes[2].set_title("Overlay", color="white", fontsize=11)

        for ax in axes:
            ax.set_facecolor(self.dark_bg)
            ax.axis("off")

        fig.suptitle(title, color="white", fontsize=14, fontweight="bold")
        plt.tight_layout(pad=0.5)
        out = self.out_viz / (filename or f"sample_{title.replace(' ', '_')}.png")
        plt.savefig(str(out), dpi=self.dpi, bbox_inches="tight", facecolor=self.dark_bg)
        plt.close()
        return out

    # ── 2. Preprocessing Comparison ───────────────────────────

    def plot_preprocessing_comparison(
        self,
        raw: np.ndarray,
        processed: np.ndarray,
        mask: Optional[np.ndarray] = None,
        title: str = "Preprocessing",
        filename: Optional[str] = None,
    ) -> Path:
        """
        Before/after preprocessing comparison with optional mask column.
        """
        n_cols = 4 if mask is not None else 2
        fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 5 + 1, 5))
        fig.patch.set_facecolor(self.dark_bg)
        if n_cols == 2:
            axes = list(axes)
        else:
            axes = list(axes)

        axes[0].imshow(raw)
        axes[0].set_title("Raw (original)", color="white", fontsize=10)

        axes[1].imshow(processed)
        axes[1].set_title("Processed (RGB uint8)", color="white", fontsize=10)

        if mask is not None and n_cols == 4:
            axes[2].imshow(mask, cmap="Greens", vmin=0, vmax=255)
            axes[2].set_title("Ground Truth Mask", color="white", fontsize=10)
            axes[3].imshow(_road_overlay(processed, mask))
            axes[3].set_title("Overlay", color="white", fontsize=10)

        for ax in axes:
            ax.set_facecolor(self.dark_bg)
            ax.axis("off")

        fig.suptitle(title, color="white", fontsize=13, fontweight="bold")
        plt.tight_layout(pad=0.5)
        out = self.out_viz / (filename or "preprocessing_comparison.png")
        plt.savefig(str(out), dpi=self.dpi, bbox_inches="tight", facecolor=self.dark_bg)
        plt.close()
        return out

    # ── 3. Tile Grid ──────────────────────────────────────────

    def plot_tile_samples(
        self,
        tiles: List[Any],
        n_rows: int = 3,
        n_cols: int = 4,
        filename: Optional[str] = None,
    ) -> Path:
        """
        Display a grid of random tile image+mask pairs.
        """
        samples = tiles[:n_rows * n_cols]
        fig, axes = plt.subplots(n_rows, n_cols * 2, figsize=(n_cols * 4, n_rows * 3))
        fig.patch.set_facecolor(self.dark_bg)
        axes = axes.reshape(n_rows, -1)

        for row in range(n_rows):
            for col in range(n_cols):
                idx = row * n_cols + col
                img_ax = axes[row, col * 2]
                msk_ax = axes[row, col * 2 + 1]
                for ax in [img_ax, msk_ax]:
                    ax.set_facecolor(self.dark_bg)
                    ax.axis("off")

                if idx >= len(samples):
                    continue

                tile = samples[idx]
                try:
                    img = load_image(Path(tile.image_tile_path))
                    img_ax.imshow(img)
                    img_ax.set_title(
                        f"{tile.source_dataset}\nroad={tile.road_pixel_pct:.1f}%",
                        color="white", fontsize=7,
                    )
                except Exception:
                    pass

                if tile.mask_tile_path and Path(tile.mask_tile_path).exists():
                    try:
                        mask = load_mask(Path(tile.mask_tile_path))
                        msk_ax.imshow(mask, cmap="Greens", vmin=0, vmax=255)
                        msk_ax.set_title("mask", color="white", fontsize=7)
                    except Exception:
                        pass

        fig.suptitle("Tile Samples (image | mask)", color="white", fontsize=13, fontweight="bold")
        plt.tight_layout(pad=0.3)
        out = self.out_viz / (filename or "tile_samples.png")
        plt.savefig(str(out), dpi=self.dpi, bbox_inches="tight", facecolor=self.dark_bg)
        plt.close()
        return out

    # ── 4. Full Pipeline Summary ──────────────────────────────

    def plot_pipeline_summary(
        self,
        record: Any,
        tile: Any,
        aug_image: Optional[np.ndarray] = None,
        aug_mask: Optional[np.ndarray] = None,
        occluded_image: Optional[np.ndarray] = None,
        filename: Optional[str] = None,
    ) -> Path:
        """
        6-panel summary showing all pipeline stages for one sample:
        Raw → Processed → Mask → Tile → Augmented → Occluded
        """
        fig = plt.figure(figsize=(24, 5))
        fig.patch.set_facecolor(self.dark_bg)
        gs = gridspec.GridSpec(1, 6, figure=fig, wspace=0.08)

        panels: List[Tuple[str, Any, Dict]] = []

        # 1 Raw
        try:
            raw = load_image(Path(record.image_path))
            panels.append(("Raw Satellite", raw, {}))
        except Exception:
            panels.append(("Raw Satellite", _placeholder(512, 512), {}))

        # 2 Processed
        proc_path = Path("data/processed/images") / (record.image_id + ".png")
        try:
            proc = load_image(proc_path) if proc_path.exists() else load_image(Path(record.image_path))
            panels.append(("Preprocessed RGB", proc, {}))
        except Exception:
            panels.append(("Preprocessed RGB", _placeholder(512, 512), {}))

        # 3 Mask
        if record.has_mask and record.mask_path:
            try:
                mask = load_mask(Path(record.mask_path))
                panels.append(("Ground Truth Mask", mask, {"cmap": "Greens", "vmin": 0, "vmax": 255}))
            except Exception:
                panels.append(("Ground Truth Mask", _placeholder(512, 512, gray=True), {}))
        else:
            panels.append(("Ground Truth Mask", _placeholder(512, 512, gray=True), {}))

        # 4 Tile
        try:
            tile_img = load_image(Path(tile.image_tile_path))
            panels.append(("Tiled (512×512)", tile_img, {}))
        except Exception:
            panels.append(("Tiled (512×512)", _placeholder(512, 512), {}))

        # 5 Augmented
        if aug_image is not None:
            panels.append(("Augmented", aug_image, {}))
        else:
            panels.append(("Augmented", _placeholder(512, 512), {}))

        # 6 Occluded
        if occluded_image is not None:
            panels.append(("Occluded", occluded_image, {}))
        else:
            panels.append(("Occluded", _placeholder(512, 512), {}))

        for i, (label, img, kwargs) in enumerate(panels):
            ax = fig.add_subplot(gs[i])
            ax.set_facecolor(self.dark_bg)
            ax.axis("off")
            ax.imshow(img, **kwargs)
            ax.set_title(label, color="white", fontsize=9, pad=4)

            if i < len(panels) - 1:
                ax.annotate(
                    "", xy=(1.08, 0.5), xycoords="axes fraction",
                    xytext=(1.02, 0.5), textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color="#7b68ee", lw=1.5),
                )

        fig.suptitle(
            f"Phase 1 Pipeline — {record.image_id}",
            color="white", fontsize=14, fontweight="bold", y=1.03
        )
        out = self.out_viz / (filename or f"pipeline_summary_{record.image_id}.png")
        plt.savefig(str(out), dpi=self.dpi, bbox_inches="tight", facecolor=self.dark_bg)
        plt.close()
        logger.info(f"Pipeline summary saved → {out}")
        return out

    # ── 5. Occlusion Gallery ──────────────────────────────────

    def plot_occlusion_gallery(
        self,
        occlusion_results: List[Any],
        n: int = 4,
        filename: Optional[str] = None,
    ) -> Optional[Path]:
        """
        Show n occlusion comparisons in a grid.
        """
        samples = occlusion_results[:n]
        if not samples:
            logger.warning("No occlusion results to plot.")
            return None

        fig, axes = plt.subplots(n, 3, figsize=(18, 5 * n))
        fig.patch.set_facecolor(self.dark_bg)
        if n == 1:
            axes = axes[np.newaxis, :]

        for row, result in enumerate(samples):
            titles = [
                "Original",
                f"Occluded ({result.occlusion_type})",
                f"Severity: {result.severity}  |  Road hidden: {result.road_coverage_pct:.1f}%",
            ]
            paths = [result.original_path, result.occluded_path, result.comparison_path]

            for col, (p, t) in enumerate(zip(paths, titles)):
                ax = axes[row, col]
                ax.set_facecolor(self.dark_bg)
                ax.axis("off")
                if col < 2:
                    try:
                        img = load_image(Path(p))
                        ax.imshow(img)
                    except Exception:
                        pass
                else:
                    try:
                        # comparison is already a 3-panel saved image — show full
                        comp = cv2.imread(str(p))
                        if comp is not None:
                            comp = cv2.cvtColor(comp, cv2.COLOR_BGR2RGB)
                            ax.imshow(comp)
                    except Exception:
                        pass
                ax.set_title(t, color="white", fontsize=9)

        fig.suptitle("Occlusion Simulation Gallery", color="white", fontsize=14, fontweight="bold")
        plt.tight_layout(pad=0.5)
        out = self.out_viz / (filename or "occlusion_gallery.png")
        plt.savefig(str(out), dpi=self.dpi, bbox_inches="tight", facecolor=self.dark_bg)
        plt.close()
        logger.info(f"Occlusion gallery saved → {out}")
        return out


# ─────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────

def _road_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return image with road pixels highlighted in red-orange."""
    overlay = image.copy().astype(np.float32)
    road = mask > 0
    overlay[road, 0] = np.clip(overlay[road, 0] * 0.4 + 200, 0, 255)
    overlay[road, 1] = np.clip(overlay[road, 1] * 0.3 + 60, 0, 255)
    overlay[road, 2] = np.clip(overlay[road, 2] * 0.3, 0, 255)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def _placeholder(h: int, w: int, gray: bool = False) -> np.ndarray:
    """Return a dark placeholder image for missing data."""
    if gray:
        return np.full((h, w), 40, dtype=np.uint8)
    return np.full((h, w, 3), 30, dtype=np.uint8)


# ─────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────

def run_visualization(
    config: Dict[str, Any],
    records: List[Any],
    tile_infos: List[Any],
    occlusion_results: Optional[List[Any]] = None,
) -> PipelineVisualizer:
    """
    Generate all pipeline visualizations.

    Returns:
        Configured PipelineVisualizer instance for notebook reuse.
    """
    viz = PipelineVisualizer(config)

    # Sample plots
    sample_records = [r for r in records if r.has_mask][:3]
    for rec in sample_records:
        try:
            img = load_image(Path(rec.image_path))
            mask = load_mask(Path(rec.mask_path))
            viz.plot_sample(img, mask, title=rec.image_id, filename=f"sample_{rec.image_id}.png")
        except Exception as exc:
            logger.warning(f"Could not plot sample {rec.image_id}: {exc}")

    # Tile samples
    if tile_infos:
        viz.plot_tile_samples(tile_infos, n_rows=3, n_cols=4)

    # Occlusion gallery
    if occlusion_results:
        viz.plot_occlusion_gallery(occlusion_results, n=min(4, len(occlusion_results)))

    return viz
