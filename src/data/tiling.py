"""
Route Resilience — Phase 1
src/tiling.py

Step 3: Image Tiling
  - Sliding-window 512×512 tile extraction with configurable stride
  - Reflection padding to capture edge regions
  - Information-content filter (road pixel % + image std)
  - Tile statistics report
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from src.core.io     import ensure_dir, load_image, load_mask, save_image, save_json
from src.core.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────
#  Data structures
# ─────────────────────────────────────────────────────────────

@dataclass
class TileInfo:
    """Metadata for a single extracted tile."""
    tile_id: str
    source_image_id: str
    source_dataset: str
    image_tile_path: str
    mask_tile_path: Optional[str]
    row: int           # tile row index
    col: int           # tile col index
    x_start: int       # pixel coords in source image
    y_start: int
    x_end: int
    y_end: int
    tile_width: int
    tile_height: int
    road_pixel_pct: float
    image_std: float
    kept: bool
    discard_reason: Optional[str]


# ─────────────────────────────────────────────────────────────
#  ImageTiler
# ─────────────────────────────────────────────────────────────

class ImageTiler:
    """
    Extracts 512×512 overlapping tiles from satellite images and masks.

    Tiles with:
      - road pixel % < min_road_pixel_pct  → discarded
      - image std dev < min_image_std      → discarded (near-blank)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        tc = config.get("tiling", {})
        self.tile_size: int = tc.get("tile_size", 512)
        self.stride: int = tc.get("stride", 256)
        self.pad_mode: str = tc.get("pad_mode", "reflect")
        self.pad_value: int = tc.get("pad_value", 0)
        self.min_road_pct: float = tc.get("min_road_pixel_pct", 0.5)
        self.min_img_std: float = tc.get("min_image_std", 5.0)

        paths = config.get("paths", {})
        self.out_img_dir = ensure_dir(paths.get("tiles_images", "data/tiles/images"))
        self.out_msk_dir = ensure_dir(paths.get("tiles_masks", "data/tiles/masks"))
        self.out_viz = ensure_dir(paths.get("visualizations", "outputs/visualizations"))
        self.report_dir = ensure_dir(paths.get("reports", "outputs/reports"))

        logger.info(
            f"ImageTiler initialised: tile={self.tile_size}px, "
            f"stride={self.stride}px, overlap={100*(1-self.stride/self.tile_size):.0f}%"
        )

    # ── Public API ────────────────────────────────────────────

    def tile_all(self, records: List[Any]) -> List[TileInfo]:
        """
        Tile all processed records (uses processed images if available, raw otherwise).

        Args:
            records: List of ImageRecord objects from ingestion.

        Returns:
            List of TileInfo for kept tiles.
        """
        all_tiles: List[TileInfo] = []
        total = len(records)
        for i, rec in enumerate(records):
            logger.debug(f"Tiling [{i+1}/{total}]: {rec.image_id}")
            tiles = self._tile_one(rec)
            all_tiles.extend(tiles)

        kept = [t for t in all_tiles if t.kept]
        discarded = [t for t in all_tiles if not t.kept]
        logger.info(
            f"Tiling complete: {len(all_tiles)} total tiles, "
            f"{len(kept)} kept, {len(discarded)} discarded."
        )

        stats = self._compute_tile_statistics(all_tiles)
        save_json(stats, self.report_dir / "tile_statistics.json")
        logger.info(f"Tile statistics saved → {self.report_dir / 'tile_statistics.json'}")

        return kept

    def generate_tile_grid_visualization(
        self,
        records: List[Any],
        tile_infos: List[TileInfo],
        n_samples: int = 3,
    ) -> None:
        """
        Visualize tiling pattern on source images.

        Shows the source image with tile bounding boxes overlaid (green=kept, red=discarded).
        """
        # Group tiles by source image
        from collections import defaultdict
        tiles_by_source: Dict[str, List[TileInfo]] = defaultdict(list)
        all_tiles_by_source: Dict[str, List[TileInfo]] = defaultdict(list)

        # Re-tile for visualization (to get discard info too)
        for rec in records[:n_samples]:
            all_t = self._tile_one(rec, save=False)
            for t in all_t:
                all_tiles_by_source[rec.image_id].append(t)

        for source_id, source_tiles in list(all_tiles_by_source.items())[:n_samples]:
            rec = next((r for r in records if r.image_id == source_id), None)
            if rec is None:
                continue
            self._visualize_tile_grid(rec, source_tiles)

    # ── Internal tiling ───────────────────────────────────────

    def _tile_one(self, rec: Any, save: bool = True) -> List[TileInfo]:
        """Extract tiles from one ImageRecord."""
        # Prefer processed image
        from pathlib import Path as P
        processed_img_path = Path("data/processed/images") / (rec.image_id + ".png")
        processed_msk_path = Path("data/processed/masks") / (rec.image_id + "_mask.png")

        img_path = processed_img_path if processed_img_path.exists() else Path(rec.image_path)
        msk_path = processed_msk_path if processed_msk_path.exists() else (
            Path(rec.mask_path) if rec.mask_path else None
        )

        try:
            img = load_image(img_path)
        except Exception as exc:
            logger.error(f"Cannot load image for tiling: {img_path}: {exc}")
            return []

        mask = None
        if msk_path and msk_path.exists():
            try:
                mask = load_mask(msk_path)
            except Exception as exc:
                logger.warning(f"Cannot load mask for tiling: {msk_path}: {exc}")

        tiles = []
        for tile_img, tile_mask, meta in self._sliding_window(img, mask, rec.image_id):
            # Compute statistics
            img_std = float(tile_img.std())
            road_pct = 0.0
            if tile_mask is not None:
                road_pct = float(np.sum(tile_mask > 0)) / (tile_mask.size) * 100.0

            # Filter
            kept = True
            discard_reason = None
            if road_pct < self.min_road_pct:
                kept = False
                discard_reason = f"road_pixel_pct={road_pct:.2f}% < {self.min_road_pct}%"
            elif img_std < self.min_img_std:
                kept = False
                discard_reason = f"image_std={img_std:.2f} < {self.min_img_std}"

            tile_img_path = None
            tile_msk_path = None

            if kept and save:
                tile_id = f"{rec.image_id}_r{meta['row']:03d}_c{meta['col']:03d}"
                tile_img_fname = tile_id + ".png"
                tile_msk_fname = tile_id + "_mask.png"

                save_image(tile_img, self.out_img_dir / tile_img_fname)
                tile_img_path = str(self.out_img_dir / tile_img_fname)

                if tile_mask is not None:
                    save_image(tile_mask, self.out_msk_dir / tile_msk_fname, is_mask=True)
                    tile_msk_path = str(self.out_msk_dir / tile_msk_fname)
            elif save:
                tile_id = f"{rec.image_id}_r{meta['row']:03d}_c{meta['col']:03d}_DISCARD"
            else:
                tile_id = f"{rec.image_id}_r{meta['row']:03d}_c{meta['col']:03d}"

            tiles.append(TileInfo(
                tile_id=tile_id,
                source_image_id=rec.image_id,
                source_dataset=getattr(rec, "source_dataset", "unknown"),
                image_tile_path=tile_img_path or "",
                mask_tile_path=tile_msk_path,
                row=meta["row"],
                col=meta["col"],
                x_start=meta["x_start"],
                y_start=meta["y_start"],
                x_end=meta["x_end"],
                y_end=meta["y_end"],
                tile_width=tile_img.shape[1],
                tile_height=tile_img.shape[0],
                road_pixel_pct=road_pct,
                image_std=img_std,
                kept=kept,
                discard_reason=discard_reason,
            ))

        return tiles

    def _sliding_window(
        self,
        img: np.ndarray,
        mask: Optional[np.ndarray],
        image_id: str,
    ) -> Generator[Tuple[np.ndarray, Optional[np.ndarray], Dict], None, None]:
        """
        Yield (tile_img, tile_mask, metadata) for every sliding window position.
        Pads the image so all positions are covered.
        """
        h, w = img.shape[:2]
        ts = self.tile_size
        st = self.stride

        # Compute padded size
        pad_h = max(0, ts - (h - ts) % st) if h > ts else ts - h
        pad_w = max(0, ts - (w - ts) % st) if w > ts else ts - w

        if self.pad_mode == "reflect":
            img_padded = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            if mask is not None:
                mask_padded = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="reflect")
            else:
                mask_padded = None
        else:
            img_padded = np.pad(
                img,
                ((0, pad_h), (0, pad_w), (0, 0)),
                mode="constant",
                constant_values=self.pad_value,
            )
            mask_padded = None
            if mask is not None:
                mask_padded = np.pad(
                    mask,
                    ((0, pad_h), (0, pad_w)),
                    mode="constant",
                    constant_values=0,
                )

        ph, pw = img_padded.shape[:2]
        row_idx = 0
        for y in range(0, ph - ts + 1, st):
            col_idx = 0
            for x in range(0, pw - ts + 1, st):
                tile_img = img_padded[y:y+ts, x:x+ts]
                tile_msk = mask_padded[y:y+ts, x:x+ts] if mask_padded is not None else None

                meta = {
                    "row": row_idx,
                    "col": col_idx,
                    "x_start": x,
                    "y_start": y,
                    "x_end": x + ts,
                    "y_end": y + ts,
                }
                yield tile_img, tile_msk, meta
                col_idx += 1
            row_idx += 1

    # ── Statistics ────────────────────────────────────────────

    def _compute_tile_statistics(self, tiles: List[TileInfo]) -> Dict[str, Any]:
        kept = [t for t in tiles if t.kept]
        discarded = [t for t in tiles if not t.kept]
        road_pcts = [t.road_pixel_pct for t in kept]

        by_source: Dict[str, Dict] = {}
        for t in tiles:
            s = t.source_dataset
            if s not in by_source:
                by_source[s] = {"total": 0, "kept": 0, "discarded": 0}
            by_source[s]["total"] += 1
            if t.kept:
                by_source[s]["kept"] += 1
            else:
                by_source[s]["discarded"] += 1

        return {
            "total_tiles_generated": len(tiles),
            "tiles_kept": len(kept),
            "tiles_discarded": len(discarded),
            "keep_rate_pct": round(len(kept) / len(tiles) * 100, 2) if tiles else 0,
            "tile_size": self.tile_size,
            "stride": self.stride,
            "overlap_pct": round((1 - self.stride / self.tile_size) * 100, 1),
            "road_pixel_pct_in_kept": {
                "mean": round(float(np.mean(road_pcts)), 4) if road_pcts else 0,
                "std": round(float(np.std(road_pcts)), 4) if road_pcts else 0,
                "min": round(float(np.min(road_pcts)), 4) if road_pcts else 0,
                "max": round(float(np.max(road_pcts)), 4) if road_pcts else 0,
            },
            "discard_reasons": _count_reasons([t.discard_reason for t in discarded]),
            "per_source": by_source,
        }

    # ── Visualization ─────────────────────────────────────────

    def _visualize_tile_grid(self, rec: Any, tiles: List[TileInfo]) -> None:
        """Overlay tile bounding boxes on source image and save."""
        try:
            img_path = Path("data/processed/images") / (rec.image_id + ".png")
            if not img_path.exists():
                img_path = Path(rec.image_path)
            img = load_image(img_path)
        except Exception:
            return

        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#0e1117")
        ax.imshow(img)
        ax.set_title(
            f"Tile Grid: {rec.image_id}  "
            f"(green=kept, red=discarded)",
            color="white", fontsize=11
        )
        ax.axis("off")

        for t in tiles:
            color = "#00ff88" if t.kept else "#ff4444"
            alpha = 0.25 if t.kept else 0.15
            rect = patches.Rectangle(
                (t.x_start, t.y_start),
                t.tile_width, t.tile_height,
                linewidth=0.8, edgecolor=color,
                facecolor=color, alpha=alpha,
            )
            ax.add_patch(rect)

        kept_n = sum(1 for t in tiles if t.kept)
        disc_n = sum(1 for t in tiles if not t.kept)
        ax.text(
            0.01, 0.98,
            f"Kept: {kept_n}  Discarded: {disc_n}",
            transform=ax.transAxes,
            color="white", fontsize=10, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#333", alpha=0.8),
        )

        out = self.out_viz / f"tile_grid_{rec.image_id}.png"
        plt.tight_layout()
        plt.savefig(str(out), dpi=120, bbox_inches="tight", facecolor="#0e1117")
        plt.close()
        logger.debug(f"Tile grid visualization saved → {out}")


# ─────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────

def _count_reasons(reasons: List[Optional[str]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for r in reasons:
        key = r or "unknown"
        # Simplify to category
        if "road_pixel" in key:
            key = "low_road_density"
        elif "image_std" in key:
            key = "near_blank_image"
        counts[key] = counts.get(key, 0) + 1
    return counts


# ─────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────

def run_tiling(
    config: Dict[str, Any],
    records: List[Any],
) -> List[TileInfo]:
    """
    Main tiling entry point.

    Args:
        config:  Full pipeline config.
        records: List of ImageRecord objects from ingestion.

    Returns:
        List of TileInfo for kept tiles.
    """
    tiler = ImageTiler(config)
    kept_tiles = tiler.tile_all(records)
    tiler.generate_tile_grid_visualization(records, kept_tiles, n_samples=3)
    return kept_tiles
