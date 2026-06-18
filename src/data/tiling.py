"""
Route Resilience — Phase 1
src/tiling.py

Step 3: Image Tiling (Lazy, Split-Aware, Storage-Efficient)

Phase 1 Design
--------------
  * Reads raw images directly — no dependency on data/processed/.
  * Standardizes each image in-memory before tiling (calls standardize_image).
  * Processes one image at a time with explicit del + gc.collect().
  * Outputs to data/tiles/{split}/images/ and data/tiles/{split}/masks/.
  * Phase 1: only the 'train' split is tiled (valid/test have no masks).
  * Saves image tiles as JPEG (quality=85), mask tiles as PNG (level=9).
  * Enforces max_tiles_per_image to cap storage growth.
  * Checks free disk space before every image (storage_guard_gb threshold).
"""

from __future__ import annotations

import gc
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from src.core.io     import ensure_dir, load_image, save_json
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
    split: str                      # 'train' | 'valid' | 'test'
    image_tile_path: str
    mask_tile_path: Optional[str]
    row: int
    col: int
    x_start: int
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
#  Storage guard
# ─────────────────────────────────────────────────────────────

def _check_storage(guard_gb: float) -> None:
    """
    Raise RuntimeError if free disk space falls below *guard_gb* GB.
    Called before processing each image.
    """
    free_gb = shutil.disk_usage(".").free / (1024 ** 3)
    if free_gb < guard_gb:
        raise RuntimeError(
            f"\n⛔  LOW DISK SPACE: {free_gb:.1f} GB free < {guard_gb} GB threshold.\n"
            "  Pipeline halted to prevent storage overflow.\n"
            "  Free up space and re-run."
        )


# ─────────────────────────────────────────────────────────────
#  ImageTiler
# ─────────────────────────────────────────────────────────────

class ImageTiler:
    """
    Lazy, split-aware 512×512 tile extractor.

    Memory model
    ------------
    For each source image:
        1. Load image + mask
        2. Standardize in memory
        3. Slide window → compute stats
        4. Save kept tiles (JPEG image, PNG mask)
        5. del arrays; gc.collect()

    Peak memory ≈ (2 × image size) + (max_tiles_per_image × tile_size²).
    For 1024×1024 RGB that is ≈ 6 MB + 6 × 0.75 MB ≈ 11 MB per image.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        tc = config.get("tiling", {})
        self.tile_size: int   = tc.get("tile_size", 512)
        self.stride: int      = tc.get("stride", 256)
        self.pad_mode: str    = tc.get("pad_mode", "reflect")
        self.pad_value: int   = tc.get("pad_value", 0)
        self.min_road_pct: float   = tc.get("min_road_pixel_pct", 0.01)
        self.min_img_std: float    = tc.get("min_image_std", 5.0)
        self.max_tiles: int        = tc.get("max_tiles_per_image", 6)

        # Compression settings
        self.img_fmt: str     = tc.get("image_format", "jpg")
        self.img_quality: int = tc.get("image_quality", 85)
        self.msk_fmt: str     = tc.get("mask_format", "png")
        self.msk_compress: int = tc.get("mask_compression", 9)

        # Storage guard (GB)
        self.storage_guard_gb: float = tc.get("storage_guard_gb", 15.0)

        paths = config.get("paths", {})
        self.tiles_root = ensure_dir(paths.get("tiles_root", "data/tiles"))
        self.out_viz    = ensure_dir(paths.get("visualizations", "outputs/visualizations"))
        self.report_dir = ensure_dir(paths.get("reports", "outputs/reports"))

        logger.info(
            f"ImageTiler: tile={self.tile_size}px, stride={self.stride}px, "
            f"max_per_image={self.max_tiles}, "
            f"img_fmt={self.img_fmt}@{self.img_quality}, "
            f"mask_fmt={self.msk_fmt}@compress={self.msk_compress}"
        )

    # ── Public API ────────────────────────────────────────────

    def tile_all(self, records: List[Any]) -> List[TileInfo]:
        """
        Tile all records lazily, one at a time.

        Returns:
            List of TileInfo for kept tiles.
        """
        all_kept: List[TileInfo] = []
        total = len(records)
        total_generated = 0
        total_discarded = 0

        logger.info(f"Starting lazy tiling of {total} records…")

        for i, rec in enumerate(records):
            # Storage guard before each image
            try:
                _check_storage(self.storage_guard_gb)
            except RuntimeError as exc:
                logger.error(str(exc))
                break

            split = getattr(rec, "split", "train") or "train"
            logger.info(
                f"  [{i+1:>5}/{total}] {rec.image_id}  split={split}"
            )

            kept, generated, discarded = self._tile_one(rec, split)
            all_kept.extend(kept)
            total_generated += generated
            total_discarded += discarded

            # Explicit memory cleanup
            del kept
            gc.collect()

        logger.info(
            f"Tiling complete: {total_generated} generated, "
            f"{len(all_kept)} kept, {total_discarded} discarded."
        )

        stats = self._compute_tile_statistics(all_kept, total_generated, total_discarded)
        save_json(stats, self.report_dir / "tile_statistics.json")
        logger.info(f"Tile statistics → {self.report_dir / 'tile_statistics.json'}")
        return all_kept

    def generate_tile_grid_visualization(
        self,
        records: List[Any],
        tile_infos: List[TileInfo],
        n_samples: int = 3,
    ) -> None:
        """Visualize tiling pattern on a small sample of source images."""
        for rec in records[:n_samples]:
            try:
                self._visualize_tile_grid(rec, tile_infos)
            except Exception as exc:
                logger.debug(f"Visualization skipped for {rec.image_id}: {exc}")

    # ── Internal tiling ───────────────────────────────────────

    def _tile_one(
        self, rec: Any, split: str
    ) -> Tuple[List[TileInfo], int, int]:
        """
        Tile a single image record. Returns (kept_tiles, n_generated, n_discarded).
        """
        from src.data.standardization import standardize_image, standardize_mask

        img_path = Path(rec.image_path)
        msk_path = Path(rec.mask_path) if rec.mask_path else None

        # Output directories for this split
        out_img_dir = ensure_dir(self.tiles_root / split / "images")
        out_msk_dir = ensure_dir(self.tiles_root / split / "masks") if msk_path else None

        # Load + standardize image
        try:
            img_raw = load_image(img_path)
            img = standardize_image(img_raw, self.config)
            del img_raw
        except Exception as exc:
            logger.error(f"Cannot load/standardize {rec.image_id}: {exc}")
            return [], 0, 0

        # Load + standardize mask
        mask: Optional[np.ndarray] = None
        if msk_path and msk_path.exists():
            try:
                # DeepGlobe masks are RGB — load as grayscale
                msk_raw = cv2.imread(str(msk_path), cv2.IMREAD_GRAYSCALE)
                if msk_raw is not None:
                    mask = np.where(msk_raw > 127, 255, 0).astype(np.uint8)
                    del msk_raw
            except Exception as exc:
                logger.warning(f"Cannot load mask {msk_path}: {exc}")

        # Slide window and collect candidates
        candidates: List[Tuple[np.ndarray, Optional[np.ndarray], Dict, float, float]] = []
        for tile_img, tile_mask, meta in self._sliding_window(img, mask, rec.image_id):
            img_std  = float(tile_img.std())
            road_pct = 0.0
            if tile_mask is not None:
                road_pct = float(np.sum(tile_mask > 0)) / tile_mask.size * 100.0
            candidates.append((tile_img, tile_mask, meta, img_std, road_pct))

        # Free the large source arrays now
        del img
        if mask is not None:
            del mask
        gc.collect()

        # Filter candidates
        valid = [c for c in candidates if self._should_keep(*c[3:])]
        discarded = len(candidates) - len(valid)

        # Sort by road_pct descending and cap
        valid.sort(key=lambda c: c[4], reverse=True)
        valid = valid[:self.max_tiles]

        # Save and build TileInfo
        kept: List[TileInfo] = []
        for tile_img, tile_mask, meta, img_std, road_pct in valid:
            tile_id       = f"{split}_{rec.image_id}_r{meta['row']:03d}_c{meta['col']:03d}"
            img_fname     = tile_id + f".{self.img_fmt}"
            msk_fname     = tile_id + "_mask.png"

            tile_img_path = out_img_dir / img_fname
            tile_msk_path = (out_msk_dir / msk_fname) if (out_msk_dir and tile_mask is not None) else None

            self._save_tile_image(tile_img, tile_img_path)
            if tile_msk_path is not None:
                self._save_tile_mask(tile_mask, tile_msk_path)

            kept.append(TileInfo(
                tile_id=tile_id,
                source_image_id=rec.image_id,
                source_dataset=getattr(rec, "source_dataset", "deepglobe"),
                split=split,
                image_tile_path=str(tile_img_path),
                mask_tile_path=str(tile_msk_path) if tile_msk_path else None,
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
                kept=True,
                discard_reason=None,
            ))

        # Free candidate tile arrays
        del candidates
        gc.collect()

        return kept, len(valid) + discarded, discarded

    def _should_keep(self, img_std: float, road_pct: float) -> bool:
        """Return True if the tile passes quality filters."""
        if road_pct < self.min_road_pct:
            return False
        if img_std < self.min_img_std:
            return False
        return True

    # ── Compression-aware save helpers ────────────────────────

    def _save_tile_image(self, tile: np.ndarray, path: Path) -> None:
        """Save a tile image using the configured format and quality."""
        ensure_dir(path.parent)
        if self.img_fmt.lower() in ("jpg", "jpeg"):
            img_bgr = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(path), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, self.img_quality])
        else:
            img_bgr = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(path), img_bgr)

    def _save_tile_mask(self, mask: np.ndarray, path: Path) -> None:
        """Save a mask tile as PNG with maximum compression."""
        ensure_dir(path.parent)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        cv2.imwrite(str(path), mask, [cv2.IMWRITE_PNG_COMPRESSION, self.msk_compress])

    # ── Sliding window ────────────────────────────────────────

    def _sliding_window(
        self,
        img: np.ndarray,
        mask: Optional[np.ndarray],
        image_id: str,
    ) -> Generator[Tuple[np.ndarray, Optional[np.ndarray], Dict], None, None]:
        """
        Yield (tile_img, tile_mask, metadata) for each window position.
        Pads so the full image is covered.
        """
        h, w = img.shape[:2]
        ts = self.tile_size
        st = self.stride

        # Compute padding needed
        def _padded_size(dim: int) -> int:
            if dim <= ts:
                return ts
            remainder = (dim - ts) % st
            return dim + (st - remainder) % st

        ph = _padded_size(h)
        pw = _padded_size(w)

        if self.pad_mode == "reflect":
            img_p = np.pad(img, ((0, ph - h), (0, pw - w), (0, 0)), mode="reflect")
            msk_p = np.pad(mask, ((0, ph - h), (0, pw - w)), mode="reflect") if mask is not None else None
        else:
            img_p = np.pad(img, ((0, ph - h), (0, pw - w), (0, 0)),
                           mode="constant", constant_values=self.pad_value)
            msk_p = np.pad(mask, ((0, ph - h), (0, pw - w)),
                           mode="constant", constant_values=0) if mask is not None else None

        row_idx = 0
        for y in range(0, ph - ts + 1, st):
            col_idx = 0
            for x in range(0, pw - ts + 1, st):
                yield (
                    img_p[y:y+ts, x:x+ts],
                    msk_p[y:y+ts, x:x+ts] if msk_p is not None else None,
                    {
                        "row": row_idx, "col": col_idx,
                        "x_start": x, "y_start": y,
                        "x_end": x + ts, "y_end": y + ts,
                    },
                )
                col_idx += 1
            row_idx += 1

    # ── Statistics ────────────────────────────────────────────

    def _compute_tile_statistics(
        self,
        kept: List[TileInfo],
        total_generated: int,
        total_discarded: int,
    ) -> Dict[str, Any]:
        road_pcts = [t.road_pixel_pct for t in kept]
        by_split: Dict[str, Dict] = {}
        for t in kept:
            s = t.split
            if s not in by_split:
                by_split[s] = {"kept": 0}
            by_split[s]["kept"] += 1

        return {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "total_tiles_generated": total_generated,
            "tiles_kept": len(kept),
            "tiles_discarded": total_discarded,
            "keep_rate_pct": round(len(kept) / total_generated * 100, 2) if total_generated else 0,
            "tile_size": self.tile_size,
            "stride": self.stride,
            "overlap_pct": round((1 - self.stride / self.tile_size) * 100, 1),
            "max_tiles_per_image": self.max_tiles,
            "road_pixel_pct_in_kept": {
                "mean": round(float(np.mean(road_pcts)), 4) if road_pcts else 0,
                "std":  round(float(np.std(road_pcts)), 4) if road_pcts else 0,
                "min":  round(float(np.min(road_pcts)), 4) if road_pcts else 0,
                "max":  round(float(np.max(road_pcts)), 4) if road_pcts else 0,
            },
            "by_split": by_split,
        }

    # ── Visualization ─────────────────────────────────────────

    def _visualize_tile_grid(self, rec: Any, tile_infos: List[TileInfo]) -> None:
        """Overlay tile bounding boxes on source image and save."""
        try:
            img = load_image(Path(rec.image_path))
        except Exception:
            return

        rec_tiles = [t for t in tile_infos if t.source_image_id == rec.image_id]
        if not rec_tiles:
            return

        fig, ax = plt.subplots(1, 1, figsize=(10, 9))
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#0e1117")
        ax.imshow(img)
        ax.set_title(
            f"Tile Grid: {rec.image_id} ({len(rec_tiles)} kept, green)",
            color="white", fontsize=10,
        )
        ax.axis("off")

        for t in rec_tiles:
            rect = patches.Rectangle(
                (t.x_start, t.y_start), t.tile_width, t.tile_height,
                linewidth=0.8, edgecolor="#00ff88", facecolor="#00ff88", alpha=0.20,
            )
            ax.add_patch(rect)

        out = self.out_viz / f"tile_grid_{rec.image_id}.png"
        plt.tight_layout()
        plt.savefig(str(out), dpi=100, bbox_inches="tight", facecolor="#0e1117")
        plt.close()
        del img
        gc.collect()
        logger.debug(f"Tile grid visualization → {out}")


# ─────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────

def run_tiling(
    config: Dict[str, Any],
    records: List[Any],
) -> List[TileInfo]:
    """
    Main tiling entry point.

    Processes only records whose split is in config.datasets.deepglobe.active_splits.
    For Phase 1 this is ['train'] only.

    Returns:
        List of TileInfo for kept tiles.
    """
    tiler = ImageTiler(config)
    kept_tiles = tiler.tile_all(records)

    if kept_tiles:
        # Visualize only first 3 unique source images
        unique_ids = []
        seen = set()
        for t in kept_tiles:
            if t.source_image_id not in seen:
                seen.add(t.source_image_id)
                unique_ids.append(t.source_image_id)
            if len(unique_ids) >= 3:
                break
        sample_records = [r for r in records if r.image_id in seen]
        tiler.generate_tile_grid_visualization(sample_records, kept_tiles, n_samples=3)

    return kept_tiles
