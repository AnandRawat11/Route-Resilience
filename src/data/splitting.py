"""
Route Resilience — Phase 1
src/splitting.py

Step 4: Dataset Splitting
  - Stratified 70/15/15 train/val/test split
  - Reproducible (random_seed=42)
  - Zero data leakage verified
  - Copies tiles to data/train|val|test/
  - Saves train.csv, val.csv, test.csv, split_metadata.json
"""

from __future__ import annotations

import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.core.io     import ensure_dir, save_json
from src.core.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────
#  DatasetSplitter
# ─────────────────────────────────────────────────────────────

class DatasetSplitter:
    """
    Splits tile-level data into train/val/test sets.

    Supports:
      - Stratification by dataset source (so each split mirrors source distribution)
      - Deterministic random seed
      - Verification of zero overlap
      - CSV and JSON metadata output
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        sc = config.get("splitting", {})
        self.train_ratio: float = sc.get("train_ratio", 0.70)
        self.val_ratio: float = sc.get("val_ratio", 0.15)
        self.test_ratio: float = sc.get("test_ratio", 0.15)
        self.stratify: bool = sc.get("stratify_by_source", True)
        self.seed: int = config.get("project", {}).get("random_seed", 42)

        paths = config.get("paths", {})
        self.train_img_dir = ensure_dir(paths.get("train_images", "data/train/images"))
        self.train_msk_dir = ensure_dir(paths.get("train_masks", "data/train/masks"))
        self.val_img_dir = ensure_dir(paths.get("val_images", "data/val/images"))
        self.val_msk_dir = ensure_dir(paths.get("val_masks", "data/val/masks"))
        self.test_img_dir = ensure_dir(paths.get("test_images", "data/test/images"))
        self.test_msk_dir = ensure_dir(paths.get("test_masks", "data/test/masks"))
        self.report_dir = ensure_dir(paths.get("reports", "outputs/reports"))

        # Validate ratios sum to 1
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Split ratios must sum to 1.0, got {total:.4f}. "
                f"Check configs/config.yaml splitting section."
            )

        logger.info(
            f"DatasetSplitter: train={self.train_ratio:.0%}, "
            f"val={self.val_ratio:.0%}, test={self.test_ratio:.0%}, "
            f"seed={self.seed}, stratify={self.stratify}"
        )

    # ── Public API ────────────────────────────────────────────

    def split(
        self, tile_infos: List[Any]
    ) -> Tuple[List[Any], List[Any], List[Any]]:
        """
        Split tile_infos into train/val/test lists.

        Args:
            tile_infos: List of TileInfo objects from tiling step.

        Returns:
            (train_tiles, val_tiles, test_tiles) — no duplicates.
        """
        if not tile_infos:
            logger.warning("No tiles to split.")
            return [], [], []

        random.seed(self.seed)
        np.random.seed(self.seed)

        if self.stratify:
            train, val, test = self._stratified_split(tile_infos)
        else:
            train, val, test = self._simple_split(tile_infos)

        self._verify_no_leakage(train, val, test)

        # Copy files
        logger.info("Copying tiles to split directories…")
        self._copy_tiles(train, "train")
        self._copy_tiles(val, "val")
        self._copy_tiles(test, "test")

        # Save CSVs
        self._save_csv(train, "train")
        self._save_csv(val, "val")
        self._save_csv(test, "test")

        # Save split metadata
        self._save_split_metadata(train, val, test)

        logger.info(
            f"Split complete — train: {len(train)}, "
            f"val: {len(val)}, test: {len(test)}"
        )
        return train, val, test

    # ── Splitting strategies ───────────────────────────────────

    def _stratified_split(
        self, tiles: List[Any]
    ) -> Tuple[List[Any], List[Any], List[Any]]:
        """
        Stratified split: maintains source distribution within each split.
        Groups by source_dataset, splits each group independently, then merges.
        """
        by_source: Dict[str, List[Any]] = defaultdict(list)
        for t in tiles:
            by_source[t.source_dataset].append(t)

        all_train, all_val, all_test = [], [], []

        for source, source_tiles in by_source.items():
            random.shuffle(source_tiles)
            n = len(source_tiles)
            n_train = max(1, int(n * self.train_ratio))
            n_val = max(1, int(n * self.val_ratio))
            # Remainder goes to test
            n_test = n - n_train - n_val

            if n_test < 0:
                # Edge case: very small source
                n_val = max(0, n - n_train)
                n_test = 0

            train_s = source_tiles[:n_train]
            val_s = source_tiles[n_train:n_train + n_val]
            test_s = source_tiles[n_train + n_val:]

            all_train.extend(train_s)
            all_val.extend(val_s)
            all_test.extend(test_s)

            logger.debug(
                f"  {source}: {n} tiles → "
                f"train={len(train_s)}, val={len(val_s)}, test={len(test_s)}"
            )

        return all_train, all_val, all_test

    def _simple_split(
        self, tiles: List[Any]
    ) -> Tuple[List[Any], List[Any], List[Any]]:
        """Non-stratified random split."""
        shuffled = tiles.copy()
        random.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)
        return (
            shuffled[:n_train],
            shuffled[n_train:n_train + n_val],
            shuffled[n_train + n_val:],
        )

    # ── Verification ──────────────────────────────────────────

    def _verify_no_leakage(
        self,
        train: List[Any],
        val: List[Any],
        test: List[Any],
    ) -> None:
        """Assert zero overlap between splits (by tile_id)."""
        train_ids = {t.tile_id for t in train}
        val_ids = {t.tile_id for t in val}
        test_ids = {t.tile_id for t in test}

        tv = train_ids & val_ids
        tt = train_ids & test_ids
        vt = val_ids & test_ids

        if tv:
            raise RuntimeError(f"Data leakage detected: {len(tv)} tiles in both train and val!")
        if tt:
            raise RuntimeError(f"Data leakage detected: {len(tt)} tiles in both train and test!")
        if vt:
            raise RuntimeError(f"Data leakage detected: {len(vt)} tiles in both val and test!")

        logger.info("Data leakage check: PASSED — all splits are disjoint.")

    # ── File operations ───────────────────────────────────────

    def _copy_tiles(self, tiles: List[Any], split: str) -> None:
        """Copy tile image and mask files to the split directory."""
        img_dir = getattr(self, f"{split}_img_dir")
        msk_dir = getattr(self, f"{split}_msk_dir")

        for t in tiles:
            if t.image_tile_path and Path(t.image_tile_path).exists():
                shutil.copy2(t.image_tile_path, img_dir / Path(t.image_tile_path).name)
            if t.mask_tile_path and Path(t.mask_tile_path).exists():
                shutil.copy2(t.mask_tile_path, msk_dir / Path(t.mask_tile_path).name)

    def _save_csv(self, tiles: List[Any], split: str) -> None:
        """Save split metadata to CSV."""
        rows = []
        for t in tiles:
            rows.append({
                "tile_id": t.tile_id,
                "source_dataset": t.source_dataset,
                "source_image_id": t.source_image_id,
                "image_tile_path": t.image_tile_path,
                "mask_tile_path": t.mask_tile_path or "",
                "road_pixel_pct": round(t.road_pixel_pct, 4),
                "image_std": round(t.image_std, 4),
                "row": t.row,
                "col": t.col,
                "split": split,
            })

        if not rows:
            logger.warning(f"No tiles for {split} split — CSV will be empty.")

        df = pd.DataFrame(rows)
        csv_path = self.report_dir / f"{split}.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"  Saved {csv_path} ({len(rows)} rows)")

    def _save_split_metadata(
        self,
        train: List[Any],
        val: List[Any],
        test: List[Any],
    ) -> None:
        """Save split_metadata.json with counts and source distributions."""
        def source_dist(tiles: List[Any]) -> Dict[str, int]:
            counts: Dict[str, int] = {}
            for t in tiles:
                counts[t.source_dataset] = counts.get(t.source_dataset, 0) + 1
            return counts

        meta = {
            "random_seed": self.seed,
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
            "stratified": self.stratify,
            "counts": {
                "train": len(train),
                "val": len(val),
                "test": len(test),
                "total": len(train) + len(val) + len(test),
            },
            "source_distribution": {
                "train": source_dist(train),
                "val": source_dist(val),
                "test": source_dist(test),
            },
        }
        save_json(meta, self.report_dir / "split_metadata.json")
        logger.info(f"Split metadata saved → {self.report_dir / 'split_metadata.json'}")


# ─────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────

def run_splitting(
    config: Dict[str, Any],
    tile_infos: List[Any],
) -> Tuple[List[Any], List[Any], List[Any]]:
    """
    Main splitting entry point.

    Returns:
        (train_tiles, val_tiles, test_tiles)
    """
    splitter = DatasetSplitter(config)
    return splitter.split(tile_infos)
