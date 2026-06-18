"""
Route Resilience — src/data/adapters/deepglobe_adapter.py

Adapter for the DeepGlobe Road Extraction Dataset
(Kaggle/CodaLab competition format).

Dataset Layout
--------------
<root>/
    train/
        {id}_sat.jpg     ← satellite image
        {id}_mask.png    ← road mask (RGB: white=road, black=bg)
    valid/
        {id}_sat.jpg     ← images only (no masks)
    test/
        {id}_sat.jpg     ← images only (no masks)
    metadata.csv         ← image_id, split, sat_image_path, mask_path
    class_dict.csv       ← road: 255,255,255  background: 0,0,0

Notes
-----
* Masks are RGB with white (255,255,255) for road pixels.
  They are converted to binary (0 / 255 grayscale) during scanning.
* valid and test have no masks — has_mask=False for those records.
* active_splits=['train'] will skip valid/test entirely (Phase 1 policy).
"""

from __future__ import annotations

import csv
import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from src.data.adapters.base_adapter import BaseAdapter
from src.core.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────
#  Reuse ImageRecord dataclass from ingestion
# ─────────────────────────────────────────────────────────────

def _make_image_record(**kwargs):
    """
    Lazily import ImageRecord to avoid circular imports.
    Returns a fully-populated ImageRecord instance.
    """
    from src.data.ingestion import ImageRecord
    return ImageRecord(**kwargs)


# ─────────────────────────────────────────────────────────────
#  DeepGlobeAdapter
# ─────────────────────────────────────────────────────────────

class DeepGlobeAdapter(BaseAdapter):
    """
    Adapter for the DeepGlobe Road Extraction Dataset.

    Reads the official metadata.csv to discover all image/mask pairs
    and their split assignments (train/valid/test).

    Only splits listed in *active_splits* are processed.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        dg_cfg = config.get("datasets", {}).get("deepglobe", {})
        raw_dir = dg_cfg.get("raw_dir", "data/raw/deepglobe")
        self.root = Path(raw_dir)
        self.metadata_csv = self.root / "metadata.csv"
        logger.info(f"DeepGlobeAdapter initialised: root={self.root}")

    # ── Public API ────────────────────────────────────────────

    def scan(
        self,
        active_splits: Optional[List[str]] = None,
    ) -> List[Any]:
        """
        Scan the dataset and return ImageRecords for *active_splits* only.

        Args:
            active_splits: List like ['train']. Defaults to ['train'].

        Returns:
            List of ImageRecord objects.
        """
        if active_splits is None:
            active_splits = ["train"]

        if not self.root.exists():
            logger.error(
                f"DeepGlobe dataset directory not found: {self.root}\n"
                "Ensure the symlink data/raw/deepglobe points to the dataset."
            )
            return []

        rows = self._read_metadata()
        if not rows:
            # Fallback: auto-discover from directory listing
            logger.warning(
                "metadata.csv not found or empty. "
                "Auto-discovering from directory listing."
            )
            rows = self._autodiscover()

        # Filter by active splits
        rows = [r for r in rows if r["split"] in active_splits]
        logger.info(
            f"DeepGlobe: processing splits={active_splits}, "
            f"{len(rows)} records found."
        )

        records = []
        for i, row in enumerate(rows):
            rec = self._build_record(row)
            if rec is not None:
                records.append(rec)
            if (i + 1) % 500 == 0:
                logger.info(
                    f"  Scanned {i + 1}/{len(rows)} records…"
                )

        logger.info(
            f"DeepGlobe scan complete: {len(records)} valid records "
            f"({sum(1 for r in records if r.has_mask)} with masks)."
        )
        return records

    def generate_statistics(self, records: List[Any]) -> Dict[str, Any]:
        """
        Produce per-split statistics for the dataset report.
        """
        splits: Dict[str, Dict] = {}
        for rec in records:
            sp = getattr(rec, "split", "unknown")
            if sp not in splits:
                splits[sp] = {
                    "total_images": 0,
                    "total_masks": 0,
                    "missing_masks": 0,
                    "invalid_images": 0,
                    "road_pcts": [],
                    "widths": [],
                    "heights": [],
                    "total_size_bytes": 0,
                }
            s = splits[sp]
            s["total_images"] += 1
            s["total_size_bytes"] += rec.image_size_bytes
            s["widths"].append(rec.image_width)
            s["heights"].append(rec.image_height)
            if rec.has_mask:
                s["total_masks"] += 1
                if rec.road_pixel_pct > 0:
                    s["road_pcts"].append(rec.road_pixel_pct)
            else:
                s["missing_masks"] += 1
            if not rec.is_valid():
                s["invalid_images"] += 1

        result: Dict[str, Any] = {}
        for sp, s in splits.items():
            pcts = s["road_pcts"]
            result[sp] = {
                "total_images": s["total_images"],
                "total_masks": s["total_masks"],
                "missing_masks": s["missing_masks"],
                "invalid_images": s["invalid_images"],
                "road_pixel_pct": {
                    "mean": round(float(np.mean(pcts)), 4) if pcts else 0.0,
                    "std": round(float(np.std(pcts)), 4) if pcts else 0.0,
                    "min": round(float(np.min(pcts)), 4) if pcts else 0.0,
                    "max": round(float(np.max(pcts)), 4) if pcts else 0.0,
                },
                "image_dimensions": {
                    "width_mean": round(float(np.mean(s["widths"])), 1) if s["widths"] else 0,
                    "height_mean": round(float(np.mean(s["heights"])), 1) if s["heights"] else 0,
                },
                "total_size": _fmt_bytes(s["total_size_bytes"]),
            }
        return result

    # ── Internal helpers ──────────────────────────────────────

    def _read_metadata(self) -> List[Dict[str, str]]:
        """Parse metadata.csv → list of row dicts."""
        if not self.metadata_csv.exists():
            return []
        rows = []
        try:
            with open(self.metadata_csv, newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    rows.append({
                        "image_id": row["image_id"].strip(),
                        "split": row["split"].strip(),
                        "sat_path": row.get("sat_image_path", "").strip(),
                        "mask_path": row.get("mask_path", "").strip(),
                    })
        except Exception as exc:
            logger.error(f"Failed to read metadata.csv: {exc}")
        return rows

    def _autodiscover(self) -> List[Dict[str, str]]:
        """
        Fallback: discover records by scanning split directories directly.
        """
        rows = []
        for split in ("train", "valid", "test"):
            split_dir = self.root / split
            if not split_dir.exists():
                continue
            for sat_file in sorted(split_dir.glob("*_sat.jpg")):
                image_id = sat_file.stem.replace("_sat", "")
                mask_file = split_dir / f"{image_id}_mask.png"
                rows.append({
                    "image_id": image_id,
                    "split": split,
                    "sat_path": str(sat_file.relative_to(self.root)),
                    "mask_path": str(mask_file.relative_to(self.root))
                    if mask_file.exists()
                    else "",
                })
        return rows

    def _build_record(self, row: Dict[str, str]) -> Optional[Any]:
        """
        Build an ImageRecord from a metadata row.

        Converts DeepGlobe RGB mask to binary on load and caches basic stats.
        Does NOT load the full image into memory here — only metadata is read.
        """
        image_id = row["image_id"]
        split = row["split"]

        # Resolve paths
        sat_rel = row.get("sat_path", "")
        if sat_rel:
            img_path = self.root / sat_rel
        else:
            img_path = self.root / split / f"{image_id}_sat.jpg"

        mask_rel = row.get("mask_path", "")
        if mask_rel:
            mask_path = self.root / mask_rel
        else:
            mask_path = self.root / split / f"{image_id}_mask.png"

        if not img_path.exists():
            logger.debug(f"Image not found, skipping: {img_path}")
            return None

        has_mask = mask_path.exists()
        errors: List[str] = []

        # Read image dimensions without loading full array
        try:
            img_meta = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if img_meta is None:
                errors.append("Cannot read image header")
                img_h, img_w, img_c = 0, 0, 0
            else:
                img_h, img_w = img_meta.shape[:2]
                img_c = 1 if img_meta.ndim == 2 else img_meta.shape[2]
                del img_meta
        except Exception as exc:
            errors.append(f"Image read error: {exc}")
            img_h, img_w, img_c = 0, 0, 0

        img_size = img_path.stat().st_size

        # Road pixel stats (cheap: read mask grayscale, threshold)
        road_pixels = 0
        total_pixels = img_w * img_h
        msk_w = msk_h = None
        md5_mask = None

        if has_mask:
            try:
                # DeepGlobe masks are RGB white=road; read as grayscale
                msk = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if msk is not None:
                    msk_h, msk_w = msk.shape[:2]
                    # White pixels (>127) = road
                    road_pixels = int(np.sum(msk > 127))
                    del msk
                else:
                    errors.append("Cannot read mask")
                    has_mask = False
            except Exception as exc:
                errors.append(f"Mask read error: {exc}")
                has_mask = False

        road_pct = (road_pixels / total_pixels * 100.0) if total_pixels > 0 else 0.0

        # Quick MD5 (header only — first 64 KB)
        md5_img = _fast_md5(img_path)

        rec = _make_image_record(
            image_id=image_id,
            source_dataset="deepglobe",
            image_path=str(img_path),
            mask_path=str(mask_path) if has_mask else None,
            has_mask=has_mask,
            image_width=img_w,
            image_height=img_h,
            image_channels=img_c,
            image_dtype="uint8",
            image_size_bytes=img_size,
            mask_width=msk_w,
            mask_height=msk_h,
            road_pixel_count=road_pixels,
            total_pixel_count=total_pixels,
            road_pixel_pct=road_pct,
            is_georeferenced=False,
            crs=None,
            bounds=None,
            transform=None,
            resolution_x=None,
            resolution_y=None,
            md5_image=md5_img,
            md5_mask=md5_mask,
            validation_errors=errors,
        )
        # Attach split as a plain attribute (not in base dataclass)
        rec.split = split
        return rec


# ─────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────

def _fast_md5(path: Path, chunk_size: int = 65_536) -> str:
    """MD5 of first chunk only — fast fingerprint, not cryptographic."""
    h = hashlib.md5()
    try:
        with open(path, "rb") as fh:
            h.update(fh.read(chunk_size))
    except Exception:
        pass
    return h.hexdigest()


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"
