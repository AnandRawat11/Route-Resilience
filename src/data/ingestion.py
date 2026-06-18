"""
Route Resilience — Phase 1
src/ingestion.py

Step 1: Data Ingestion
  - Discover image/mask pairs for each dataset source
  - Validate file integrity
  - Detect missing masks
  - Compute per-dataset statistics
  - Output: outputs/reports/dataset_report.json

Adapter dispatch:
  If a dataset config has use_adapter: true, the scanner delegates
  to the corresponding adapter (e.g. DeepGlobeAdapter) instead of
  using the generic images/ + masks/ directory scan.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.core.io     import ensure_dir, load_image, load_mask, save_json, format_bytes, extract_geospatial_metadata
from src.core.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────
#  Data structures
# ─────────────────────────────────────────────────────────────

@dataclass
class ImageRecord:
    """Holds all metadata for a single image-mask pair."""
    image_id: str
    source_dataset: str
    image_path: str
    mask_path: Optional[str]
    has_mask: bool
    image_width: int
    image_height: int
    image_channels: int
    image_dtype: str
    image_size_bytes: int
    mask_width: Optional[int]
    mask_height: Optional[int]
    road_pixel_count: int
    total_pixel_count: int
    road_pixel_pct: float
    is_georeferenced: bool
    crs: Optional[str]
    bounds: Optional[Dict]
    transform: Optional[Dict]
    resolution_x: Optional[float]
    resolution_y: Optional[float]
    md5_image: str
    md5_mask: Optional[str]
    validation_errors: List[str] = field(default_factory=list)
    # Optional: set by adapters that know the official split
    split: Optional[str] = field(default=None)

    def is_valid(self) -> bool:
        return len(self.validation_errors) == 0


@dataclass
class DatasetStats:
    """Aggregated statistics for one dataset source."""
    source: str
    total_images: int = 0
    total_masks: int = 0
    missing_masks: int = 0
    invalid_images: int = 0
    road_pixel_pct_mean: float = 0.0
    road_pixel_pct_std: float = 0.0
    road_pixel_pct_min: float = 0.0
    road_pixel_pct_max: float = 0.0
    image_widths: List[int] = field(default_factory=list)
    image_heights: List[int] = field(default_factory=list)
    total_size_bytes: int = 0
    georeferenced_count: int = 0


# ─────────────────────────────────────────────────────────────
#  Dataset scanner
# ─────────────────────────────────────────────────────────────

class DatasetScanner:
    """
    Discovers and validates image/mask pairs for all configured dataset sources.

    Usage:
        scanner = DatasetScanner(config)
        records = scanner.scan_all()
        report  = scanner.generate_report(records)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.dataset_cfg = config.get("datasets", {})
        self.report_dir = Path(config.get("paths", {}).get("reports", "outputs/reports"))
        ensure_dir(self.report_dir)
        logger.info("DatasetScanner initialised.")

    # ── Public API ────────────────────────────────────────────

    def scan_all(self) -> List[ImageRecord]:
        """Scan all enabled dataset sources and return a flat list of ImageRecords."""
        all_records: List[ImageRecord] = []
        enabled_sources = [
            (name, cfg)
            for name, cfg in self.dataset_cfg.items()
            if isinstance(cfg, dict) and cfg.get("enabled", False)
        ]

        if not enabled_sources:
            logger.warning(
                "No enabled datasets found in config. "
                "Set 'enabled: true' under datasets in configs/config.yaml."
            )
            return []

        for source_name, source_cfg in enabled_sources:
            logger.info(f"Scanning dataset: {source_name}")
            try:
                if source_cfg.get("use_adapter", False):
                    records = self._scan_via_adapter(source_name, source_cfg)
                else:
                    records = self._scan_source(source_name, source_cfg)
                all_records.extend(records)
                logger.info(
                    f"  {source_name}: {len(records)} records "
                    f"({sum(1 for r in records if r.has_mask)} with masks)"
                )
            except Exception as exc:
                logger.error(f"  Failed to scan {source_name}: {exc}")

        logger.info(f"Total records found: {len(all_records)}")
        return all_records

    def _scan_via_adapter(
        self, source_name: str, source_cfg: Dict[str, Any]
    ) -> List[ImageRecord]:
        """Delegate scanning to a dataset-specific adapter."""
        adapter_name = source_cfg.get("adapter", source_name)
        active_splits = source_cfg.get("active_splits", None)

        if adapter_name == "deepglobe":
            from src.data.adapters.deepglobe_adapter import DeepGlobeAdapter
            adapter = DeepGlobeAdapter(self.config)
            return adapter.scan(active_splits=active_splits)
        else:
            logger.warning(
                f"Unknown adapter '{adapter_name}' for dataset '{source_name}'. "
                "Falling back to generic scanner."
            )
            return self._scan_source(source_name, source_cfg)

    def generate_report(self, records: List[ImageRecord]) -> Dict[str, Any]:
        """
        Build the full dataset_report.json from a list of ImageRecords.

        Saves to outputs/reports/dataset_report.json and returns the dict.
        """
        if not records:
            logger.warning("No records to report. Datasets may be missing or empty.")
            report = self._empty_report()
            save_json(report, self.report_dir / "dataset_report.json")
            return report

        per_source: Dict[str, DatasetStats] = {}
        for rec in records:
            if rec.source_dataset not in per_source:
                per_source[rec.source_dataset] = DatasetStats(source=rec.source_dataset)
            stats = per_source[rec.source_dataset]
            stats.total_images += 1
            stats.total_size_bytes += rec.image_size_bytes
            stats.image_widths.append(rec.image_width)
            stats.image_heights.append(rec.image_height)
            if rec.has_mask:
                stats.total_masks += 1
            else:
                stats.missing_masks += 1
            if not rec.is_valid():
                stats.invalid_images += 1
            if rec.is_georeferenced:
                stats.georeferenced_count += 1

        # Road pixel stats per source
        for source, stats in per_source.items():
            source_records = [r for r in records if r.source_dataset == source and r.has_mask]
            if source_records:
                pcts = [r.road_pixel_pct for r in source_records]
                stats.road_pixel_pct_mean = float(np.mean(pcts))
                stats.road_pixel_pct_std = float(np.std(pcts))
                stats.road_pixel_pct_min = float(np.min(pcts))
                stats.road_pixel_pct_max = float(np.max(pcts))

        # Global stats
        all_pcts = [r.road_pixel_pct for r in records if r.has_mask]
        all_sizes = [(r.image_width, r.image_height) for r in records]
        missing = [r.image_path for r in records if not r.has_mask]
        invalid = [r.image_path for r in records if not r.is_valid()]

        report: Dict[str, Any] = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "summary": {
                "total_images": len(records),
                "total_masks": sum(1 for r in records if r.has_mask),
                "missing_masks": len(missing),
                "invalid_images": len(invalid),
                "georeferenced_images": sum(1 for r in records if r.is_georeferenced),
                "total_dataset_size": format_bytes(sum(r.image_size_bytes for r in records)),
            },
            "class_distribution": {
                "road_pixel_pct_mean": round(float(np.mean(all_pcts)), 4) if all_pcts else 0.0,
                "road_pixel_pct_std": round(float(np.std(all_pcts)), 4) if all_pcts else 0.0,
                "background_pixel_pct_mean": round(100.0 - float(np.mean(all_pcts)), 4) if all_pcts else 100.0,
            },
            "image_dimensions": {
                "widths": {
                    "mean": round(float(np.mean([w for w, _ in all_sizes])), 1) if all_sizes else 0,
                    "min": int(min(w for w, _ in all_sizes)) if all_sizes else 0,
                    "max": int(max(w for w, _ in all_sizes)) if all_sizes else 0,
                },
                "heights": {
                    "mean": round(float(np.mean([h for _, h in all_sizes])), 1) if all_sizes else 0,
                    "min": int(min(h for _, h in all_sizes)) if all_sizes else 0,
                    "max": int(max(h for _, h in all_sizes)) if all_sizes else 0,
                },
            },
            "missing_masks": missing,
            "invalid_images": invalid,
            "per_dataset_stats": {
                src: {
                    "total_images": s.total_images,
                    "total_masks": s.total_masks,
                    "missing_masks": s.missing_masks,
                    "invalid_images": s.invalid_images,
                    "georeferenced_count": s.georeferenced_count,
                    "road_pixel_pct": {
                        "mean": round(s.road_pixel_pct_mean, 4),
                        "std": round(s.road_pixel_pct_std, 4),
                        "min": round(s.road_pixel_pct_min, 4),
                        "max": round(s.road_pixel_pct_max, 4),
                    },
                    "image_widths": {
                        "mean": round(float(np.mean(s.image_widths)), 1) if s.image_widths else 0,
                        "min": int(min(s.image_widths)) if s.image_widths else 0,
                        "max": int(max(s.image_widths)) if s.image_widths else 0,
                    },
                    "image_heights": {
                        "mean": round(float(np.mean(s.image_heights)), 1) if s.image_heights else 0,
                        "min": int(min(s.image_heights)) if s.image_heights else 0,
                        "max": int(max(s.image_heights)) if s.image_heights else 0,
                    },
                    "total_size": format_bytes(s.total_size_bytes),
                }
                for src, s in per_source.items()
            },
        }

        report_path = self.report_dir / "dataset_report.json"
        save_json(report, report_path)
        logger.info(f"Dataset report saved → {report_path}")
        return report

    def print_setup_instructions(self) -> None:
        """Print clear instructions when no dataset is found."""
        instructions = """
╔══════════════════════════════════════════════════════════════════╗
║          ROUTE RESILIENCE — DATASET SETUP INSTRUCTIONS          ║
╚══════════════════════════════════════════════════════════════════╝

No datasets were found. Please download and place them as follows:

── 1. SpaceNet Roads Dataset ──────────────────────────────────────
   Source : https://spacenet.ai/roads/
   AWS    : s3://spacenet-dataset/spacenet/SN3_roads/
   Place  : data/raw/spacenet/
   Structure:
     data/raw/spacenet/images/   ← satellite .tif files
     data/raw/spacenet/masks/    ← road mask .tif files
   Download helper: python src/download_utils.py --dataset spacenet

── 2. DeepGlobe Road Extraction ──────────────────────────────────
   Source : https://competitions.codalab.org/competitions/18467
   Kaggle : https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset
   Place  : data/raw/deepglobe/
   Structure:
     data/raw/deepglobe/images/   ← *_sat.jpg files
     data/raw/deepglobe/masks/    ← *_mask.png files
   Download helper: python src/download_utils.py --dataset deepglobe

── 3. OpenSatMap ─────────────────────────────────────────────────
   Source : https://github.com/OpenSatMap/OpenSatMap
   Place  : data/raw/opensatmap/
   Structure:
     data/raw/opensatmap/images/
     data/raw/opensatmap/masks/
   Download helper: python src/download_utils.py --dataset opensatmap

── 4. OpenStreetMap Road Vectors ─────────────────────────────────
   Source : https://download.geofabrik.de/ (download .osm.pbf)
           or use Overpass API (see src/download_utils.py)
   Place  : data/raw/osm/vectors/   ← .geojson or .shp files
   Conversion: python src/vector_utils.py --convert-to-masks

── After placing datasets, re-run: ───────────────────────────────
   python run_pipeline.py --steps ingest

╔══════════════════════════════════════════════════════════════════╗
"""
        print(instructions)

    # ── Internal helpers ──────────────────────────────────────

    def _scan_source(
        self, source_name: str, source_cfg: Dict[str, Any]
    ) -> List[ImageRecord]:
        """Scan one dataset source directory and return a list of ImageRecords."""
        raw_dir = Path(source_cfg.get("raw_dir", f"data/raw/{source_name}"))
        image_subdir = source_cfg.get("image_subdir", "images")
        mask_subdir = source_cfg.get("mask_subdir", "masks")
        image_exts = set(source_cfg.get("image_extensions", [".tif", ".png", ".jpg"]))
        mask_exts = set(source_cfg.get("mask_extensions", [".tif", ".png"]))

        img_dir = raw_dir / image_subdir
        msk_dir = raw_dir / mask_subdir

        if not raw_dir.exists():
            logger.warning(
                f"Dataset directory not found: {raw_dir}  "
                f"[{source_name} will be skipped]"
            )
            return []

        if not img_dir.exists():
            logger.warning(f"Images subdirectory not found: {img_dir}")
            return []

        # Collect image files
        image_files = sorted([
            f for f in img_dir.iterdir()
            if f.is_file() and f.suffix.lower() in image_exts
        ])

        if not image_files:
            logger.warning(f"No image files found in {img_dir}")
            return []

        records: List[ImageRecord] = []
        for img_path in image_files:
            rec = self._build_record(
                img_path, msk_dir, mask_exts, source_name
            )
            records.append(rec)

        return records

    def _build_record(
        self,
        img_path: Path,
        msk_dir: Path,
        mask_exts: set,
        source_name: str,
    ) -> ImageRecord:
        """Build an ImageRecord for a single image file."""
        errors: List[str] = []
        image_id = img_path.stem

        # Find matching mask
        mask_path: Optional[Path] = None
        for ext in mask_exts:
            candidate = msk_dir / (img_path.stem + ext)
            if candidate.exists():
                mask_path = candidate
                break

        # Validate image
        img_width = img_height = img_channels = img_size = 0
        img_dtype = "unknown"
        is_geo = False
        crs = bounds = transform = res_x = res_y = None

        try:
            geo = extract_geospatial_metadata(img_path)
            img_width = geo.get("width") or 0
            img_height = geo.get("height") or 0
            img_channels = geo.get("band_count") or 0
            img_dtype = geo.get("dtype") or "unknown"
            is_geo = geo.get("is_georeferenced", False)
            crs = geo.get("crs")
            bounds = geo.get("bounds")
            transform = geo.get("transform")
            res_x = geo.get("resolution_x")
            res_y = geo.get("resolution_y")
            img_size = img_path.stat().st_size
        except Exception as exc:
            errors.append(f"Image metadata error: {exc}")

        if img_width == 0 or img_height == 0:
            errors.append("Zero-size image dimensions detected")

        # Validate and compute mask stats
        road_pixels = 0
        total_pixels = img_width * img_height
        msk_width = msk_height = None
        md5_mask = None

        if mask_path and mask_path.exists():
            try:
                mask = load_mask(mask_path)
                msk_height, msk_width = mask.shape[:2]
                road_pixels = int(np.sum(mask > 0))

                if msk_width != img_width or msk_height != img_height:
                    errors.append(
                        f"Image/mask dimension mismatch: "
                        f"image={img_width}×{img_height}, "
                        f"mask={msk_width}×{msk_height}"
                    )

                md5_mask = compute_md5(mask_path)
            except Exception as exc:
                errors.append(f"Mask load error: {exc}")
                mask_path = None

        road_pct = (road_pixels / total_pixels * 100.0) if total_pixels > 0 else 0.0

        # Image MD5
        md5_img = ""
        try:
            md5_img = compute_md5(img_path)
        except Exception:
            pass

        return ImageRecord(
            image_id=image_id,
            source_dataset=source_name,
            image_path=str(img_path),
            mask_path=str(mask_path) if mask_path else None,
            has_mask=mask_path is not None,
            image_width=img_width,
            image_height=img_height,
            image_channels=img_channels,
            image_dtype=img_dtype,
            image_size_bytes=img_size,
            mask_width=msk_width,
            mask_height=msk_height,
            road_pixel_count=road_pixels,
            total_pixel_count=total_pixels,
            road_pixel_pct=road_pct,
            is_georeferenced=is_geo,
            crs=crs,
            bounds=bounds,
            transform=transform,
            resolution_x=res_x,
            resolution_y=res_y,
            md5_image=md5_img,
            md5_mask=md5_mask,
            validation_errors=errors,
        )

    @staticmethod
    def _empty_report() -> Dict[str, Any]:
        return {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "summary": {
                "total_images": 0,
                "total_masks": 0,
                "missing_masks": 0,
                "invalid_images": 0,
                "georeferenced_images": 0,
                "total_dataset_size": "0 B",
            },
            "class_distribution": {
                "road_pixel_pct_mean": 0.0,
                "road_pixel_pct_std": 0.0,
                "background_pixel_pct_mean": 100.0,
            },
            "image_dimensions": {},
            "missing_masks": [],
            "invalid_images": [],
            "per_dataset_stats": {},
        }


# ─────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────

def run_ingestion(config: Dict[str, Any]) -> Tuple[List[ImageRecord], Dict[str, Any]]:
    """
    Main ingestion entry point.

    Returns:
        (records, report) tuple.
        records: list of ImageRecord objects.
        report: dict matching dataset_report.json schema.
    """
    scanner = DatasetScanner(config)
    records = scanner.scan_all()

    if not records:
        scanner.print_setup_instructions()
        logger.error(
            "No valid data found. Please download datasets and retry. "
            "See setup instructions above."
        )
        return [], scanner._empty_report()

    report = scanner.generate_report(records)

    # Summary to console
    logger.info("── Ingestion Summary ──────────────────────────────")
    logger.info(f"  Total images   : {report['summary']['total_images']}")
    logger.info(f"  Total masks    : {report['summary']['total_masks']}")
    logger.info(f"  Missing masks  : {report['summary']['missing_masks']}")
    logger.info(f"  Georeferenced  : {report['summary']['georeferenced_images']}")
    logger.info(f"  Road pixel %   : {report['class_distribution']['road_pixel_pct_mean']:.2f}% ± "
                f"{report['class_distribution']['road_pixel_pct_std']:.2f}%")
    logger.info("───────────────────────────────────────────────────")

    return records, report
