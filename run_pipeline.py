#!/usr/bin/env python3
"""
Route Resilience — run_pipeline.py

Backward-compatible CLI entry point for the Phase 1 data pipeline.

Usage (unchanged from before):
    python run_pipeline.py                              # all steps
    python run_pipeline.py --steps ingest
    python run_pipeline.py --steps ingest standardize
    python run_pipeline.py --config configs/config.yaml --steps all

Canonical entry point (new architecture):
    python scripts/preprocess.py                        # same behaviour

Steps:
    ingest       Step 1: Discover and validate all datasets
    standardize  Step 2: Convert images to RGB, preserve geospatial metadata
    tile         Step 3: Extract 512×512 overlapping tiles
    split        Step 4: 70/15/15 train/val/test split
    augment      Step 5: Build augmentation pipeline + previews
    occlude      Step 6: Simulate tree canopy, shadow, vehicle, cloud occlusions
    quality      Step 7: Quality analysis, road width stats, master_dataset.csv
    visualize    Step 8: Generate pipeline visualization summaries
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

# Ensure project root is on path regardless of working directory
sys.path.insert(0, str(Path(__file__).parent))

# ── Core imports (new canonical paths) ──────────────────────
from src.core.config import load_config
from src.core.logger import get_logger

logger = get_logger("run_pipeline")

ALL_STEPS = [
    "ingest",
    "standardize",
    "tile",
    # "split" — skipped: DeepGlobe provides official splits
    "augment",
    "occlude",
    "quality",
    "visualize",
]

VALID_STEP_CHOICES = ALL_STEPS + ["all", "split"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Route Resilience — Phase 1 Data Pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to YAML configuration file (default: configs/config.yaml)",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        default=["all"],
        choices=VALID_STEP_CHOICES,
        help=(
            "Pipeline steps to run. Use 'all' for the full pipeline.\n"
            f"Available: {', '.join(ALL_STEPS)}"
        ),
    )
    return parser.parse_args()


def run_step(name: str, fn, *args, **kwargs):
    """Run a single pipeline step with timing and error handling."""
    logger.info(f"\n{'━'*60}")
    logger.info(f"  STEP: {name.upper()}")
    logger.info(f"{'━'*60}")
    start = time.perf_counter()
    try:
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"  ✓ {name} completed in {elapsed:.1f}s")
        return result
    except KeyboardInterrupt:
        logger.warning(f"\n⚠ Pipeline interrupted during step: {name}")
        sys.exit(0)
    except Exception as exc:
        elapsed = time.perf_counter() - start
        logger.error(f"  ✗ {name} FAILED after {elapsed:.1f}s: {exc}")
        raise


def main() -> None:
    args = parse_args()

    # Load config
    try:
        config = load_config(args.config)
    except (FileNotFoundError, Exception) as exc:
        logger.error(str(exc))
        sys.exit(1)

    steps: List[str] = ALL_STEPS if "all" in args.steps else args.steps
    # Quietly skip the split step — DeepGlobe provides official splits
    steps = [s for s in steps if s != "split"]

    logger.info("=" * 60)
    logger.info("  ROUTE RESILIENCE — PHASE 1 PIPELINE")
    logger.info(f"  Steps: {', '.join(steps)}")
    logger.info(f"  Config: {args.config}")
    logger.info("=" * 60)

    # Storage guard — abort immediately if disk is already too full
    import shutil
    tc = config.get("tiling", {})
    guard_gb = tc.get("storage_guard_gb", 15.0)
    free_gb = shutil.disk_usage(".").free / (1024 ** 3)
    logger.info(f"  Free disk: {free_gb:.1f} GB  (guard={guard_gb} GB)")
    if free_gb < guard_gb:
        logger.error(
            f"⛔ Only {free_gb:.1f} GB free. Pipeline requires ≥{guard_gb} GB."
        )
        sys.exit(1)

    pipeline_start = time.perf_counter()

    # ── State shared between steps ─────────────────────────────
    records           = []   # List[ImageRecord] from ingestion
    processed         = []   # List[dict]        from standardization
    tile_infos        = []   # List[TileInfo]     from tiling
    train_tiles       = []
    val_tiles         = []
    test_tiles        = []
    aug_pipeline      = None
    occlusion_results = []

    # ── Step 1: Ingest ─────────────────────────────────────────
    if "ingest" in steps:
        from src.data.ingestion import run_ingestion
        records, report = run_step("ingest", run_ingestion, config)
        if not records:
            logger.error(
                "\nNo data available. Pipeline cannot continue.\n"
                "Please download datasets and retry.\n"
                "See instructions above, or run:\n"
                "  python src/data/download_utils.py --dataset check"
            )
            sys.exit(1)

    # ── Step 2: Standardize ────────────────────────────────────
    if "standardize" in steps:
        if not records:
            logger.warning("No records in memory. Running ingest first…")
            from src.data.ingestion import run_ingestion
            records, _ = run_step("ingest", run_ingestion, config)
            if not records:
                logger.error("No data found. Exiting.")
                sys.exit(1)

        from src.data.standardization import run_standardization
        processed = run_step("standardize", run_standardization, config, records)

    # ── Step 3: Tile ───────────────────────────────────────────
    if "tile" in steps:
        if not records:
            from src.data.ingestion import run_ingestion
            records, _ = run_step("ingest", run_ingestion, config)
            if not records:
                logger.error("No data found. Exiting.")
                sys.exit(1)

        from src.data.tiling import run_tiling
        tile_infos = run_step("tile", run_tiling, config, records)

        if not tile_infos:
            logger.error(
                "No tiles were kept after filtering. "
                "Check 'min_road_pixel_pct' and 'min_image_std' in config."
            )
            sys.exit(1)

    # ── Step 4: Split ──────────────────────────────────────────
    if "split" in steps:
        logger.info(
            "Skipping split step: DeepGlobe provides official "
            "train/valid/test splits embedded in tile paths."
        )

    # Reconstruct records and tile_infos from CSV if not in memory but needed
    needed_for_later = any(s in steps for s in ["augment", "occlude", "quality", "visualize"])
    if needed_for_later:
        if not records:
            logger.info("Loading records via ingestion...")
            from src.data.ingestion import run_ingestion
            records, _ = run_step("ingest", run_ingestion, config)

        if not tile_infos:
            csv_path = Path("outputs/reports/master_dataset.csv")
            if csv_path.exists():
                logger.info("Reconstructing tile_infos from outputs/reports/master_dataset.csv...")
                import pandas as pd
                from src.data.tiling import TileInfo
                df = pd.read_csv(csv_path)
                tile_infos = []
                for _, row in df.iterrows():
                    dims = str(row["image_dimensions"]).split("x")
                    w = int(dims[0]) if len(dims) > 0 else 512
                    h = int(dims[1]) if len(dims) > 1 else 512
                    parts = str(row["image_id"]).split("_")
                    row_num, col_num = 0, 0
                    if len(parts) >= 4:
                        try:
                            row_num = int(parts[-2].replace("r", ""))
                            col_num = int(parts[-1].replace("c", ""))
                        except ValueError:
                            pass
                    tile_infos.append(TileInfo(
                        tile_id=row["image_id"],
                        source_image_id=str(row["source_image_id"]),
                        source_dataset=row["source_dataset"],
                        split=row["split"],
                        image_tile_path=row["image_tile_path"],
                        mask_tile_path=row["mask_tile_path"] if pd.notna(row["mask_tile_path"]) else None,
                        row=row_num,
                        col=col_num,
                        x_start=col_num * 256,
                        y_start=row_num * 256,
                        x_end=col_num * 256 + w,
                        y_end=row_num * 256 + h,
                        tile_width=w,
                        tile_height=h,
                        road_pixel_pct=float(row["road_pixel_pct"]),
                        image_std=15.0,
                        kept=True,
                        discard_reason=None
                    ))
                logger.info(f"Loaded {len(tile_infos)} tile infos from master_dataset.csv.")

    # ── Step 5: Augment ────────────────────────────────────────
    if "augment" in steps:
        all_tiles = train_tiles or tile_infos
        from src.data.augmentation import run_augmentation
        aug_pipeline = run_step("augment", run_augmentation, config, all_tiles)

    # ── Step 6: Occlude ────────────────────────────────────────
    if "occlude" in steps:
        all_tiles = train_tiles or tile_infos
        if not all_tiles:
            logger.error(
                "No tiles available for occlusion simulation. "
                "Run tile and split steps first."
            )
            sys.exit(1)

        from src.data.occlusion import run_occlusion
        occlusion_results = run_step("occlude", run_occlusion, config, all_tiles)

    # ── Step 7: Quality ─────────────────────────────────────────
    if "quality" in steps:
        if not records or not tile_infos:
            logger.error(
                "Records and tiles are required for quality analysis. "
                "Run ingest and tile steps first."
            )
            sys.exit(1)

        from src.data.quality import run_quality_analysis
        # Pass all tile_infos as train_tiles — splits are embedded in tile.split
        quality_report = run_step(
            "quality",
            run_quality_analysis,
            config,
            records,
            tile_infos,
            tile_infos,   # train_tiles
            [],           # val_tiles  (not processed in Phase 1)
            [],           # test_tiles (not processed in Phase 1)
            occlusion_results or [],
        )

        logger.info(
            f"\n  Class imbalance ratio (bg:road): "
            f"{quality_report['class_imbalance']['imbalance_ratio_bg_to_road']:.1f}:1"
        )
        logger.info(f"  Recommendation: {quality_report['class_imbalance']['recommendation']}")

    # ── Step 8: Visualize ──────────────────────────────────────
    if "visualize" in steps:
        if not records:
            logger.warning("No records for visualization.")
        else:
            from src.visualization.plots import run_visualization
            run_step(
                "visualize",
                run_visualization,
                config,
                records,
                tile_infos or [],
                occlusion_results or [],
            )

    # ── Summary ────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - pipeline_start
    logger.info("\n" + "=" * 60)
    logger.info("  PIPELINE COMPLETE")
    logger.info(f"  Total time: {total_elapsed:.1f}s")
    logger.info("  Outputs:")
    logger.info("    outputs/reports/dataset_report.json")
    logger.info("    outputs/reports/quality_report.json")
    logger.info("    outputs/reports/master_dataset.csv")
    logger.info("    outputs/visualizations/  ← PNG charts")
    logger.info("    outputs/occlusion_samples/  ← Occlusion examples")
    logger.info("    data/tiles/train/  ← 512×512 JPEG tiles + PNG masks")
    import shutil as _su
    used_gb = (_su.disk_usage(".").total - _su.disk_usage(".").free) / (1024**3)
    free_gb = _su.disk_usage(".").free / (1024**3)
    logger.info(f"  Disk: {free_gb:.1f} GB free")
    logger.info("=" * 60 + "\n")


if __name__ == "__main__":
    main()
