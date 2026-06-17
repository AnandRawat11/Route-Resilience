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
    "split",
    "augment",
    "occlude",
    "quality",
    "visualize",
]


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
        choices=ALL_STEPS + ["all"],
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

    logger.info("=" * 60)
    logger.info("  ROUTE RESILIENCE — PHASE 1 PIPELINE")
    logger.info(f"  Steps: {', '.join(steps)}")
    logger.info(f"  Config: {args.config}")
    logger.info("=" * 60)

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
        if not tile_infos:
            logger.error(
                "No tiles in memory for splitting. "
                "Run 'tile' step first or use --steps tile split."
            )
            sys.exit(1)

        from src.data.splitting import run_splitting
        train_tiles, val_tiles, test_tiles = run_step(
            "split", run_splitting, config, tile_infos
        )

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

    # ── Step 7: Quality ────────────────────────────────────────
    if "quality" in steps:
        if not records or not tile_infos:
            logger.error(
                "Records and tiles are required for quality analysis. "
                "Run ingest and tile steps first."
            )
            sys.exit(1)

        from src.data.quality import run_quality_analysis
        quality_report = run_step(
            "quality",
            run_quality_analysis,
            config,
            records,
            tile_infos,
            train_tiles or tile_infos,
            val_tiles or [],
            test_tiles or [],
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
    logger.info("    outputs/reports/train.csv  val.csv  test.csv")
    logger.info("    outputs/visualizations/  ← PNG charts")
    logger.info("    outputs/occlusion_samples/  ← Occlusion examples")
    logger.info("    data/processed/  ← Standardised images and masks")
    logger.info("    data/tiles/      ← 512×512 tiles")
    logger.info("    data/train|val|test/  ← Split datasets")
    logger.info("=" * 60 + "\n")


if __name__ == "__main__":
    main()
