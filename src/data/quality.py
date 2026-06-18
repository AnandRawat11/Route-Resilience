"""
Route Resilience — Phase 1
src/quality.py

Step 7: Dataset Quality Analysis
  - Road pixel percentage (mean, std, min, max)
  - Road width estimation via distance transform
  - Road density per tile
  - Class imbalance metrics
  - Generates quality_report.json
  - Generates master_dataset.csv (central source of truth)
  - Visual charts (histograms, pie charts, scatter plots)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

from src.core.io     import ensure_dir, load_json, load_mask, save_json
from src.core.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────
#  QualityAnalyzer
# ─────────────────────────────────────────────────────────────

class QualityAnalyzer:
    """
    Computes quality metrics for the full dataset and each split.

    Produces:
      - outputs/reports/quality_report.json
      - outputs/reports/master_dataset.csv
      - outputs/visualizations/quality_*.png charts
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        qc = config.get("quality", {})
        self.road_width_enabled: bool = qc.get("road_width_estimation", {}).get("enabled", True)
        self.road_width_method: str = qc.get("road_width_estimation", {}).get("method", "distance_transform")
        self.generate_charts: bool = qc.get("generate_charts", True)

        paths = config.get("paths", {})
        self.report_dir = ensure_dir(paths.get("reports", "outputs/reports"))
        self.out_viz = ensure_dir(paths.get("visualizations", "outputs/visualizations"))

        logger.info("QualityAnalyzer initialised.")

    # ── Public API ────────────────────────────────────────────

    def analyze(
        self,
        records: List[Any],
        tile_infos: List[Any],
        train_tiles: List[Any],
        val_tiles: List[Any],
        test_tiles: List[Any],
        occlusion_results: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """
        Full quality analysis pass.

        Returns:
            quality_report dict (also saved as quality_report.json).
        """
        logger.info("Running quality analysis…")

        # Per-tile metrics
        tile_rows = self._compute_tile_metrics(
            train_tiles, val_tiles, test_tiles, occlusion_results
        )

        # Aggregate stats
        quality_report = self._compute_report(records, tile_rows, train_tiles, val_tiles, test_tiles)

        # Save reports
        save_json(quality_report, self.report_dir / "quality_report.json")
        logger.info(f"Quality report saved → {self.report_dir / 'quality_report.json'}")

        # Master dataset CSV
        self._save_master_csv(records, tile_rows, train_tiles, val_tiles, test_tiles, occlusion_results)

        # Visual charts
        if self.generate_charts:
            self._generate_charts(tile_rows, quality_report)

        return quality_report

    # ── Tile-level metrics ─────────────────────────────────────

    def _compute_tile_metrics(
        self,
        train: List[Any],
        val: List[Any],
        test: List[Any],
        occlusion_results: Optional[List[Any]],
    ) -> List[Dict[str, Any]]:
        """Compute per-tile quality metrics including road width."""
        # Combine all tiles; determine split from tile.split attribute if available
        all_tiles_with_split = []
        for t in train:
            all_tiles_with_split.append((t, getattr(t, "split", "train")))
        for t in val:
            all_tiles_with_split.append((t, getattr(t, "split", "val")))
        for t in test:
            all_tiles_with_split.append((t, getattr(t, "split", "test")))

        # Build occlusion lookup: tile_id → OcclusionResult
        occ_lookup: Dict[str, Any] = {}
        if occlusion_results:
            for occ in occlusion_results:
                occ_lookup[occ.image_id] = occ

        rows = []
        for tile, split in all_tiles_with_split:
            row: Dict[str, Any] = {
                "tile_id": tile.tile_id,
                "source_dataset": tile.source_dataset,
                "source_image_id": tile.source_image_id,
                "split": split,
                "image_tile_path": tile.image_tile_path,
                "mask_tile_path": tile.mask_tile_path or "",
                "road_pixel_pct": round(tile.road_pixel_pct, 4),
                "background_pixel_pct": round(100.0 - tile.road_pixel_pct, 4),
                "image_std": round(tile.image_std, 4),
                "tile_width": tile.tile_width,
                "tile_height": tile.tile_height,
                "avg_road_width_px": None,
                "max_road_width_px": None,
                "road_density": None,
                "occlusion_type": None,
                "occlusion_severity": None,
                "occlusion_coverage_pct": None,
                "occlusion_road_coverage_pct": None,
            }

            # Road width + density from mask
            if tile.mask_tile_path and Path(tile.mask_tile_path).exists() and self.road_width_enabled:
                try:
                    mask = load_mask(Path(tile.mask_tile_path))
                    avg_w, max_w, density = self._compute_road_stats(mask)
                    row["avg_road_width_px"] = round(avg_w, 2) if avg_w else None
                    row["max_road_width_px"] = round(max_w, 2) if max_w else None
                    row["road_density"] = round(density, 6) if density else None
                except Exception as exc:
                    logger.debug(f"Road stats failed for {tile.tile_id}: {exc}")

            # Occlusion metadata
            if tile.tile_id in occ_lookup:
                occ = occ_lookup[tile.tile_id]
                row["occlusion_type"] = occ.occlusion_type
                row["occlusion_severity"] = occ.severity
                row["occlusion_coverage_pct"] = occ.coverage_pct
                row["occlusion_road_coverage_pct"] = occ.road_coverage_pct

            rows.append(row)

        return rows

    def _compute_road_stats(
        self, mask: np.ndarray
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Estimate road width using morphological distance transform.

        Returns:
            (avg_width_px, max_width_px, road_density)
        """
        binary = (mask > 0).astype(np.uint8)
        if binary.sum() == 0:
            return None, None, 0.0

        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        road_px = dist[dist > 0]

        # Each local maximum in distance transform ≈ half the road width
        avg_half_width = float(road_px.mean()) if len(road_px) > 0 else 0.0
        max_half_width = float(road_px.max()) if len(road_px) > 0 else 0.0

        total_px = mask.shape[0] * mask.shape[1]
        density = float(binary.sum()) / total_px

        # Width = 2× half-width (diameter of inscribed circle)
        return avg_half_width * 2, max_half_width * 2, density

    # ── Report aggregation ────────────────────────────────────

    def _compute_report(
        self,
        records: List[Any],
        tile_rows: List[Dict],
        train: List[Any],
        val: List[Any],
        test: List[Any],
    ) -> Dict[str, Any]:
        road_pcts = [r["road_pixel_pct"] for r in tile_rows if r["road_pixel_pct"] is not None]
        widths = [r["avg_road_width_px"] for r in tile_rows if r["avg_road_width_px"] is not None]
        densities = [r["road_density"] for r in tile_rows if r["road_density"] is not None]

        def safe_stat(vals: List) -> Dict:
            if not vals:
                return {"mean": 0, "std": 0, "min": 0, "max": 0}
            return {
                "mean": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals)), 4),
                "min": round(float(np.min(vals)), 4),
                "max": round(float(np.max(vals)), 4),
            }

        # Class imbalance
        bg_pcts = [r["background_pixel_pct"] for r in tile_rows if r["background_pixel_pct"] is not None]
        avg_road = np.mean(road_pcts) if road_pcts else 0
        avg_bg = np.mean(bg_pcts) if bg_pcts else 100
        imbalance_ratio = avg_bg / max(avg_road, 1e-6)

        # Occlusion coverage
        occ_covs = [r["occlusion_coverage_pct"] for r in tile_rows if r["occlusion_coverage_pct"] is not None]
        occ_road_covs = [r["occlusion_road_coverage_pct"] for r in tile_rows if r["occlusion_road_coverage_pct"] is not None]

        by_severity: Dict[str, List] = {}
        for r in tile_rows:
            sev = r.get("occlusion_severity")
            if sev:
                by_severity.setdefault(sev, []).append(r.get("occlusion_road_coverage_pct", 0) or 0)

        return {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "total_records": len(records),
            "total_tiles": len(tile_rows),
            "split_counts": {
                sp: sum(1 for r in tile_rows if r["split"] == sp)
                for sp in sorted({r["split"] for r in tile_rows})
            },
            "road_pixel_percentage": safe_stat(road_pcts),
            "background_pixel_percentage": safe_stat(bg_pcts),
            "class_imbalance": {
                "road_pct_mean": round(float(avg_road), 4),
                "background_pct_mean": round(float(avg_bg), 4),
                "imbalance_ratio_bg_to_road": round(float(imbalance_ratio), 2),
                "recommendation": (
                    "Severe class imbalance detected. "
                    "Consider weighted loss (Dice + BCE) or oversampling road-heavy tiles."
                    if imbalance_ratio > 10 else
                    "Moderate class imbalance. Dice loss recommended."
                    if imbalance_ratio > 5 else
                    "Acceptable class balance."
                ),
            },
            "road_width_statistics": {
                "avg_road_width_px": safe_stat(widths),
                "max_road_width_px": safe_stat([r["max_road_width_px"] for r in tile_rows if r["max_road_width_px"]]),
                "method": self.road_width_method,
                "note": "Road width estimated via distance transform (2× inscribed circle radius). "
                        "Used for future weighted graph construction.",
            },
            "road_density": safe_stat(densities),
            "occlusion_summary": {
                "total_occluded_samples": len(occ_covs),
                "coverage_pct": safe_stat(occ_covs) if occ_covs else {},
                "road_coverage_pct": safe_stat(occ_road_covs) if occ_road_covs else {},
                "by_severity": {
                    sev: {
                        "count": len(vals),
                        "avg_road_coverage_pct": round(float(np.mean(vals)), 2) if vals else 0,
                    }
                    for sev, vals in by_severity.items()
                },
            },
            "image_size_distribution": {
                "tile_width": {"fixed": 512},
                "tile_height": {"fixed": 512},
            },
        }

    # ── Master CSV ────────────────────────────────────────────

    def _save_master_csv(
        self,
        records: List[Any],
        tile_rows: List[Dict],
        train: List[Any],
        val: List[Any],
        test: List[Any],
        occlusion_results: Optional[List[Any]],
    ) -> None:
        """
        Save master_dataset.csv — the central source of truth for the entire project.
        """
        # Build geo reference lookup from records
        geo_lookup: Dict[str, str] = {}
        for rec in records:
            meta_path = Path("data/processed/metadata") / (rec.image_id + "_geo.json")
            if meta_path.exists():
                geo_lookup[rec.image_id] = str(meta_path)

        rows = []
        for row in tile_rows:
            geo_ref = geo_lookup.get(row["source_image_id"], "")
            rows.append({
                "image_id": row["tile_id"],
                "source_dataset": row["source_dataset"],
                "source_image_id": row["source_image_id"],
                "image_dimensions": f"{row['tile_width']}x{row['tile_height']}",
                "road_density": row["road_density"],
                "avg_road_width_px": row["avg_road_width_px"],
                "max_road_width_px": row["max_road_width_px"],
                "road_pixel_pct": row["road_pixel_pct"],
                "background_pixel_pct": row["background_pixel_pct"],
                "occlusion_type": row["occlusion_type"] or "",
                "occlusion_severity": row["occlusion_severity"] or "",
                "occlusion_coverage_pct": row["occlusion_coverage_pct"] or "",
                "occlusion_road_coverage_pct": row["occlusion_road_coverage_pct"] or "",
                "split": row["split"],
                "image_tile_path": row["image_tile_path"],
                "mask_tile_path": row["mask_tile_path"],
                "geospatial_metadata_reference": geo_ref,
            })

        df = pd.DataFrame(rows)
        csv_path = self.report_dir / "master_dataset.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Master dataset CSV saved → {csv_path} ({len(rows)} rows)")

    # ── Charts ────────────────────────────────────────────────

    def _generate_charts(
        self, tile_rows: List[Dict], report: Dict[str, Any]
    ) -> None:
        """Generate and save quality analysis charts."""
        dark = "#0e1117"
        accent = "#7b68ee"

        fig = plt.figure(figsize=(20, 16))
        fig.patch.set_facecolor(dark)
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

        # 1. Road pixel % histogram
        ax1 = fig.add_subplot(gs[0, 0])
        road_pcts = [r["road_pixel_pct"] for r in tile_rows if r["road_pixel_pct"] is not None]
        ax1.hist(road_pcts, bins=40, color=accent, edgecolor="#333", alpha=0.85)
        ax1.set_title("Road Pixel % Distribution", color="white", fontsize=11)
        ax1.set_xlabel("Road Pixel %", color="#aaa")
        ax1.set_ylabel("Count", color="#aaa")
        _style_ax(ax1, dark)

        # 2. Class balance pie
        ax2 = fig.add_subplot(gs[0, 1])
        avg_road = report["road_pixel_percentage"]["mean"]
        avg_bg = report["background_pixel_percentage"]["mean"]
        pie_colors = ["#7b68ee", "#2d2d4e"]
        wedges, texts, autotexts = ax2.pie(
            [avg_road, avg_bg],
            labels=["Road", "Background"],
            autopct="%1.1f%%",
            colors=pie_colors,
            textprops={"color": "white"},
            startangle=90,
        )
        ax2.set_title("Avg Class Balance", color="white", fontsize=11)
        ax2.set_facecolor(dark)

        # 3. Road width histogram
        ax3 = fig.add_subplot(gs[0, 2])
        widths = [r["avg_road_width_px"] for r in tile_rows if r["avg_road_width_px"] is not None]
        if widths:
            ax3.hist(widths, bins=30, color="#00d4aa", edgecolor="#333", alpha=0.85)
        ax3.set_title("Avg Road Width Distribution (px)", color="white", fontsize=11)
        ax3.set_xlabel("Width (pixels)", color="#aaa")
        ax3.set_ylabel("Count", color="#aaa")
        _style_ax(ax3, dark)

        # 4. Split size bar
        ax4 = fig.add_subplot(gs[1, 0])
        splits = ["Train", "Val", "Test"]
        counts = [
            report["split_counts"].get("train", 0),
            report["split_counts"].get("val", 0),
            report["split_counts"].get("test", 0),
        ]
        colors = ["#7b68ee", "#00d4aa", "#ff6b6b"]
        bars = ax4.bar(splits, counts, color=colors, edgecolor="#333")
        for bar, count in zip(bars, counts):
            ax4.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                str(count),
                ha="center", color="white", fontsize=10
            )
        ax4.set_title("Split Distribution", color="white", fontsize=11)
        ax4.set_ylabel("Tile Count", color="#aaa")
        _style_ax(ax4, dark)

        # 5. Road density scatter vs road pixel %
        ax5 = fig.add_subplot(gs[1, 1])
        densities = [r["road_density"] for r in tile_rows if r["road_density"] is not None]
        road_pcts2 = [r["road_pixel_pct"] for r in tile_rows if r["road_density"] is not None]
        if densities and road_pcts2:
            ax5.scatter(densities, road_pcts2, alpha=0.4, s=10, c=accent)
        ax5.set_title("Road Density vs Road Pixel %", color="white", fontsize=11)
        ax5.set_xlabel("Road Density (fraction)", color="#aaa")
        ax5.set_ylabel("Road Pixel %", color="#aaa")
        _style_ax(ax5, dark)

        # 6. Occlusion coverage by severity
        ax6 = fig.add_subplot(gs[1, 2])
        occ_by_sev = report.get("occlusion_summary", {}).get("by_severity", {})
        if occ_by_sev:
            sev_labels = list(occ_by_sev.keys())
            sev_means = [occ_by_sev[s].get("avg_road_coverage_pct", 0) for s in sev_labels]
            sev_colors = ["#ffd700", "#ff9900", "#ff3300"]
            ax6.bar(sev_labels, sev_means, color=sev_colors[:len(sev_labels)], edgecolor="#333")
            ax6.set_title("Occlusion Road Coverage by Severity", color="white", fontsize=11)
            ax6.set_ylabel("Avg Road Coverage %", color="#aaa")
            _style_ax(ax6, dark)
        else:
            ax6.text(0.5, 0.5, "No occlusion data", ha="center", va="center",
                     color="#888", transform=ax6.transAxes)
            ax6.set_facecolor(dark)
            ax6.axis("off")

        fig.suptitle(
            "Route Resilience — Phase 1 Quality Analysis",
            color="white", fontsize=16, fontweight="bold"
        )
        out = self.out_viz / "quality_analysis.png"
        plt.savefig(str(out), dpi=150, bbox_inches="tight", facecolor=dark)
        plt.close()
        logger.info(f"Quality charts saved → {out}")


# ─────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────

def _style_ax(ax: Any, bg: str = "#0e1117") -> None:
    ax.set_facecolor(bg)
    ax.tick_params(colors="#aaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")


# ─────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────

def run_quality_analysis(
    config: Dict[str, Any],
    records: List[Any],
    tile_infos: List[Any],
    train_tiles: List[Any],
    val_tiles: List[Any],
    test_tiles: List[Any],
    occlusion_results: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """
    Main quality analysis entry point.

    Returns:
        quality_report dict.
    """
    analyzer = QualityAnalyzer(config)
    return analyzer.analyze(
        records, tile_infos, train_tiles, val_tiles, test_tiles, occlusion_results
    )
