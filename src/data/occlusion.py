"""
Route Resilience — Phase 1
src/occlusion.py

Step 6: Occlusion Simulation (Key Innovation)

Generates four types of realistic synthetic occlusions:
  1. Tree Canopy   — Perlin-noise organic blobs, green overlay
  2. Building Shadow — Directional gradient polygons, dark desaturated
  3. Vehicle        — Road-aligned rectangles, metallic colors
  4. Cloud Cover    — Smooth Gaussian blobs, white/grey opacity

Each occlusion is tagged with severity: light | medium | heavy
Outputs: original.png, occluded.png, comparison.png per sample
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.core.io     import ensure_dir, load_image, load_mask, save_image, save_json
from src.core.logger import get_logger

logger = get_logger(__name__)

# Perlin noise — graceful fallback to Gaussian blobs
try:
    import noise  # type: ignore
    _HAS_NOISE = True
except ImportError:
    _HAS_NOISE = False
    logger.warning(
        "Perlin 'noise' package not installed. "
        "Tree canopy simulation will use Gaussian blobs instead. "
        "Install with: pip install noise"
    )


# ─────────────────────────────────────────────────────────────
#  Severity
# ─────────────────────────────────────────────────────────────

SEVERITY_LEVELS = ["light", "medium", "heavy"]


@dataclass
class OcclusionResult:
    """Result of applying one occlusion to an image."""
    image_id: str
    occlusion_type: str
    severity: str
    coverage_pct: float           # fraction of image occluded
    road_coverage_pct: float      # fraction of road pixels occluded
    original_path: str
    occluded_path: str
    comparison_path: str


# ─────────────────────────────────────────────────────────────
#  OcclusionSimulator
# ─────────────────────────────────────────────────────────────

class OcclusionSimulator:
    """
    Applies synthetic occlusions to real satellite imagery.

    Strictly operates on real satellite images — no synthetic image generation.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        oc = config.get("occlusion", {})
        self.enabled: bool = oc.get("enabled", True)
        self.n_samples: int = oc.get("samples_to_generate", 20)
        self.types_cfg: Dict = oc.get("types", {})
        self.seed: int = config.get("project", {}).get("random_seed", 42)

        paths = config.get("paths", {})
        self.out_dir = ensure_dir(paths.get("occlusion_samples", "outputs/occlusion_samples"))
        self.report_dir = ensure_dir(paths.get("reports", "outputs/reports"))

        random.seed(self.seed)
        np.random.seed(self.seed)

        logger.info("OcclusionSimulator initialised.")

    # ── Public API ────────────────────────────────────────────

    def simulate_all(
        self,
        tile_infos: List[Any],
    ) -> List[OcclusionResult]:
        """
        Apply occlusions to a random selection of tiles and save sample outputs.

        Args:
            tile_infos: List of TileInfo objects (kept tiles).

        Returns:
            List of OcclusionResult metadata.
        """
        if not self.enabled:
            logger.info("Occlusion simulation disabled in config.")
            return []

        # Filter tiles that have both image and mask
        valid = [t for t in tile_infos if t.image_tile_path and t.mask_tile_path]
        if not valid:
            logger.warning("No tiles with masks found for occlusion simulation.")
            return []

        # Sample tiles
        n = min(self.n_samples, len(valid))
        selected = random.sample(valid, n)

        results: List[OcclusionResult] = []
        enabled_types = [
            t for t in ["tree_canopy", "building_shadow", "vehicle", "cloud_cover"]
            if self.types_cfg.get(t, {}).get("enabled", True)
        ]

        if not enabled_types:
            logger.warning("All occlusion types disabled in config.")
            return []

        for i, tile in enumerate(selected):
            occ_type = enabled_types[i % len(enabled_types)]
            severity = SEVERITY_LEVELS[i % 3]
            logger.debug(
                f"Occluding [{i+1}/{n}]: {tile.tile_id} "
                f"type={occ_type} severity={severity}"
            )
            try:
                result = self._simulate_one(tile, occ_type, severity)
                if result:
                    results.append(result)
            except Exception as exc:
                logger.error(f"Occlusion failed for {tile.tile_id}: {exc}")

        # Save summary
        stats = self._compute_statistics(results)
        save_json(stats, self.report_dir / "occlusion_statistics.json")
        logger.info(
            f"Occlusion simulation complete: {len(results)} samples generated."
        )
        return results

    # ── Core simulation ───────────────────────────────────────

    def _simulate_one(
        self,
        tile: Any,
        occ_type: str,
        severity: str,
    ) -> Optional[OcclusionResult]:
        """Apply one occlusion type at a given severity to one tile."""
        img = load_image(Path(tile.image_tile_path))
        mask = load_mask(Path(tile.mask_tile_path))

        # Apply occlusion
        if occ_type == "tree_canopy":
            occluded, occlusion_mask = self._apply_tree_canopy(img, mask, severity)
        elif occ_type == "building_shadow":
            occluded, occlusion_mask = self._apply_building_shadow(img, mask, severity)
        elif occ_type == "vehicle":
            occluded, occlusion_mask = self._apply_vehicle(img, mask, severity)
        elif occ_type == "cloud_cover":
            occluded, occlusion_mask = self._apply_cloud_cover(img, severity)
        else:
            return None

        # Coverage stats
        total_px = img.shape[0] * img.shape[1]
        coverage_pct = float(np.sum(occlusion_mask > 0)) / total_px * 100.0
        road_px = np.sum(mask > 0)
        road_coverage_pct = 0.0
        if road_px > 0:
            road_coverage_pct = float(
                np.sum((occlusion_mask > 0) & (mask > 0))
            ) / road_px * 100.0

        # Save outputs
        base = f"{tile.tile_id}_{occ_type}_{severity}"
        orig_path = self.out_dir / f"{base}_original.png"
        occ_path = self.out_dir / f"{base}_occluded.png"
        comp_path = self.out_dir / f"{base}_comparison.png"

        save_image(img, orig_path)
        save_image(occluded, occ_path)
        self._save_comparison(img, occluded, occlusion_mask, mask, comp_path, occ_type, severity)

        return OcclusionResult(
            image_id=tile.tile_id,
            occlusion_type=occ_type,
            severity=severity,
            coverage_pct=round(coverage_pct, 2),
            road_coverage_pct=round(road_coverage_pct, 2),
            original_path=str(orig_path),
            occluded_path=str(occ_path),
            comparison_path=str(comp_path),
        )

    # ── Occlusion types ───────────────────────────────────────

    def _apply_tree_canopy(
        self,
        img: np.ndarray,
        mask: np.ndarray,
        severity: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate tree canopy using Perlin noise (or Gaussian blobs fallback).
        Generates organic irregular shapes in green-brown tones.
        """
        h, w = img.shape[:2]
        cfg = self.types_cfg.get("tree_canopy", {})
        sev_cfg = cfg.get(severity, {})
        cov_min = sev_cfg.get("coverage_min", 0.25)
        cov_max = sev_cfg.get("coverage_max", 0.50)
        blob_min, blob_max = cfg.get("blob_size_range", [30, 120])
        color_var = cfg.get("color_variation", 20)

        target_coverage = random.uniform(cov_min, cov_max)
        occluded = img.copy().astype(np.float32)
        occlusion_mask = np.zeros((h, w), dtype=np.uint8)

        total_px = h * w
        covered = 0

        while covered / total_px < target_coverage:
            blob_r = random.randint(blob_min // 2, blob_max // 2)
            cx = random.randint(blob_r, w - blob_r)
            cy = random.randint(blob_r, h - blob_r)

            if _HAS_NOISE:
                # Perlin-noise shaped blob
                blob = _perlin_blob(h, w, cx, cy, blob_r,
                                    scale=cfg.get("noise_scale", 0.05))
            else:
                # Gaussian blob fallback
                blob = _gaussian_blob(h, w, cx, cy, blob_r)

            blob = (blob > 0.35).astype(np.uint8)

            # Green-brown color for canopy
            base_green = np.array([34, 85, 34], dtype=np.float32)  # dark green
            color = base_green + np.random.uniform(-color_var, color_var, 3)
            color = np.clip(color, 0, 255)

            # Alpha blend
            alpha = np.where(blob, random.uniform(0.55, 0.80), 0)
            for c in range(3):
                occluded[:, :, c] = (
                    occluded[:, :, c] * (1 - alpha) + color[c] * alpha
                )

            occlusion_mask = np.maximum(occlusion_mask, (blob * 255).astype(np.uint8))
            covered = int(np.sum(occlusion_mask > 0))

        return np.clip(occluded, 0, 255).astype(np.uint8), occlusion_mask

    def _apply_building_shadow(
        self,
        img: np.ndarray,
        mask: np.ndarray,
        severity: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate building shadow as elongated dark polygons with gradient edges.
        Sun angle determines shadow direction.
        """
        h, w = img.shape[:2]
        cfg = self.types_cfg.get("building_shadow", {})
        sev_cfg = cfg.get(severity, {})
        cov_min = sev_cfg.get("coverage_min", 0.20)
        cov_max = sev_cfg.get("coverage_max", 0.40)
        darkening = cfg.get("darkening_factor", 0.35)
        sun_min, sun_max = cfg.get("sun_angle_range", [30, 150])
        len_min, len_max = cfg.get("shadow_length_range", [0.5, 2.0])

        target_coverage = random.uniform(cov_min, cov_max)
        occluded = img.copy().astype(np.float32)
        occlusion_mask = np.zeros((h, w), dtype=np.uint8)

        total_px = h * w
        covered = 0
        max_iters = 50

        for _ in range(max_iters):
            if covered / total_px >= target_coverage:
                break

            # Shadow origin (simulate a building corner)
            ox = random.randint(w // 8, 7 * w // 8)
            oy = random.randint(h // 8, 7 * h // 8)
            bld_w = random.randint(20, 80)
            bld_h = random.randint(20, 80)
            sun_angle = random.uniform(sun_min, sun_max)
            shadow_len = random.uniform(len_min, len_max) * max(bld_w, bld_h)

            shadow_poly = _compute_shadow_polygon(ox, oy, bld_w, bld_h, sun_angle, shadow_len)
            shadow_poly = np.array(shadow_poly, dtype=np.int32)

            shadow_img = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(shadow_img, [shadow_poly], 255)

            # Gradient: darker near building, lighter at tip
            dist_map = cv2.distanceTransform(shadow_img, cv2.DIST_L2, 5)
            if dist_map.max() > 0:
                gradient = 1.0 - (dist_map / dist_map.max()) * 0.4
            else:
                gradient = np.ones((h, w), dtype=np.float32)

            alpha = np.where(shadow_img > 0, gradient * (1 - darkening), 0)
            for c in range(3):
                occluded[:, :, c] = occluded[:, :, c] * (1 - alpha * (1 - darkening))

            occlusion_mask = np.maximum(occlusion_mask, shadow_img)
            covered = int(np.sum(occlusion_mask > 0))

        return np.clip(occluded, 0, 255).astype(np.uint8), occlusion_mask

    def _apply_vehicle(
        self,
        img: np.ndarray,
        mask: np.ndarray,
        severity: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate vehicles as rectangles placed along road pixels.
        Vehicle colors sampled from realistic metallic palettes.
        """
        h, w = img.shape[:2]
        cfg = self.types_cfg.get("vehicle", {})
        sev_cfg = cfg.get(severity, {})
        count_range = sev_cfg.get("count_range", [3, 8])
        veh_w_range = cfg.get("size_range_w", [8, 25])
        veh_h_range = cfg.get("size_range_h", [15, 45])

        n_vehicles = random.randint(*count_range)
        occluded = img.copy()
        occlusion_mask = np.zeros((h, w), dtype=np.uint8)

        # Find road pixels as candidate vehicle centers
        road_ys, road_xs = np.where(mask > 0)
        if len(road_ys) == 0:
            # No road: place randomly
            road_ys = np.arange(0, h)
            road_xs = np.arange(0, w)

        # Vehicle color palette (metallic tones)
        vehicle_colors = [
            (200, 200, 200),  # silver
            (40, 40, 40),     # black
            (180, 30, 30),    # red
            (30, 30, 160),    # blue
            (200, 180, 30),   # yellow (taxi)
            (50, 100, 50),    # dark green
        ]

        for _ in range(n_vehicles):
            idx = random.randint(0, len(road_ys) - 1)
            cx, cy = int(road_xs[idx]), int(road_ys[idx])
            vw = random.randint(*veh_w_range)
            vh = random.randint(*veh_h_range)
            color = random.choice(vehicle_colors)

            x1 = max(0, cx - vw // 2)
            y1 = max(0, cy - vh // 2)
            x2 = min(w, cx + vw // 2)
            y2 = min(h, cy + vh // 2)

            cv2.rectangle(occluded, (x1, y1), (x2, y2), color, -1)
            # Add highlight line (roof reflection)
            cv2.rectangle(
                occluded,
                (x1 + 1, y1 + 1),
                (x2 - 1, y1 + max(2, vh // 6)),
                tuple(min(255, c + 60) for c in color),
                -1,
            )
            occlusion_mask[y1:y2, x1:x2] = 255

        return occluded, occlusion_mask

    def _apply_cloud_cover(
        self,
        img: np.ndarray,
        severity: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate cloud cover using Gaussian blobs with variable opacity.
        Produces realistic white/grey cloud patches.
        """
        h, w = img.shape[:2]
        cfg = self.types_cfg.get("cloud_cover", {})
        sev_cfg = cfg.get(severity, {})
        cov_min = sev_cfg.get("coverage_min", 0.20)
        cov_max = sev_cfg.get("coverage_max", 0.45)
        opacity_min, opacity_max = sev_cfg.get("opacity_range", [0.5, 0.75])
        sigma_min, sigma_max = cfg.get("blur_sigma_range", [20, 60])

        target_coverage = random.uniform(cov_min, cov_max)
        occluded = img.copy().astype(np.float32)
        cloud_layer = np.zeros((h, w), dtype=np.float32)

        total_px = h * w
        covered = 0
        max_iters = 30

        for _ in range(max_iters):
            if covered / total_px >= target_coverage:
                break

            sigma = random.uniform(sigma_min, sigma_max)
            cx = random.randint(0, w)
            cy = random.randint(0, h)

            blob = _gaussian_blob(h, w, cx, cy, radius=int(sigma * 1.5))
            cloud_layer = np.maximum(cloud_layer, blob)
            covered = int(np.sum(cloud_layer > 0.1) )

        # Blur for smooth edges
        cloud_blurred = cv2.GaussianBlur(cloud_layer, (0, 0), sigmaX=15)
        cloud_blurred = np.clip(cloud_blurred / (cloud_blurred.max() + 1e-8), 0, 1)

        opacity = random.uniform(opacity_min, opacity_max)
        cloud_color = np.array([240, 242, 245], dtype=np.float32)  # slightly off-white

        alpha_map = cloud_blurred * opacity
        for c in range(3):
            occluded[:, :, c] = (
                occluded[:, :, c] * (1 - alpha_map) + cloud_color[c] * alpha_map
            )

        occlusion_mask = (cloud_blurred > 0.15).astype(np.uint8) * 255
        return np.clip(occluded, 0, 255).astype(np.uint8), occlusion_mask

    # ── Visualization ─────────────────────────────────────────

    def _save_comparison(
        self,
        original: np.ndarray,
        occluded: np.ndarray,
        occlusion_mask: np.ndarray,
        road_mask: np.ndarray,
        out_path: Path,
        occ_type: str,
        severity: str,
    ) -> None:
        """Save 3-panel comparison: original | occluded | diff overlay."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.patch.set_facecolor("#0e1117")

        titles = ["Original", f"Occluded ({occ_type})", "Occlusion Mask Overlay"]
        for ax in axes:
            ax.set_facecolor("#0e1117")
            ax.axis("off")

        axes[0].imshow(original)
        axes[0].set_title(titles[0], color="white", fontsize=11)

        axes[1].imshow(occluded)
        axes[1].set_title(titles[1], color="white", fontsize=11)

        # Third panel: show occluded image with road mask and occlusion mask overlaid
        overlay = occluded.copy().astype(np.float32)
        # Road pixels: green tint
        road_px = road_mask > 0
        overlay[road_px, 0] = overlay[road_px, 0] * 0.5
        overlay[road_px, 1] = np.clip(overlay[road_px, 1] * 0.5 + 100, 0, 255)
        overlay[road_px, 2] = overlay[road_px, 2] * 0.5
        # Occluded pixels: red border
        occ_px = (occlusion_mask > 0) & road_px
        overlay[occ_px, 0] = 255
        overlay[occ_px, 1] = 0
        overlay[occ_px, 2] = 0

        axes[2].imshow(np.clip(overlay, 0, 255).astype(np.uint8))
        axes[2].set_title(
            f"{titles[2]}\n(green=road, red=occluded road)",
            color="white", fontsize=11
        )

        fig.suptitle(
            f"Occlusion: {occ_type.replace('_', ' ').title()} — {severity.title()}",
            color="white", fontsize=14, fontweight="bold"
        )
        plt.tight_layout(pad=0.5)
        plt.savefig(str(out_path), dpi=120, bbox_inches="tight", facecolor="#0e1117")
        plt.close()

    # ── Statistics ────────────────────────────────────────────

    @staticmethod
    def _compute_statistics(results: List[OcclusionResult]) -> Dict[str, Any]:
        if not results:
            return {"total_samples": 0}

        by_type: Dict[str, List] = {}
        by_severity: Dict[str, List] = {}

        for r in results:
            by_type.setdefault(r.occlusion_type, []).append(r)
            by_severity.setdefault(r.severity, []).append(r)

        def summarize(recs: List[OcclusionResult]) -> Dict:
            coverages = [r.coverage_pct for r in recs]
            road_covs = [r.road_coverage_pct for r in recs]
            return {
                "count": len(recs),
                "coverage_pct_mean": round(float(np.mean(coverages)), 2),
                "coverage_pct_std": round(float(np.std(coverages)), 2),
                "road_coverage_pct_mean": round(float(np.mean(road_covs)), 2),
            }

        return {
            "total_samples": len(results),
            "by_occlusion_type": {k: summarize(v) for k, v in by_type.items()},
            "by_severity": {k: summarize(v) for k, v in by_severity.items()},
        }


# ─────────────────────────────────────────────────────────────
#  Geometric / noise helpers
# ─────────────────────────────────────────────────────────────

def _perlin_blob(
    h: int, w: int, cx: int, cy: int, radius: int, scale: float = 0.05
) -> np.ndarray:
    """Generate Perlin-noise shaped blob centered at (cx, cy)."""
    blob = np.zeros((h, w), dtype=np.float32)
    offset_x = random.random() * 1000
    offset_y = random.random() * 1000
    for y in range(max(0, cy - radius), min(h, cy + radius)):
        for x in range(max(0, cx - radius), min(w, cx + radius)):
            dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            if dist < radius:
                n = noise.pnoise2(
                    x * scale + offset_x,
                    y * scale + offset_y,
                    octaves=4,
                )
                falloff = 1.0 - (dist / radius) ** 2
                blob[y, x] = max(0.0, (n + 0.5) * falloff)
    return blob


def _gaussian_blob(
    h: int, w: int, cx: int, cy: int, radius: int
) -> np.ndarray:
    """Generate smooth Gaussian blob centered at (cx, cy)."""
    Y, X = np.ogrid[:h, :w]
    dist2 = (X - cx) ** 2 + (Y - cy) ** 2
    sigma = radius / 2.0
    blob = np.exp(-dist2 / (2 * sigma ** 2))
    return blob.astype(np.float32)


def _compute_shadow_polygon(
    ox: int, oy: int,
    bld_w: int, bld_h: int,
    sun_angle_deg: float,
    shadow_len: float,
) -> List[Tuple[int, int]]:
    """
    Compute shadow polygon for a building rect at (ox, oy).

    Returns list of (x, y) polygon vertices.
    """
    angle_rad = np.radians(sun_angle_deg)
    dx = int(shadow_len * np.cos(angle_rad))
    dy = int(shadow_len * np.sin(angle_rad))

    corners = [
        (ox, oy),
        (ox + bld_w, oy),
        (ox + bld_w, oy + bld_h),
        (ox, oy + bld_h),
    ]
    shadow = [(x + dx, y + dy) for x, y in corners]
    return corners + shadow[::-1]


# ─────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────

def run_occlusion(
    config: Dict[str, Any],
    tile_infos: List[Any],
) -> List[OcclusionResult]:
    """
    Main occlusion simulation entry point.

    Args:
        config:     Full pipeline config.
        tile_infos: List of TileInfo (kept tiles with masks).

    Returns:
        List of OcclusionResult metadata.
    """
    simulator = OcclusionSimulator(config)
    results = simulator.simulate_all(tile_infos)
    return results
