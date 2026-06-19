"""
Route Resilience — Phase 2
src/segmentation/datasets/road_dataset.py

Custom PyTorch Dataset and on-the-fly Synthetic Occlusion Transforms.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Import perlin noise package with graceful fallback
try:
    import noise  # type: ignore
    _HAS_NOISE = True
except ImportError:
    _HAS_NOISE = False

# Default occlusion configurations matching configs/config.yaml
DEFAULT_OCCLUSION_CFG = {
    "tree_canopy": {
        "light": {"coverage_min": 0.10, "coverage_max": 0.25},
        "medium": {"coverage_min": 0.25, "coverage_max": 0.50},
        "heavy": {"coverage_min": 0.50, "coverage_max": 0.75},
        "noise_scale": 0.05,
        "blob_size_range": [30, 120],
        "color_variation": 20
    },
    "building_shadow": {
        "light": {"coverage_min": 0.05, "coverage_max": 0.20},
        "medium": {"coverage_min": 0.20, "coverage_max": 0.40},
        "heavy": {"coverage_min": 0.40, "coverage_max": 0.65},
        "sun_angle_range": [30, 150],
        "shadow_length_range": [0.5, 2.0],
        "darkening_factor": 0.35
    },
    "vehicle": {
        "light": {"count_range": [1, 3]},
        "medium": {"count_range": [3, 8]},
        "heavy": {"count_range": [8, 20]},
        "size_range_w": [8, 25],
        "size_range_h": [15, 45]
    },
    "cloud_cover": {
        "light": {"coverage_min": 0.05, "coverage_max": 0.20, "opacity_range": [0.2, 0.5]},
        "medium": {"coverage_min": 0.20, "coverage_max": 0.45, "opacity_range": [0.5, 0.75]},
        "heavy": {"coverage_min": 0.45, "coverage_max": 0.70, "opacity_range": [0.75, 0.95]},
        "blur_sigma_range": [20, 60]
    }
}


# ─────────────────────────────────────────────────────────────
#  Occlusion Helpers
# ─────────────────────────────────────────────────────────────

def _gaussian_blob(h: int, w: int, cx: int, cy: int, radius: int) -> np.ndarray:
    """Generate smooth Gaussian blob centered at (cx, cy)."""
    Y, X = np.ogrid[:h, :w]
    dist2 = (X - cx) ** 2 + (Y - cy) ** 2
    sigma = max(1.0, radius / 2.0)
    blob = np.exp(-dist2 / (2 * sigma ** 2))
    return blob.astype(np.float32)


def _perlin_blob(h: int, w: int, cx: int, cy: int, radius: int, scale: float = 0.05) -> np.ndarray:
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


def _compute_shadow_polygon(
    ox: int, oy: int, bld_w: int, bld_h: int, sun_angle_deg: float, shadow_len: float
) -> List[Tuple[int, int]]:
    """Compute shadow polygon vertices for building corner at (ox, oy)."""
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


def apply_tree_canopy(img: np.ndarray, mask: np.ndarray, severity: str, cfg: Dict) -> np.ndarray:
    """Simulate tree canopy using Perlin noise or Gaussian fallback."""
    h, w = img.shape[:2]
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
    max_attempts = 100
    attempts = 0

    while covered / total_px < target_coverage and attempts < max_attempts:
        attempts += 1
        blob_r = random.randint(blob_min // 2, blob_max // 2)
        cx = random.randint(blob_r, w - blob_r)
        cy = random.randint(blob_r, h - blob_r)

        if _HAS_NOISE:
            blob = _perlin_blob(h, w, cx, cy, blob_r, scale=cfg.get("noise_scale", 0.05))
        else:
            blob = _gaussian_blob(h, w, cx, cy, blob_r)

        blob = (blob > 0.35).astype(np.uint8)

        base_green = np.array([34, 85, 34], dtype=np.float32)
        color = base_green + np.random.uniform(-color_var, color_var, 3)
        color = np.clip(color, 0, 255)

        alpha = np.where(blob, random.uniform(0.55, 0.80), 0.0)
        alpha = np.expand_dims(alpha, axis=-1)

        occluded = occluded * (1 - alpha) + color * alpha
        occlusion_mask = np.maximum(occlusion_mask, (blob * 255).astype(np.uint8))
        covered = int(np.sum(occlusion_mask > 0))

    return np.clip(occluded, 0, 255).astype(np.uint8)


def apply_building_shadow(img: np.ndarray, mask: np.ndarray, severity: str, cfg: Dict) -> np.ndarray:
    """Simulate building shadows as dark polygons."""
    h, w = img.shape[:2]
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

        dist_map = cv2.distanceTransform(shadow_img, cv2.DIST_L2, 5)
        if dist_map.max() > 0:
            gradient = 1.0 - (dist_map / dist_map.max()) * 0.4
        else:
            gradient = np.ones((h, w), dtype=np.float32)

        alpha = np.where(shadow_img > 0, gradient * (1.0 - darkening), 0.0)
        alpha = np.expand_dims(alpha, axis=-1)

        occluded = occluded * (1.0 - alpha * (1.0 - darkening))
        occlusion_mask = np.maximum(occlusion_mask, shadow_img)
        covered = int(np.sum(occlusion_mask > 0))

    return np.clip(occluded, 0, 255).astype(np.uint8)


def apply_vehicle(img: np.ndarray, mask: np.ndarray, severity: str, cfg: Dict) -> np.ndarray:
    """Simulate vehicles as small rectangles placed on road pixels."""
    h, w = img.shape[:2]
    sev_cfg = cfg.get(severity, {})
    count_range = sev_cfg.get("count_range", [3, 8])
    veh_w_range = cfg.get("size_range_w", [8, 25])
    veh_h_range = cfg.get("size_range_h", [15, 45])

    n_vehicles = random.randint(*count_range)
    occluded = img.copy()

    road_ys, road_xs = np.where(mask > 0)
    if len(road_ys) == 0:
        road_ys = np.arange(0, h)
        road_xs = np.arange(0, w)

    vehicle_colors = [
        (200, 200, 200),  # silver
        (40, 40, 40),     # black
        (180, 30, 30),    # red
        (30, 30, 160),    # blue
        (200, 180, 30),   # yellow
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
        cv2.rectangle(
            occluded,
            (x1 + 1, y1 + 1),
            (x2 - 1, y1 + max(2, vh // 6)),
            tuple(min(255, c + 60) for c in color),
            -1,
        )

    return occluded


def apply_cloud_cover(img: np.ndarray, severity: str, cfg: Dict) -> np.ndarray:
    """Simulate cloud cover using Gaussian blurred blobs."""
    h, w = img.shape[:2]
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
        covered = int(np.sum(cloud_layer > 0.1))

    cloud_blurred = cv2.GaussianBlur(cloud_layer, (0, 0), sigmaX=15)
    cloud_blurred = np.clip(cloud_blurred / (cloud_blurred.max() + 1e-8), 0, 1)

    opacity = random.uniform(opacity_min, opacity_max)
    cloud_color = np.array([240, 242, 245], dtype=np.float32)

    alpha_map = np.expand_dims(cloud_blurred * opacity, axis=-1)
    occluded = occluded * (1.0 - alpha_map) + cloud_color * alpha_map

    return np.clip(occluded, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────
#  Transform & Dataset Classes
# ─────────────────────────────────────────────────────────────

class SyntheticOcclusionTransform:
    """
    Applies synthetic occlusions (Tree canopy, Shadow, Vehicle, Cloud)
    on-the-fly to satellite images during training.
    """
    def __init__(self, p: float = 0.5, types: Optional[List[str]] = None, config: Optional[Dict] = None):
        self.p = p
        self.types = types or ["tree_canopy", "building_shadow", "vehicle", "cloud_cover"]
        self.config = config or DEFAULT_OCCLUSION_CFG

    def __call__(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() > self.p:
            return img, mask

        occ_type = random.choice(self.types)
        severity = random.choice(["light", "medium", "heavy"])

        if occ_type == "tree_canopy":
            img = apply_tree_canopy(img, mask, severity, self.config["tree_canopy"])
        elif occ_type == "building_shadow":
            img = apply_building_shadow(img, mask, severity, self.config["building_shadow"])
        elif occ_type == "vehicle":
            img = apply_vehicle(img, mask, severity, self.config["vehicle"])
        elif occ_type == "cloud_cover":
            img = apply_cloud_cover(img, severity, self.config["cloud_cover"])

        return img, mask


def resolve_tile_path(csv_path_val: str, dataset_dir: str) -> str:
    """
    Resolves tile paths dynamically. Handles local development structures
    and maps them to the correct Kaggle directories.
    """
    if not csv_path_val:
        return ""

    basename = os.path.basename(csv_path_val)
    is_mask = "masks" in csv_path_val or "mask" in basename.lower()
    subdir = "tiles/train/masks" if is_mask else "tiles/train/images"

    # Candidate 1: Standard Kaggle structure
    candidate1 = os.path.join(dataset_dir, subdir, basename)
    if os.path.exists(candidate1):
        return candidate1

    # Candidate 2: Replacing "data/" prefix with dataset_dir
    norm_path = os.path.normpath(csv_path_val)
    parts = norm_path.split(os.sep)
    if "data" in parts:
        data_idx = parts.index("data")
        candidate2 = os.path.join(dataset_dir, *parts[data_idx + 1:])
        if os.path.exists(candidate2):
            return candidate2

    # Candidate 3: Try relative to active workspace directory
    if os.path.exists(csv_path_val):
        return csv_path_val

    # Fallback to candidate 1 path even if file is missing (to avoid silent failures)
    return candidate1


class RoadDataset(Dataset):
    """
    PyTorch Dataset for DeepGlobe roads with support for on-the-fly occlusions.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        dataset_dir: str,
        transform: Optional[Any] = None,
        occlusion_transform: Optional[SyntheticOcclusionTransform] = None
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.occlusion_transform = occlusion_transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]

        img_path = resolve_tile_path(row["image_tile_path"], self.dataset_dir)
        msk_path = resolve_tile_path(row["mask_tile_path"], self.dataset_dir)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image tile not found: {img_path}")
        if not os.path.exists(msk_path):
            raise FileNotFoundError(f"Mask tile not found: {msk_path}")

        # Load image (BGR to RGB)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask (grayscale)
        mask = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.float32)

        # Apply synthetic occlusion on-the-fly (training only)
        if self.occlusion_transform is not None:
            image, mask = self.occlusion_transform(image, mask)

        # Apply Albumentations (augmentations and normalization)
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # Ensure correct dimensions
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image.transpose(2, 0, 1))
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask)
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)

        # Make sure mask values are strictly 0.0 or 1.0 float32
        mask = (mask > 0.5).float()

        return image, mask
