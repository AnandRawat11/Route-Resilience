"""
Route Resilience — src/core/io.py

All file I/O helpers: image loading/saving, JSON, geospatial metadata,
directory creation, file integrity, and misc utilities.

This is the single I/O dependency for all pipeline modules.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────
#  Directory helpers
# ─────────────────────────────────────────────────────────────

def ensure_dir(path: Union[str, Path]) -> Path:
    """Create directory (and parents) if it does not exist. Returns Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ─────────────────────────────────────────────────────────────
#  Image I/O
# ─────────────────────────────────────────────────────────────

def load_image(
    path: Union[str, Path],
    as_float: bool = False,
) -> np.ndarray:
    """
    Load an image from disk. Supports PNG, JPEG, and GeoTIFF.

    For GeoTIFF files, tries rasterio first; falls back to OpenCV.

    Args:
        path:     Path to image file.
        as_float: If True, normalise to [0.0, 1.0] float32.

    Returns:
        numpy array (H, W, C) in RGB order, dtype uint8 or float32.

    Raises:
        FileNotFoundError: If file does not exist.
        RuntimeError:      If image cannot be decoded.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    img: Optional[np.ndarray] = None

    # Try rasterio for GeoTIFF
    if path.suffix.lower() in (".tif", ".tiff"):
        try:
            import rasterio  # type: ignore
            with rasterio.open(path) as src:
                bands = min(src.count, 3)
                data  = src.read(list(range(1, bands + 1)))   # (C, H, W)
                img   = np.transpose(data, (1, 2, 0))         # (H, W, C)
                if img.dtype != np.uint8:
                    img = _stretch_to_uint8(img)
        except Exception:
            img = None

    # Fallback: OpenCV
    if img is None:
        raw = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if raw is None:
            raise RuntimeError(f"Cannot decode image: {path}")
        img = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)

    # Ensure 3 channels
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.shape[2] > 3:
        img = img[:, :, :3]

    if as_float:
        img = img.astype(np.float32) / 255.0

    return img


def load_mask(path: Union[str, Path]) -> np.ndarray:
    """
    Load a binary road mask from disk.

    Returns:
        numpy array (H, W), dtype uint8, values 0 or 255.

    Raises:
        FileNotFoundError: If file does not exist.
        RuntimeError:      If mask cannot be decoded.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Mask not found: {path}")

    if path.suffix.lower() in (".tif", ".tiff"):
        try:
            import rasterio  # type: ignore
            with rasterio.open(path) as src:
                data = src.read(1)
                mask = np.where(data > 0, 255, 0).astype(np.uint8)
                return mask
        except Exception:
            pass

    raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise RuntimeError(f"Cannot decode mask: {path}")
    return np.where(raw > 127, 255, 0).astype(np.uint8)


def save_image(
    img: np.ndarray,
    path: Union[str, Path],
    is_mask: bool = False,
) -> None:
    """
    Save a numpy image array to disk.

    Args:
        img:     (H, W, C) RGB or (H, W) for masks.
        path:    Destination path.
        is_mask: If True, treated as single-channel mask.
    """
    path = Path(path)
    ensure_dir(path.parent)
    if is_mask:
        if img.ndim == 3:
            img = img[:, :, 0]
        cv2.imwrite(str(path), img)
    else:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.ndim == 3 else img
        cv2.imwrite(str(path), img_bgr)


def _stretch_to_uint8(img: np.ndarray) -> np.ndarray:
    """Percentile-stretch a multi-band image to uint8."""
    n_bands = min(img.shape[2], 3) if img.ndim == 3 else 1
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    out = np.zeros((*img.shape[:2], n_bands), dtype=np.uint8)
    for i in range(n_bands):
        band = img[:, :, i].astype(np.float64)
        lo, hi = np.percentile(band, 2), np.percentile(band, 98)
        if hi > lo:
            band = np.clip((band - lo) / (hi - lo) * 255, 0, 255)
        else:
            band = np.zeros_like(band)
        out[:, :, i] = band.astype(np.uint8)
    return out


# ─────────────────────────────────────────────────────────────
#  JSON helpers
# ─────────────────────────────────────────────────────────────

def save_json(data: Any, path: Union[str, Path], indent: int = 2) -> None:
    """Serialise *data* to a pretty-printed JSON file."""
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=indent, default=_json_serializer)


def load_json(path: Union[str, Path]) -> Any:
    """Load and parse a JSON file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _json_serializer(obj: Any) -> Any:
    """Custom JSON serialiser for numpy types and Path objects."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")


# ─────────────────────────────────────────────────────────────
#  Geospatial metadata
# ─────────────────────────────────────────────────────────────

def extract_geospatial_metadata(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Extract geospatial metadata from a raster file using rasterio.

    Returns a dict with: crs, bounds, transform, resolution, width, height, count.
    For non-geo images returns defaults with ``is_georeferenced=False``.
    """
    path = Path(path)
    meta: Dict[str, Any] = {
        "file":              str(path),
        "is_georeferenced":  False,
        "crs":               None,
        "bounds":            None,
        "transform":         None,
        "resolution_x":      None,
        "resolution_y":      None,
        "width":             None,
        "height":            None,
        "band_count":        None,
        "dtype":             None,
        "nodata":            None,
    }

    if not path.exists():
        return meta

    try:
        import rasterio  # type: ignore
        with rasterio.open(path) as src:
            meta["width"]      = src.width
            meta["height"]     = src.height
            meta["band_count"] = src.count
            meta["dtype"]      = str(src.dtypes[0])
            meta["nodata"]     = src.nodata

            if src.crs is not None:
                meta["is_georeferenced"] = True
                meta["crs"]              = src.crs.to_string()
                b                        = src.bounds
                meta["bounds"]           = {"left": b.left, "bottom": b.bottom,
                                             "right": b.right, "top": b.top}
                t = src.transform
                meta["transform"]        = {"a": t.a, "b": t.b, "c": t.c,
                                             "d": t.d, "e": t.e, "f": t.f}
                meta["resolution_x"]     = abs(t.a)
                meta["resolution_y"]     = abs(t.e)
    except Exception:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is not None:
            meta["height"]     = img.shape[0]
            meta["width"]      = img.shape[1]
            meta["band_count"] = 1 if img.ndim == 2 else img.shape[2]
            meta["dtype"]      = str(img.dtype)

    return meta


# ─────────────────────────────────────────────────────────────
#  File integrity
# ─────────────────────────────────────────────────────────────

def compute_md5(path: Union[str, Path], chunk_size: int = 65_536) -> str:
    """Return hex MD5 digest of a file."""
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def file_is_readable(path: Union[str, Path]) -> bool:
    """Return True if *path* exists and is readable."""
    p = Path(path)
    return p.exists() and p.is_file() and os.access(p, os.R_OK)


# ─────────────────────────────────────────────────────────────
#  Misc
# ─────────────────────────────────────────────────────────────

def format_bytes(n: int) -> str:
    """Return human-readable file size string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


@contextlib.contextmanager
def timer(label: str = ""):
    """Context manager that prints elapsed time."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"[{label}] elapsed: {elapsed:.2f}s")
