"""
Route Resilience — Phase 1
src/vector_utils.py

Vector Data Support:
  - Read GeoJSON road vectors
  - Read Shapefile road vectors
  - Convert vector roads → binary raster masks (for OSM integration)
  - Convert binary masks → vector lines (for graph reconstruction)
  - Coordinate reprojection utilities
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from src.core.io     import ensure_dir, save_json
from src.core.logger import get_logger

logger = get_logger(__name__)

# Optional geospatial deps — graceful import
try:
    import fiona  # type: ignore
    import geopandas as gpd  # type: ignore
    import rasterio  # type: ignore
    from rasterio import features as rfeatures  # type: ignore
    from rasterio.transform import from_bounds  # type: ignore
    from pyproj import Transformer  # type: ignore
    from shapely.geometry import (  # type: ignore
        LineString, MultiLineString, shape, mapping
    )
    _HAS_GEO = True
except ImportError as _geo_err:
    _HAS_GEO = False
    logger.warning(
        f"Geospatial libraries not fully available ({_geo_err}). "
        "Vector conversion features will be limited. "
        "Install: pip install geopandas fiona rasterio pyproj shapely"
    )


# ─────────────────────────────────────────────────────────────
#  VectorLoader — read GeoJSON / Shapefile
# ─────────────────────────────────────────────────────────────

class VectorLoader:
    """Reads road vector files (GeoJSON, Shapefile) into a GeoDataFrame."""

    @staticmethod
    def load(path: Union[str, Path]) -> Any:
        """
        Load a vector file into a GeoDataFrame.

        Args:
            path: Path to .geojson or .shp file.

        Returns:
            geopandas.GeoDataFrame

        Raises:
            ImportError:  If geopandas/fiona not installed.
            FileNotFoundError: If file does not exist.
            ValueError:   If file is empty or unreadable.
        """
        if not _HAS_GEO:
            raise ImportError(
                "geopandas and fiona are required for vector loading. "
                "Install: pip install geopandas fiona"
            )

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Vector file not found: {path}")

        try:
            gdf = gpd.read_file(str(path))
        except Exception as exc:
            raise ValueError(f"Cannot read vector file {path}: {exc}") from exc

        if gdf.empty:
            raise ValueError(f"Vector file is empty: {path}")

        logger.info(
            f"Loaded {len(gdf)} features from {path.name} "
            f"(CRS: {gdf.crs})"
        )
        return gdf

    @staticmethod
    def load_all_from_dir(
        vector_dir: Union[str, Path],
        extensions: Tuple[str, ...] = (".geojson", ".shp"),
    ) -> Any:
        """
        Load and concatenate all vector files in a directory.

        Returns:
            Single merged GeoDataFrame (or None if no files).
        """
        if not _HAS_GEO:
            raise ImportError("geopandas required. Install: pip install geopandas fiona")

        vector_dir = Path(vector_dir)
        if not vector_dir.exists():
            logger.warning(f"Vector directory not found: {vector_dir}")
            return None

        files = [f for f in vector_dir.iterdir() if f.suffix.lower() in extensions]
        if not files:
            logger.warning(f"No vector files found in {vector_dir}")
            return None

        gdfs = []
        for f in files:
            try:
                gdf = VectorLoader.load(f)
                gdfs.append(gdf)
            except Exception as exc:
                logger.error(f"Failed to load {f}: {exc}")

        if not gdfs:
            return None

        merged = gpd.pd.concat(gdfs, ignore_index=True)
        return gpd.GeoDataFrame(merged, crs=gdfs[0].crs)


# ─────────────────────────────────────────────────────────────
#  VectorToMask — rasterize road vectors → binary mask
# ─────────────────────────────────────────────────────────────

class VectorToMask:
    """
    Converts vector road lines (GeoJSON/Shapefile) into binary raster masks.

    Used for:
      - Generating road masks from OSM road vectors
      - Creating ground truth masks for areas with no labeled raster
    """

    def __init__(self, buffer_meters: float = 3.0) -> None:
        """
        Args:
            buffer_meters: Buffer radius (meters) applied around road centerlines
                           to produce filled road masks. Should match typical road width.
        """
        self.buffer_meters = buffer_meters

    def convert(
        self,
        gdf: Any,
        output_crs: str = "EPSG:4326",
        width: int = 1024,
        height: int = 1024,
        bounds: Optional[Tuple[float, float, float, float]] = None,
        out_path: Optional[Union[str, Path]] = None,
    ) -> np.ndarray:
        """
        Rasterize a GeoDataFrame of road lines into a binary mask.

        Args:
            gdf:        GeoDataFrame with road geometries.
            output_crs: Target CRS string (e.g., "EPSG:4326").
            width:      Output mask width in pixels.
            height:     Output mask height in pixels.
            bounds:     (left, bottom, right, top) in output_crs coordinates.
                        If None, inferred from gdf extent.
            out_path:   If provided, saves mask to this path.

        Returns:
            Binary numpy array (H, W), dtype uint8, values 0 or 255.
        """
        if not _HAS_GEO:
            raise ImportError("rasterio, geopandas, shapely required.")

        # Reproject to output CRS
        if gdf.crs is not None and str(gdf.crs) != output_crs:
            gdf = gdf.to_crs(output_crs)

        # Buffer to get road width
        # Work in a projected CRS for accurate meter buffering
        try:
            gdf_proj = gdf.to_crs("EPSG:3857")  # Web Mercator
            gdf_proj["geometry"] = gdf_proj["geometry"].buffer(self.buffer_meters)
            gdf = gdf_proj.to_crs(output_crs)
        except Exception as exc:
            logger.warning(f"Buffering failed ({exc}), using unbuffered geometry.")

        # Compute bounds
        if bounds is None:
            b = gdf.total_bounds  # (minx, miny, maxx, maxy)
            bounds = (b[0], b[1], b[2], b[3])

        left, bottom, right, top = bounds
        transform = from_bounds(left, bottom, right, top, width, height)

        shapes = [(geom, 255) for geom in gdf.geometry if geom is not None and not geom.is_empty]

        if not shapes:
            logger.warning("No valid geometries to rasterize.")
            return np.zeros((height, width), dtype=np.uint8)

        mask = rfeatures.rasterize(
            shapes=shapes,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=np.uint8,
            all_touched=True,
        )

        if out_path:
            out_path = Path(out_path)
            ensure_dir(out_path.parent)
            cv2.imwrite(str(out_path), mask)
            logger.info(f"Rasterized mask saved → {out_path}")

        return mask

    def batch_convert(
        self,
        vector_dir: Union[str, Path],
        out_dir: Union[str, Path],
        width: int = 1024,
        height: int = 1024,
    ) -> List[str]:
        """
        Convert all vector files in a directory to masks.

        Returns:
            List of output mask file paths.
        """
        gdf = VectorLoader.load_all_from_dir(vector_dir)
        if gdf is None:
            logger.warning("No vector data to convert.")
            return []

        out_dir = ensure_dir(out_dir)
        out_path = out_dir / "osm_road_mask.png"
        self.convert(gdf, width=width, height=height, out_path=out_path)
        return [str(out_path)]


# ─────────────────────────────────────────────────────────────
#  MaskToVector — extract road centerlines as GeoJSON
# ─────────────────────────────────────────────────────────────

class MaskToVector:
    """
    Converts binary road masks back to vector lines.

    Used for:
      - Preparing input for graph reconstruction (Phase 2)
      - OSM comparison / validation
    """

    @staticmethod
    def convert(
        mask: np.ndarray,
        transform: Optional[Any] = None,
        crs: Optional[str] = None,
        out_path: Optional[Union[str, Path]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Skeletonize mask and extract road centerline vectors as GeoJSON features.

        Args:
            mask:      (H, W) uint8 binary mask (0 or 255).
            transform: Rasterio affine transform for pixel→geo coordinate mapping.
            crs:       CRS string for output GeoJSON.
            out_path:  If provided, saves GeoJSON to this path.

        Returns:
            List of GeoJSON Feature dicts.
        """
        from skimage.morphology import skeletonize  # type: ignore

        binary = (mask > 0).astype(np.uint8)
        if binary.sum() == 0:
            logger.warning("Empty mask — no roads to vectorize.")
            return []

        # Skeletonize to 1-pixel centerlines
        skeleton = skeletonize(binary).astype(np.uint8) * 255

        # Find contours of skeleton segments
        contours, _ = cv2.findContours(skeleton, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        features = []
        for cnt in contours:
            pts = cnt.reshape(-1, 2)
            if len(pts) < 2:
                continue

            if transform is not None:
                # Convert pixel coords to geo coords
                from rasterio.transform import xy  # type: ignore
                geo_pts = [
                    list(xy(transform, int(p[1]), int(p[0])))
                    for p in pts
                ]
            else:
                geo_pts = [[int(p[0]), int(p[1])] for p in pts]

            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": geo_pts,
                },
                "properties": {
                    "source": "route_resilience_mask",
                    "crs": crs or "pixel",
                },
            }
            features.append(feature)

        if out_path:
            out_path = Path(out_path)
            ensure_dir(out_path.parent)
            geojson = {
                "type": "FeatureCollection",
                "crs": {"type": "name", "properties": {"name": crs or "pixel"}},
                "features": features,
            }
            save_json(geojson, out_path)
            logger.info(f"Road vectors saved → {out_path} ({len(features)} segments)")

        return features

    @staticmethod
    def from_file(
        mask_path: Union[str, Path],
        geo_meta_path: Optional[Union[str, Path]] = None,
        out_path: Optional[Union[str, Path]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Load a mask file and convert to vector.

        Args:
            mask_path:     Path to binary mask PNG/TIFF.
            geo_meta_path: Optional path to per-image _geo.json metadata file.
                           If provided, uses stored CRS and transform for geo coords.
            out_path:      Optional output GeoJSON path.
        """
        mask_path = Path(mask_path)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Cannot read mask: {mask_path}")

        transform = None
        crs = None

        if geo_meta_path:
            try:
                from src.core.io import ensure_dir, save_json
                import rasterio.transform as rt  # type: ignore
                meta = load_json(geo_meta_path)
                crs = meta.get("crs")
                t = meta.get("transform")
                if t:
                    transform = rt.Affine(t["a"], t["b"], t["c"], t["d"], t["e"], t["f"])
            except Exception as exc:
                logger.warning(f"Could not load geo metadata for vectorization: {exc}")

        return MaskToVector.convert(mask, transform=transform, crs=crs, out_path=out_path)


# ─────────────────────────────────────────────────────────────
#  OSM road mask generator (Overpass API)
# ─────────────────────────────────────────────────────────────

def fetch_osm_road_mask(
    bbox: Tuple[float, float, float, float],
    out_mask_path: Union[str, Path],
    out_vector_path: Optional[Union[str, Path]] = None,
    width: int = 1024,
    height: int = 1024,
    buffer_meters: float = 3.0,
) -> Optional[np.ndarray]:
    """
    Fetch road vectors from OpenStreetMap Overpass API and rasterize to mask.

    Args:
        bbox:           (south, west, north, east) in WGS84.
        out_mask_path:  Save path for rasterized mask PNG.
        out_vector_path: Optional save path for road GeoJSON.
        width, height:  Output mask dimensions.
        buffer_meters:  Road width buffer in meters.

    Returns:
        Binary mask numpy array, or None on failure.
    """
    import requests  # type: ignore

    south, west, north, east = bbox
    overpass_query = f"""
    [out:json][timeout:60];
    (
      way["highway"~"motorway|trunk|primary|secondary|tertiary|residential|unclassified|road"]
         ({south},{west},{north},{east});
    );
    out geom;
    """

    logger.info(f"Fetching OSM roads for bbox={bbox}…")
    try:
        resp = requests.post(
            "https://overpass-api.de/api/interpreter",
            data={"data": overpass_query},
            timeout=90,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.error(f"OSM fetch failed: {exc}")
        return None

    # Parse ways into LineString geometries
    features = []
    for element in data.get("elements", []):
        if element.get("type") == "way" and "geometry" in element:
            coords = [(pt["lon"], pt["lat"]) for pt in element["geometry"]]
            if len(coords) >= 2:
                feature = {
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": coords},
                    "properties": {
                        "highway": element.get("tags", {}).get("highway", "road"),
                        "name": element.get("tags", {}).get("name", ""),
                        "osm_id": element.get("id"),
                    },
                }
                features.append(feature)

    if not features:
        logger.warning("No OSM road features found for given bbox.")
        return None

    geojson_data = {"type": "FeatureCollection", "features": features}

    # Save vector if requested
    if out_vector_path:
        save_json(geojson_data, out_vector_path)
        logger.info(f"OSM road vectors saved → {out_vector_path} ({len(features)} roads)")

    # Rasterize
    if not _HAS_GEO:
        logger.error("geopandas required to rasterize OSM data.")
        return None

    gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
    converter = VectorToMask(buffer_meters=buffer_meters)
    bounds = (west, south, east, north)
    mask = converter.convert(gdf, output_crs="EPSG:4326", width=width, height=height, bounds=bounds)

    cv2.imwrite(str(out_mask_path), mask)
    logger.info(f"OSM road mask saved → {out_mask_path}")
    return mask


# ─────────────────────────────────────────────────────────────
#  CLI helper
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Vector utilities for Route Resilience Phase 1"
    )
    subparsers = parser.add_subparsers(dest="command")

    # Convert vector → mask
    p_v2m = subparsers.add_parser("vector-to-mask", help="Convert GeoJSON/Shapefile to mask")
    p_v2m.add_argument("--input", required=True, help="Input vector file or directory")
    p_v2m.add_argument("--output", required=True, help="Output mask path")
    p_v2m.add_argument("--width", type=int, default=1024)
    p_v2m.add_argument("--height", type=int, default=1024)
    p_v2m.add_argument("--buffer", type=float, default=3.0, help="Road buffer in meters")

    # Convert mask → vector
    p_m2v = subparsers.add_parser("mask-to-vector", help="Convert binary mask to GeoJSON")
    p_m2v.add_argument("--mask", required=True, help="Input mask PNG/TIFF")
    p_m2v.add_argument("--output", required=True, help="Output GeoJSON path")
    p_m2v.add_argument("--geo-meta", default=None, help="Optional _geo.json metadata file")

    # Fetch OSM
    p_osm = subparsers.add_parser("fetch-osm", help="Fetch OSM road mask for a bounding box")
    p_osm.add_argument("--bbox", required=True, nargs=4, type=float,
                       metavar=("SOUTH", "WEST", "NORTH", "EAST"))
    p_osm.add_argument("--output", required=True, help="Output mask PNG path")
    p_osm.add_argument("--vector-output", default=None, help="Output road GeoJSON path")
    p_osm.add_argument("--width", type=int, default=1024)
    p_osm.add_argument("--height", type=int, default=1024)

    args = parser.parse_args()

    if args.command == "vector-to-mask":
        converter = VectorToMask(buffer_meters=args.buffer)
        p = Path(args.input)
        if p.is_dir():
            converter.batch_convert(p, Path(args.output), args.width, args.height)
        else:
            gdf = VectorLoader.load(p)
            converter.convert(gdf, width=args.width, height=args.height, out_path=args.output)

    elif args.command == "mask-to-vector":
        MaskToVector.from_file(args.mask, args.geo_meta, args.output)

    elif args.command == "fetch-osm":
        fetch_osm_road_mask(
            bbox=tuple(args.bbox),
            out_mask_path=args.output,
            out_vector_path=args.vector_output,
            width=args.width,
            height=args.height,
        )
    else:
        parser.print_help()
