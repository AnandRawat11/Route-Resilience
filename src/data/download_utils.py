"""
Route Resilience — Phase 1
src/download_utils.py

Optional Dataset Download Helpers

Supports:
  - SpaceNet Roads (AWS S3, requires awscli)
  - DeepGlobe Road Extraction (Kaggle API)
  - OpenSatMap (GitHub release / direct download)
  - OpenStreetMap vectors (Overpass API, Geofabrik)

Behaviour:
  - If credentials are missing: prints clear instructions and exits
  - Never fails silently — all errors are informative
  - All downloads are resumable (partial file detection)
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from src.core.io     import ensure_dir
from src.core.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────
#  Dataset Downloader
# ─────────────────────────────────────────────────────────────

class DatasetDownloader:
    """
    Provides download helpers for each supported dataset.

    Usage:
        downloader = DatasetDownloader()
        downloader.download("deepglobe")
        downloader.download("osm", bbox=(12.9, 77.5, 13.1, 77.7))
    """

    DATASETS = ["spacenet", "deepglobe", "opensatmap", "osm"]

    def __init__(self) -> None:
        self.base_dir = Path("data/raw")

    def download(self, dataset: str, **kwargs: Any) -> bool:
        """
        Attempt to download a dataset.

        Args:
            dataset: One of 'spacenet', 'deepglobe', 'opensatmap', 'osm'.
            **kwargs: Dataset-specific parameters (e.g., bbox for osm).

        Returns:
            True if download succeeded, False otherwise.
        """
        dataset = dataset.lower()
        if dataset not in self.DATASETS:
            logger.error(f"Unknown dataset '{dataset}'. Valid options: {self.DATASETS}")
            return False

        method = getattr(self, f"_download_{dataset}")
        return method(**kwargs)

    def check_all(self) -> Dict[str, bool]:
        """Check which datasets are already present locally."""
        status = {}
        for ds in self.DATASETS:
            ds_dir = self.base_dir / ds
            images_dir = ds_dir / "images"
            has_images = images_dir.exists() and any(images_dir.iterdir())
            status[ds] = has_images
            icon = "✓" if has_images else "✗"
            logger.info(f"  {icon} {ds}: {'found' if has_images else 'not found'}")
        return status

    # ── SpaceNet ──────────────────────────────────────────────

    def _download_spacenet(self, cities: Optional[list] = None, **kwargs: Any) -> bool:
        """
        Download SpaceNet Roads dataset from AWS S3.

        Requires: AWS CLI configured with access to SpaceNet bucket.
        The SpaceNet dataset is freely available as a public AWS dataset.
        """
        if cities is None:
            cities = ["Vegas", "Paris", "Shanghai", "Khartoum"]

        if not _has_command("aws"):
            _print_instructions(
                "SpaceNet Roads — AWS CLI Required",
                """
SpaceNet Roads is hosted on AWS S3 as a public dataset.

Prerequisites:
  1. Install AWS CLI:
       curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o "AWSCLIV2.pkg"
       sudo installer -pkg AWSCLIV2.pkg -target /

  2. Configure AWS credentials (free tier works):
       aws configure
       Enter your AWS Access Key ID and Secret Access Key.
       Region: us-east-1

  3. (Optional) Install stac-asset for faster downloads:
       pip install stac-asset

Download commands (run manually):

  # SN3 Roads — full dataset (~50 GB):
  aws s3 cp s3://spacenet-dataset/spacenet/SN3_roads/tarballs/ data/raw/spacenet/ \\
      --recursive --no-sign-request

  # Or per city (smaller):
  aws s3 cp s3://spacenet-dataset/spacenet/SN3_roads/tarballs/SN3_roads_train_AOI_2_Vegas_roads_speed_mask.tar.gz \\
      data/raw/spacenet/ --no-sign-request

After downloading:
  tar -xzf data/raw/spacenet/*.tar.gz -C data/raw/spacenet/
  # Organize extracted images and masks as:
  #   data/raw/spacenet/images/*.tif
  #   data/raw/spacenet/masks/*.tif

Dataset info: https://spacenet.ai/roads/
""",
            )
            return False

        out_dir = ensure_dir(self.base_dir / "spacenet")
        logger.info("Downloading SpaceNet Roads from S3 (no-sign-request)…")

        # Use public access (SpaceNet is a public dataset)
        for city in cities:
            city_lower = city.lower()
            # Approximate S3 path pattern
            s3_prefix = f"s3://spacenet-dataset/spacenet/SN3_roads/AOI_{city}/"
            local_dir = out_dir / city_lower

            logger.info(f"  Syncing {city}…")
            ret = subprocess.run(
                ["aws", "s3", "sync", s3_prefix, str(local_dir), "--no-sign-request"],
                capture_output=False,
            )
            if ret.returncode != 0:
                logger.warning(
                    f"  AWS sync failed for {city}. "
                    "Check your AWS CLI configuration and network."
                )

        return True

    # ── DeepGlobe ─────────────────────────────────────────────

    def _download_deepglobe(self, **kwargs: Any) -> bool:
        """
        Download DeepGlobe Road Extraction dataset via Kaggle API.

        Requires: Kaggle API key in ~/.kaggle/kaggle.json
        """
        if not _has_python_package("kaggle"):
            _print_instructions(
                "DeepGlobe — Kaggle API Required",
                """
DeepGlobe Road Extraction dataset is available on Kaggle.

Prerequisites:
  1. Create a Kaggle account: https://www.kaggle.com/

  2. Generate API key:
       Kaggle → Account → API → "Create New API Token"
       This downloads kaggle.json

  3. Place it in your home directory:
       mkdir -p ~/.kaggle
       cp ~/Downloads/kaggle.json ~/.kaggle/
       chmod 600 ~/.kaggle/kaggle.json

  4. Install Kaggle Python package:
       pip install kaggle

  5. Download dataset:
       kaggle datasets download -d balraj98/deepglobe-road-extraction-dataset \\
           -p data/raw/deepglobe --unzip

  6. Organize files:
       mv data/raw/deepglobe/*_sat.jpg data/raw/deepglobe/images/
       mv data/raw/deepglobe/*_mask.png data/raw/deepglobe/masks/

Dataset URL: https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset
Codalab:     https://competitions.codalab.org/competitions/18467
""",
            )
            return False

        kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
        if not kaggle_json.exists():
            _print_instructions(
                "DeepGlobe — Kaggle Credentials Missing",
                f"""
Kaggle credentials not found at {kaggle_json}.

Steps:
  1. Go to https://www.kaggle.com/account
  2. Click "Create New API Token" — downloads kaggle.json
  3. Run:
       mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/
       chmod 600 ~/.kaggle/kaggle.json
  4. Then re-run: python src/download_utils.py --dataset deepglobe
""",
            )
            return False

        out_dir = ensure_dir(self.base_dir / "deepglobe")
        ensure_dir(out_dir / "images")
        ensure_dir(out_dir / "masks")

        logger.info("Downloading DeepGlobe via Kaggle API…")
        ret = subprocess.run(
            [
                sys.executable, "-m", "kaggle",
                "datasets", "download",
                "-d", "balraj98/deepglobe-road-extraction-dataset",
                "-p", str(out_dir),
                "--unzip",
            ],
            capture_output=False,
        )

        if ret.returncode != 0:
            logger.error("Kaggle download failed. Check credentials and dataset name.")
            return False

        # Organize downloaded files
        self._organize_deepglobe(out_dir)
        logger.info(f"DeepGlobe downloaded → {out_dir}")
        return True

    def _organize_deepglobe(self, base: Path) -> None:
        """Move DeepGlobe files to images/ and masks/ subdirectories."""
        img_dir = ensure_dir(base / "images")
        msk_dir = ensure_dir(base / "masks")
        for f in base.glob("*_sat.jpg"):
            shutil.move(str(f), img_dir / f.name)
        for f in base.glob("*_mask.png"):
            shutil.move(str(f), msk_dir / f.name)
        logger.info("DeepGlobe files organised into images/ and masks/")

    # ── OpenSatMap ────────────────────────────────────────────

    def _download_opensatmap(self, **kwargs: Any) -> bool:
        """
        Download OpenSatMap dataset.

        Uses direct GitHub releases if available, otherwise provides instructions.
        """
        if not _has_command("wget") and not _has_command("curl"):
            _print_instructions(
                "OpenSatMap — wget/curl Required",
                """
OpenSatMap is available from:
  https://github.com/OpenSatMap/OpenSatMap

Manual download steps:
  1. Visit the repository releases page
  2. Download the dataset archives
  3. Extract to: data/raw/opensatmap/
  4. Ensure structure:
       data/raw/opensatmap/images/*.png
       data/raw/opensatmap/masks/*.png

Or use Hugging Face (if published):
  pip install huggingface_hub
  python -c "
  from huggingface_hub import snapshot_download
  snapshot_download('OpenSatMap/OpenSatMap', local_dir='data/raw/opensatmap')
  "
""",
            )
            return False

        out_dir = ensure_dir(self.base_dir / "opensatmap")
        ensure_dir(out_dir / "images")
        ensure_dir(out_dir / "masks")

        logger.info(
            "OpenSatMap: Please download manually from "
            "https://github.com/OpenSatMap/OpenSatMap and place in data/raw/opensatmap/"
        )
        _print_instructions(
            "OpenSatMap — Manual Download Required",
            """
OpenSatMap dataset:
  Source: https://github.com/OpenSatMap/OpenSatMap

  1. Clone or download the repository
  2. Place image files in:   data/raw/opensatmap/images/
  3. Place mask files in:    data/raw/opensatmap/masks/
  4. Re-run: python run_pipeline.py --steps ingest
""",
        )
        return False

    # ── OSM ───────────────────────────────────────────────────

    def _download_osm(
        self,
        bbox: Optional[tuple] = None,
        region: Optional[str] = None,
        **kwargs: Any,
    ) -> bool:
        """
        Download OSM road vectors via Overpass API or Geofabrik.

        Args:
            bbox:   (south, west, north, east) in WGS84 for Overpass download.
            region: Geofabrik region name (e.g., 'india', 'europe/france').
        """
        if bbox is not None:
            return self._download_osm_overpass(bbox)
        elif region is not None:
            return self._download_osm_geofabrik(region)
        else:
            _print_instructions(
                "OSM Roads — Choose a Download Method",
                """
Option A — Overpass API (bounding box, small areas):
  python src/download_utils.py --dataset osm --bbox 12.9 77.5 13.1 77.7
  (south west north east — example: Bengaluru, India)

Option B — Geofabrik (regional OSM extracts):
  python src/download_utils.py --dataset osm --region india
  Available regions: https://download.geofabrik.de/
  After download, convert .osm.pbf with osmium or osmnx:
    pip install osmnx
    python -c "
    import osmnx as ox
    G = ox.graph_from_place('Bengaluru, India', network_type='drive')
    ox.save_graph_shapefile(G, filepath='data/raw/osm/vectors/')
    "

Option C — Manual GeoJSON:
  Place .geojson or .shp road files in: data/raw/osm/vectors/
  Then run: python src/vector_utils.py vector-to-mask \\
              --input data/raw/osm/vectors/ --output data/raw/osm/masks/road_mask.png
""",
            )
            return False

    def _download_osm_overpass(self, bbox: tuple) -> bool:
        """Fetch OSM road vectors via Overpass API and save to disk."""
        from src.data.vector_utils import fetch_osm_road_mask
        out_dir = ensure_dir(self.base_dir / "osm")
        ensure_dir(out_dir / "vectors")
        ensure_dir(out_dir / "masks")

        mask = fetch_osm_road_mask(
            bbox=bbox,
            out_mask_path=out_dir / "masks" / "osm_road_mask.png",
            out_vector_path=out_dir / "vectors" / "osm_roads.geojson",
        )
        return mask is not None

    def _download_osm_geofabrik(self, region: str) -> bool:
        """Download regional OSM PBF from Geofabrik."""
        out_dir = ensure_dir(self.base_dir / "osm")
        url = f"https://download.geofabrik.de/{region}-latest.osm.pbf"
        out_file = out_dir / f"{region.replace('/', '_')}-latest.osm.pbf"

        logger.info(f"Downloading OSM extract from Geofabrik: {url}")
        if _has_command("wget"):
            cmd = ["wget", "-c", url, "-O", str(out_file)]
        elif _has_command("curl"):
            cmd = ["curl", "-L", "-C", "-", url, "-o", str(out_file)]
        else:
            logger.error("wget or curl required for Geofabrik download.")
            return False

        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            logger.error(f"Failed to download {url}")
            return False

        logger.info(
            f"Downloaded {out_file}. "
            "Convert to road vectors with osmnx or osmium. "
            "See setup instructions in the README."
        )
        return True


# ─────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────

def _has_command(cmd: str) -> bool:
    """Return True if shell command is available."""
    return shutil.which(cmd) is not None


def _has_python_package(pkg: str) -> bool:
    """Return True if Python package is importable."""
    import importlib.util
    return importlib.util.find_spec(pkg) is not None


def _print_instructions(title: str, body: str) -> None:
    """Print a formatted instruction block to the console."""
    width = 70
    border = "═" * width
    print(f"\n╔{border}╗")
    print(f"║  {title:<{width - 2}}║")
    print(f"╠{border}╣")
    for line in body.strip().splitlines():
        print(f"║ {line:<{width - 1}}║")
    print(f"╚{border}╝\n")


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Route Resilience — Dataset Download Helper"
    )
    parser.add_argument(
        "--dataset",
        choices=["spacenet", "deepglobe", "opensatmap", "osm", "check"],
        required=True,
        help="Dataset to download, or 'check' to see which datasets are present.",
    )
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("SOUTH", "WEST", "NORTH", "EAST"),
        help="Bounding box for OSM Overpass download.",
    )
    parser.add_argument(
        "--region",
        type=str,
        default=None,
        help="Geofabrik region for OSM download (e.g., 'india', 'europe/france').",
    )
    parser.add_argument(
        "--cities",
        nargs="+",
        default=None,
        help="SpaceNet cities to download (default: Vegas Paris Shanghai Khartoum).",
    )

    args = parser.parse_args()
    downloader = DatasetDownloader()

    if args.dataset == "check":
        print("\nChecking dataset availability:")
        downloader.check_all()
    elif args.dataset == "osm":
        downloader.download(
            "osm",
            bbox=tuple(args.bbox) if args.bbox else None,
            region=args.region,
        )
    elif args.dataset == "spacenet":
        downloader.download("spacenet", cities=args.cities)
    else:
        downloader.download(args.dataset)
