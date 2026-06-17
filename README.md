# Route Resilience — Phase 1: Data Pipeline

> **Project**: Route Resilience: Occlusion-Robust Road Extraction & Graph-Theoretic Criticality Analysis for Urban Mobility  
> **Phase**: 1 — Data Collection & Preprocessing Pipeline  
> **Target**: Bharatiya Antariksh Hackathon 2026

---

## Overview

Phase 1 builds a production-ready data pipeline that converts raw satellite imagery datasets into a high-quality, augmented, train-ready dataset for road segmentation under occlusion.

```
Satellite Imagery (SpaceNet / DeepGlobe / OpenSatMap / OSM)
         ↓
    Validation & Ingestion
         ↓
    Standardization + Geo Metadata
         ↓
    512×512 Tiling
         ↓
    70/15/15 Stratified Split
         ↓
    Albumentations Augmentation
         ↓
    Occlusion Simulation (4 types × 3 severities)
         ↓
    Quality Analysis + master_dataset.csv
         ↓
    Ready for Phase 2 Training
```

---

## Quick Start

### 1. Environment Setup

```bash
cd "Road resilience"

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Setup

Check which datasets are already available:
```bash
python src/download_utils.py --dataset check
```

#### SpaceNet Roads (AWS CLI required)
```bash
# Install AWS CLI, then:
python src/download_utils.py --dataset spacenet
# Or manually: aws s3 sync s3://spacenet-dataset/spacenet/SN3_roads/ data/raw/spacenet/ --no-sign-request
```

#### DeepGlobe Road Extraction (Kaggle API required)
```bash
# Setup Kaggle credentials (~/.kaggle/kaggle.json), then:
python src/download_utils.py --dataset deepglobe
```

#### OpenSatMap
```bash
python src/download_utils.py --dataset opensatmap
# Follow on-screen instructions for manual download
```

#### OpenStreetMap Roads (via Overpass API — example: Bengaluru)
```bash
python src/download_utils.py --dataset osm --bbox 12.9 77.5 13.1 77.7
```

#### Custom dataset placement
Any dataset can be added manually:
```
data/raw/<dataset_name>/
    images/   ← satellite images (.tif, .png, .jpg)
    masks/    ← binary road masks (same stem, .tif or .png)
```
Then enable it in `configs/config.yaml` under `datasets:`.

### 3. Run the Full Pipeline

```bash
python run_pipeline.py
```

Run individual steps:
```bash
python run_pipeline.py --steps ingest
python run_pipeline.py --steps ingest standardize tile
python run_pipeline.py --steps quality visualize
```

### 4. Launch the Notebook

```bash
cd notebooks
jupyter notebook 01_pipeline_demo.ipynb
```

---

## Directory Structure

```
Road resilience/
├── data/
│   ├── raw/
│   │   ├── spacenet/          ← SpaceNet Roads (images/ + masks/)
│   │   ├── deepglobe/         ← DeepGlobe (images/ + masks/)
│   │   ├── opensatmap/        ← OpenSatMap (images/ + masks/)
│   │   └── osm/               ← OSM vectors + rasterized masks
│   ├── processed/
│   │   ├── images/            ← Standardized RGB uint8 PNGs
│   │   ├── masks/             ← Binary road masks
│   │   └── metadata/          ← Per-image geospatial JSON
│   ├── tiles/
│   │   ├── images/            ← 512×512 image tiles
│   │   └── masks/             ← 512×512 mask tiles
│   ├── train/images/ & masks/ ← Training split (70%)
│   ├── val/images/ & masks/   ← Validation split (15%)
│   └── test/images/ & masks/  ← Test split (15%)
├── src/
│   ├── utils.py               ← Shared I/O, logging, geo metadata
│   ├── ingestion.py           ← Step 1: Dataset scanning & validation
│   ├── standardization.py     ← Step 2: RGB conversion + geo metadata
│   ├── tiling.py              ← Step 3: 512×512 sliding window
│   ├── splitting.py           ← Step 4: Stratified 70/15/15 split
│   ├── augmentation.py        ← Step 5: Albumentations pipeline
│   ├── occlusion.py           ← Step 6: Synthetic occlusion generation
│   ├── quality.py             ← Step 7: Quality metrics + master CSV
│   ├── visualization.py       ← Step 8: Pipeline visualizations
│   ├── vector_utils.py        ← GeoJSON/Shapefile ↔ mask conversion
│   └── download_utils.py      ← Dataset download helpers
├── configs/
│   └── config.yaml            ← All pipeline configuration
├── notebooks/
│   └── 01_pipeline_demo.ipynb ← Fully runnable end-to-end demo
├── outputs/
│   ├── reports/               ← JSON reports and CSVs
│   └── visualizations/        ← PNG charts and comparisons
│   └── occlusion_samples/     ← Occlusion examples per type
├── logs/                      ← Timestamped pipeline logs
├── run_pipeline.py            ← Main CLI entry point
└── requirements.txt
```

---

## Output Files

| File | Description |
|------|-------------|
| `outputs/reports/dataset_report.json` | Images, masks, dimensions, road pixel %, geo stats |
| `outputs/reports/quality_report.json` | Road width, density, class imbalance, occlusion stats |
| `outputs/reports/master_dataset.csv` | **Central source of truth** — all tiles with full metadata |
| `outputs/reports/train.csv` | Training split (70%) |
| `outputs/reports/val.csv` | Validation split (15%) |
| `outputs/reports/test.csv` | Test split (15%) |
| `outputs/reports/tile_statistics.json` | Keep/discard counts, road density histogram |
| `outputs/reports/split_metadata.json` | Seed, ratios, source distribution |
| `outputs/reports/augmentation_pipeline.json` | Serialized Albumentations pipeline |
| `outputs/reports/occlusion_statistics.json` | Coverage stats per type and severity |
| `outputs/visualizations/` | All PNG comparisons and charts |
| `outputs/occlusion_samples/` | `*_original.png`, `*_occluded.png`, `*_comparison.png` |

---

## Geospatial Metadata

For each image, `data/processed/metadata/<image_id>_geo.json` stores:

```json
{
  "crs": "EPSG:32614",
  "bounds": { "left": 458000.0, "bottom": 3756000.0, "right": 459024.0, "top": 3757024.0 },
  "transform": { "a": 1.0, "b": 0.0, "c": 458000.0, "d": 0.0, "e": -1.0, "f": 3757024.0 },
  "resolution_x": 1.0,
  "resolution_y": 1.0,
  "processed_image_path": "data/processed/images/image_id.png"
}
```

This enables Phase 2 graph reconstruction to operate in real-world coordinates.

---

## Occlusion Simulation

Four occlusion types, each at three severity levels:

| Type | Mechanism | Severity |
|------|-----------|---------|
| Tree Canopy | Perlin-noise organic blobs (green) | light / medium / heavy |
| Building Shadow | Directional gradient polygons | light / medium / heavy |
| Vehicle | Road-aligned rectangles (metallic) | light / medium / heavy |
| Cloud Cover | Smooth Gaussian blobs (white/grey) | light / medium / heavy |

---

## Vector Utilities (OSM Integration)

```bash
# Convert GeoJSON → binary mask
python src/vector_utils.py vector-to-mask \
    --input data/raw/osm/vectors/roads.geojson \
    --output data/raw/osm/masks/road_mask.png

# Convert binary mask → GeoJSON (for graph reconstruction)
python src/vector_utils.py mask-to-vector \
    --mask data/tiles/masks/tile_001.png \
    --output outputs/vectors/tile_001_roads.geojson \
    --geo-meta data/processed/metadata/tile_001_geo.json

# Fetch OSM road mask for Bengaluru
python src/vector_utils.py fetch-osm \
    --bbox 12.9 77.5 13.1 77.7 \
    --output data/raw/osm/masks/bengaluru.png \
    --vector-output data/raw/osm/vectors/bengaluru.geojson
```

---

## Configuration

All parameters are in `configs/config.yaml`. Key settings:

```yaml
tiling:
  tile_size: 512      # Output tile size (pixels)
  stride: 256         # Overlap stride (256 = 50% overlap)
  min_road_pixel_pct: 0.5  # Discard near-empty tiles

splitting:
  train_ratio: 0.70
  val_ratio: 0.15
  test_ratio: 0.15
  stratify_by_source: true  # Preserve dataset distribution

occlusion:
  types:
    tree_canopy:
      enabled: true
      heavy:
        coverage_min: 0.50
        coverage_max: 0.75
```

---

## Next Steps (Phase 2)

Phase 1 outputs are directly consumed by Phase 2:

- `master_dataset.csv` → DataLoader index
- `data/train|val|test/` → PyTorch Dataset directories
- `augmentation_pipeline.json` → Restore Albumentations pipeline
- `data/processed/metadata/` → Geo-coordinate mapping for graph reconstruction
- `outputs/vectors/` → Road centerline GeoJSON for graph initialization
