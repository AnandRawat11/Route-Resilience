# Route Resilience — Architecture

## Domain Map

```
src/
├── core/           Phase ALL  — config, logging, I/O, exceptions, constants
├── data/           Phase 1    — ingestion → tiling → splitting → augmentation → quality
├── segmentation/   Phase 2    — U-Net / DeepLabV3+ / SegFormer training & inference
├── network/        Phase 3/4  — topology reconstruction, graph analytics, resilience
└── visualization/  Phase ALL  — plots, maps, dashboards
```

## Dependency Rules

- `core/`         has **no** internal src/ dependencies (pure stdlib + yaml + numpy)
- `data/`         imports from `core/` only
- `segmentation/` imports from `core/`, `data/`
- `network/`      imports from `core/`, `data/`, `segmentation/` (predictions)
- `visualization/`imports from `core/`; receives data as arguments

## Data Flow (Phase 1)

```
data/raw/
  └─ DatasetScanner (ingestion.py)
       └─ ImageRecord list
            ├─ Standardizer    → data/processed/
            ├─ TileExtractor   → data/tiles/
            ├─ DataSplitter    → data/train|val|test/
            ├─ AugmentBuilder  → outputs/visualizations/
            ├─ OcclusionSim    → outputs/occlusion_samples/
            ├─ QualityAnalyser → outputs/reports/
            └─ PipelineViz     → outputs/visualizations/
```

## Phase Roadmap

| Phase | Domain | Key Deliverable |
|-------|--------|----------------|
| 1 ✅ | data/ | Dataset pipeline, occlusion simulation |
| 2 | segmentation/ | Trained road extraction model |
| 3 | network/ | Routable road network graph |
| 4 | network/ | Criticality scores, failure simulations |
| 5 | apps/ | Streamlit dashboard + FastAPI backend |
