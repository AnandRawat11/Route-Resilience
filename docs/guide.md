ROUTE RESILIENCE - TECHNICAL MASTER GUIDE

Project Objective

Build an Urban Mobility Resilience Platform that:

1. Extracts roads from satellite imagery under heavy occlusion.
2. Reconstructs missing road connectivity.
3. Converts roads into a connected graph network.
4. Identifies critical bottlenecks.
5. Simulates infrastructure failures.
6. Quantifies network resilience.
7. Provides actionable planning recommendations.

⸻

System Architecture

Satellite Image

↓

Occlusion-Aware Road Extraction

↓

Topology Reconstruction

↓

Road Skeletonization

↓

Road Graph Generation

↓

Graph Healing

↓

Criticality Analysis

↓

Resilience Analysis

↓

Stress Testing

↓

Urban Resilience Decision Engine

↓

Interactive Dashboard

⸻

Technology Stack

Computer Vision

* PyTorch
* SegFormer
* DeepLabV3+
* Albumentations

Geospatial Processing

* Rasterio
* GeoPandas
* GDAL
* Shapely
* OSMnx

Graph Processing

* NetworkX
* PyTorch Geometric

Dashboard

* Streamlit
* Folium
* Leaflet.js

Backend

* FastAPI

⸻

Phase 1 - Data Collection & Preprocessing

Datasets

Primary:

* SpaceNet 3 Road Network Detection

Secondary:

* DeepGlobe Road Extraction

Supporting:

* OpenStreetMap Road Vectors

Final Evaluation:

* Cartosat-3

Requirements

* Data ingestion
* Dataset validation
* Metadata extraction
* Image standardization
* Tile generation
* Train/Validation/Test split
* Augmentation pipeline
* Occlusion simulation
* Dataset quality analysis

Metadata Requirements

Preserve:

* CRS
* Bounds
* Affine Transform
* Resolution

Support:

* GeoJSON
* Shapefile

Generate:

* master_dataset.csv
* dataset_report.json
* quality_report.json

⸻

Phase 2 - Occlusion-Aware Road Extraction

Models

Baseline:

* DeepLabV3+

Primary:

* SegFormer

Ablation:

* U-Net++

Input

Satellite Image

Output

Road Probability Mask

Augmentations

Standard:

* Horizontal Flip
* Vertical Flip
* Rotation
* Brightness
* Contrast
* Blur
* Gaussian Noise

Occlusion-Based:

* Tree Canopy
* Building Shadow
* Vehicle Occlusion
* Cloud Cover

Occlusion Severity:

* Light
* Medium
* Heavy

⸻

Loss Function

Final Loss =

Dice Loss
+
Binary Cross Entropy
+
Boundary Loss
+
Connectivity Loss

Objective:

Preserve road continuity and topological connectivity.

⸻

Phase 3 - Topology Reconstruction

Objective:

Recover road continuity under severe occlusion.

Methods:

* Morphological Closing
* Endpoint Detection
* Direction Matching
* Gap Bridging
* Connectivity Recovery

Optional Innovation:

Graph Neural Network based gap prediction.

Input:

Disconnected endpoints.

Output:

Connection probability.

⸻

Phase 4 - Road Skeletonization

Convert road masks into centerlines.

Tool:

skimage.morphology.skeletonize()

Output:

Single-pixel road centerlines.

⸻

Phase 5 - Road Graph Generation

Convert centerlines into graph structure.

Nodes

* Intersections
* Endpoints

Edges

* Road Segments

Library:

* NetworkX

⸻

Phase 6 - Topological Healing

Objective:

Reconnect disconnected graph components.

Similarity Metrics

* Euclidean Distance
* Angular Alignment
* Road Width Similarity

Algorithms

* Minimum Spanning Tree (MST)
* Union Find (Disjoint Set)

Output

Connected road graph with confidence scores for healed connections.

⸻

Phase 7 - Criticality Analysis

Metrics

* Betweenness Centrality
* Edge Betweenness Centrality
* Closeness Centrality
* Eigenvector Centrality

Criticality Score

Criticality Score =
0.5 × Betweenness +
0.2 × Edge Betweenness +
0.2 × Closeness +
0.1 × Eigenvector

Output

Criticality Heatmap.

⸻

Phase 8 - Resilience Analysis

Metrics

* Network Efficiency
* Reachability
* Connected Components
* Average Shortest Path Length

Output

Resilience Index (0-100)

Classification:

* 90-100 → Highly Resilient
* 70-89 → Moderately Resilient
* Below 70 → Vulnerable

⸻

Phase 9 - Disaster Stress Testing

Scenarios

* Single Junction Failure
* Road Segment Failure
* Localized Flood Zone
* Bridge Collapse
* Multi-Node Cascade Failure

Actions

* Remove Nodes
* Remove Edges
* Recalculate Network

Outputs

* Connectivity Loss
* Travel Time Increase
* Reachability Reduction
* Resilience Degradation

⸻

Phase 10 - Facility Impact Analysis

Facilities:

* Hospitals
* Schools
* Fire Stations
* Food Distribution Centers

Metrics:

* Facility Reachability
* Accessibility Loss
* Service Disruption

Outputs:

* Affected Facilities
* Isolated Facilities
* Accessibility Impact

⸻

Phase 11 - Urban Resilience Decision Engine

Generate recommendations based on graph vulnerability.

Examples:

* Alternate Route Suggestions
* Emergency Access Routes
* Redundancy Planning
* Infrastructure Upgrade Suggestions

Input:

Criticality Analysis + Stress Testing Results

Output:

Actionable planning recommendations.

⸻

Phase 12 - Interactive Dashboard

Modules:

1. Satellite Imagery
2. Extracted Roads
3. Road Graph
4. Criticality Heatmap
5. Resilience Heatmap
6. Disaster Simulator
7. Facility Impact Analysis
8. Decision Engine Recommendations

⸻

Evaluation Metrics

Segmentation

* IoU
* Dice Score
* Precision
* Recall

Connectivity

* Largest Connected Component Ratio
* Connectivity Gain After Healing
* Connected Components Count

Graph Quality

* Path Completeness
* Topological Accuracy
* Average Path Length Error

Resilience

* Reachability
* Network Efficiency
* Resilience Index

⸻

Development Priority

Priority 1:

* SegFormer
* Skeletonization
* Graph Generation

Priority 2:

* Graph Healing
* Criticality Analysis
* Resilience Analysis

Priority 3:

* Disaster Simulation
* Facility Impact Analysis
* Decision Engine

Priority 4:

* Dashboard
* Visualization
* Presentation Layer