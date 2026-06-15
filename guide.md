ROUTE RESILIENCE - WINNING BLUEPRINT FOR BHARATIYA ANTARIKSH HACKATHON 2026

CORE STRATEGY

Most teams will build:

Satellite Image → Road Segmentation → Output Mask

We will build:

Satellite Image → Road Extraction → Road Graph → Criticality Analysis → Disaster Simulation → Decision Support Dashboard

The goal is not to build the best segmentation model.

The goal is to build the most useful urban resilience platform.

Judges will remember impact, innovation, and practical utility more than a few percentage points of segmentation accuracy.

⸻

PROJECT VISION

Build an AI-powered Urban Road Resilience Platform that:

1. Extracts roads from satellite imagery
2. Reconstructs roads hidden by trees and shadows
3. Converts roads into a connected graph network
4. Identifies critical bottlenecks
5. Simulates disasters and road failures
6. Measures city resilience
7. Recommends infrastructure improvements

⸻

FINAL SYSTEM ARCHITECTURE

Satellite Image

↓

Image Enhancement

↓

Road Segmentation

↓

Occlusion Recovery

↓

Skeletonization

↓

Road Graph Generation

↓

Graph Healing

↓

Criticality Analysis

↓

Disaster Simulation

↓

AI Recommendation Engine

↓

Interactive Dashboard

⸻

TECHNOLOGY STACK

AI / Computer Vision

* PyTorch
* SegFormer
* DeepLabV3+
* U-Net++

Geospatial Processing

* Rasterio
* GDAL
* GeoPandas
* OSMnx

Graph Theory

* NetworkX
* PyTorch Geometric

Dashboard

* Streamlit
* Folium
* Leaflet

Backend

* FastAPI

⸻

PHASE 1 - DATA COLLECTION

Datasets:

* SpaceNet Roads
* DeepGlobe Roads
* OpenSatMap
* OpenStreetMap

Use OpenStreetMap as ground truth for graph validation and routing comparison.

Directory Structure:

dataset/

images/

masks/

train/

val/

test/

⸻

PHASE 2 - DATA PREPROCESSING

Tasks:

* Tile large satellite images
* Normalize imagery
* Contrast enhancement
* Data augmentation

Augmentations:

* Rotation
* Brightness changes
* Blur
* Noise
* Artificial shadows
* Cloud cover
* Tree canopy simulation

IMPORTANT:

Create synthetic occlusions during training.

This teaches the model to infer roads hidden under trees and shadows.

Most teams will skip this.

⸻

PHASE 3 - ROAD SEGMENTATION

Baseline Model:

DeepLabV3+

Advanced Model:

SegFormer

Reason:

* Transformer based
* Better context understanding
* Strong performance on remote sensing tasks

Loss Function:

Final Loss = Dice Loss + BCE Loss + Boundary Loss

Advanced Feature:

Connectivity Loss

Instead of only penalizing wrong pixels, penalize broken roads.

This directly aligns with the problem statement.

⸻

PHASE 4 - OCCLUSION RECOVERY

Goal:

Reconnect roads hidden by:

* Tree canopies
* Building shadows
* Vehicles
* Cloud cover

Methods:

1. Morphological Closing
2. Endpoint Detection
3. Direction Matching
4. Gap Bridging

Advanced Idea:

Train a lightweight Graph Neural Network to predict whether two disconnected endpoints should be connected.

This will be a major innovation point.

⸻

PHASE 5 - SKELETONIZATION

Convert thick roads into centerlines.

Use:

skimage.morphology.skeletonize()

Output:

Single-pixel road centerlines.

This is required before graph creation.

⸻

PHASE 6 - ROAD GRAPH CREATION

Convert road centerlines into graph structure.

Nodes:

* Intersections
* Endpoints

Edges:

* Road segments

Library:

NetworkX

Graph Representation:

Intersection → Node

Road Segment → Edge

⸻

PHASE 7 - TOPOLOGICAL HEALING

Problem:

Occlusions create disconnected road networks.

Solution:

1. Detect disconnected components
2. Detect nearby endpoints
3. Calculate similarity score

Similarity Factors:

* Distance
* Angular alignment
* Road width similarity

Use:

* Minimum Spanning Tree (MST)
* Union Find (Disjoint Set)

Advanced Innovation:

Assign confidence scores to healed connections.

Example:

Connection A-B = 94% confidence

This improves explainability.

⸻

PHASE 8 - CRITICALITY ANALYSIS

Calculate:

1. Betweenness Centrality
2. Closeness Centrality
3. Eigenvector Centrality

Create a combined metric:

Criticality Score

Formula:

Criticality Score =
0.5 × Betweenness +
0.3 × Closeness +
0.2 × Eigenvector

Output:

Criticality Heatmap

Green = Safe

Yellow = Important

Red = Critical

⸻

PHASE 9 - DISASTER SIMULATION

Simulate:

1. Flood
2. Accident
3. Construction Closure
4. Bridge Collapse

Actions:

* Remove node
* Remove edge
* Recalculate network

Metrics:

* Reachability
* Travel Time
* Connectivity
* Network Efficiency

This is one of the strongest judging features.

⸻

PHASE 10 - RESILIENCE INDEX

Measure:

Before Failure

vs

After Failure

Metrics:

* Average Shortest Path
* Connected Components
* Reachability

Generate:

Resilience Score (0-100)

Example:

90-100 = Highly Resilient

70-89 = Moderate

Below 70 = Vulnerable

⸻

PHASE 11 - AI RECOMMENDATION ENGINE

Most teams stop at analytics.

We go one step further.

Example Output:

“Node #17 causes 32% mobility degradation when removed.”

Recommendation:

“Construct an alternate connector road between Junction A and Junction B.”

This transforms the project into a decision-support system.

⸻

PHASE 12 - DASHBOARD

Tab 1:

Satellite Image

Tab 2:

Extracted Roads

Tab 3:

Road Graph

Tab 4:

Criticality Heatmap

Tab 5:

Disaster Simulator

Tab 6:

AI Recommendations

⸻

OUT-OF-THE-BOX FEATURES

Feature 1:

Urban Resilience Score for every city zone

Example:

Zone A = 92

Zone B = 61

Zone C = 35

⸻

Feature 2:

Time-Aware Resilience

Different road importance during:

* Morning
* Afternoon
* Night

⸻

Feature 3:

Multi-Source Validation

Satellite Data + OSM

⸻

Feature 4:

Confidence-Based Healing

Every reconstructed road receives a confidence score.

⸻

Feature 5:

Interactive Failure Simulation

User clicks a junction.

System instantly:

* Removes junction
* Recalculates routes
* Shows mobility impact

⸻

HACKATHON EXECUTION PLAN

Hours 1-4

* Environment Setup
* Dataset Preparation

Hours 5-10

* Train Segmentation Model
* Data Augmentation

Hours 11-16

* Skeletonization
* Graph Generation

Hours 17-22

* Graph Healing
* Centrality Analysis

Hours 23-26

* Disaster Simulation
* Resilience Score

Hours 27-30

* Dashboard
* Presentation
* Demo Optimization

⸻

FINAL PITCH

“Our platform does not merely detect roads from satellite imagery. It predicts how urban mobility behaves under infrastructure failure, identifies critical bottlenecks, quantifies resilience, and recommends interventions for disaster preparedness and smart-city planning.”

This is the sentence that should appear on the first slide of the final presentation.