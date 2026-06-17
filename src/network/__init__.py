"""
Route Resilience — src/network

Merged topology + graph domain (Phase 3).
Covers: road skeleton reconstruction, graph building,
centrality analysis, healing, and resilience scoring.

Modules implemented in Phase 3:
  reconstruction.py — OCR → binary mask → road skeleton
  graph_builder.py  — skeleton → NetworkX graph
  centrality.py     — betweenness / closeness / degree centrality
  resilience.py     — connectivity loss under node/edge removal
"""
