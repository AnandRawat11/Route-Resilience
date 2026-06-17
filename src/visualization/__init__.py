"""
Route Resilience — src/visualization/__init__.py

Public API for the visualization package.
Phase 1: pipeline summary plots, tile grids, occlusion galleries.
Phase 5: will add maps.py (Folium/Leaflet) and dashboards.py (Streamlit).
"""

from src.visualization.plots import PipelineVisualizer, run_visualization

__all__ = ["PipelineVisualizer", "run_visualization"]
