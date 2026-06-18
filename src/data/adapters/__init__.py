"""
Route Resilience — src/data/adapters/__init__.py

Dataset adapters convert vendor-specific directory layouts into
the canonical ImageRecord format used by the rest of the pipeline.

Available adapters
------------------
deepglobe  :  DeepGlobe Road Extraction Dataset (Kaggle competition format)
"""

from src.data.adapters.deepglobe_adapter import DeepGlobeAdapter

__all__ = ["DeepGlobeAdapter"]
