"""
Route Resilience — src/data/adapters/base_adapter.py

Abstract base class for all dataset adapters.

An adapter converts a vendor-specific dataset layout into a list of
canonical ImageRecord objects understood by the rest of the pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseAdapter(ABC):
    """
    Abstract interface every dataset adapter must implement.

    Concrete adapters:
        DeepGlobeAdapter  — Kaggle/CodaLab competition format
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    @abstractmethod
    def scan(self, active_splits: Optional[List[str]] = None) -> List[Any]:
        """
        Discover all image/mask pairs in the dataset.

        Args:
            active_splits: Which splits to process, e.g. ['train'].
                           If None, process all available splits.

        Returns:
            List of ImageRecord objects (compatible with ingestion pipeline).
        """

    @abstractmethod
    def generate_statistics(self, records: List[Any]) -> Dict[str, Any]:
        """
        Compute per-split and overall statistics for a set of records.

        Returns:
            Dict suitable for inclusion in dataset_report.json.
        """
