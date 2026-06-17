#!/usr/bin/env python3
"""
Route Resilience — scripts/preprocess.py

Canonical Phase 1 CLI entry point.
Identical behaviour to run_pipeline.py (which is now a backward-compat shim).

Usage:
    python scripts/preprocess.py                    # all steps
    python scripts/preprocess.py --steps ingest
    python scripts/preprocess.py --steps ingest standardize tile split
    python scripts/preprocess.py --config configs/config.yaml --steps all
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Delegate entirely to the updated run_pipeline entry point
from run_pipeline import main

if __name__ == "__main__":
    main()
