"""
Route Resilience — src/core/constants.py

Shared constants used across all pipeline phases.
"""

PROJECT_NAME = "Route Resilience"
PROJECT_VERSION = "1.0.0"

SEVERITY_LEVELS = ["light", "medium", "heavy"]

SUPPORTED_IMAGE_EXTENSIONS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
SUPPORTED_MASK_EXTENSIONS  = (".tif", ".tiff", ".png")
SUPPORTED_VECTOR_EXTENSIONS = (".geojson", ".shp")

# Data split names
SPLIT_TRAIN = "train"
SPLIT_VAL   = "val"
SPLIT_TEST  = "test"

# Occlusion types
OCCLUSION_TYPES = ["tree_canopy", "building_shadow", "vehicle", "cloud_cover"]

# Standardised output dtype
OUTPUT_DTYPE = "uint8"
OUTPUT_CHANNELS = 3
