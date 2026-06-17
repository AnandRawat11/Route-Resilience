"""
Route Resilience — src/core/__init__.py

Public API for the core package.
Re-exports everything so callers can do either:

    from src.core import load_config, get_logger, ensure_dir
    from src.core.io import load_image          # explicit
    from src.core.config import load_config     # explicit
"""

from src.core.config import load_config
from src.core.logger import get_logger
from src.core.io import (
    ensure_dir,
    load_image,
    load_mask,
    save_image,
    save_json,
    load_json,
    compute_md5,
    file_is_readable,
    extract_geospatial_metadata,
    format_bytes,
    timer,
)
from src.core.exceptions import (
    RouteResilienceError,
    ConfigError,
    DatasetNotFoundError,
    PipelineError,
    VectorConversionError,
    GeospatialError,
)
from src.core.constants import (
    PROJECT_NAME,
    PROJECT_VERSION,
    SEVERITY_LEVELS,
    SUPPORTED_IMAGE_EXTENSIONS,
    SUPPORTED_MASK_EXTENSIONS,
    SUPPORTED_VECTOR_EXTENSIONS,
    SPLIT_TRAIN,
    SPLIT_VAL,
    SPLIT_TEST,
    OCCLUSION_TYPES,
)

__all__ = [
    # config
    "load_config",
    # logger
    "get_logger",
    # io
    "ensure_dir", "load_image", "load_mask", "save_image",
    "save_json", "load_json", "compute_md5", "file_is_readable",
    "extract_geospatial_metadata", "format_bytes", "timer",
    # exceptions
    "RouteResilienceError", "ConfigError", "DatasetNotFoundError",
    "PipelineError", "VectorConversionError", "GeospatialError",
    # constants
    "PROJECT_NAME", "PROJECT_VERSION", "SEVERITY_LEVELS",
    "SUPPORTED_IMAGE_EXTENSIONS", "SUPPORTED_MASK_EXTENSIONS",
    "SUPPORTED_VECTOR_EXTENSIONS",
    "SPLIT_TRAIN", "SPLIT_VAL", "SPLIT_TEST", "OCCLUSION_TYPES",
]
