"""
Route Resilience — src/core/exceptions.py

Custom exception hierarchy for the entire pipeline.
Gives precise, actionable error messages rather than generic Python exceptions.
"""


class RouteResilienceError(Exception):
    """Base class for all Route Resilience errors."""


class ConfigError(RouteResilienceError):
    """Raised when the YAML configuration is missing, malformed, or invalid."""


class DatasetNotFoundError(RouteResilienceError):
    """Raised when no images are found after scanning all configured dataset directories."""


class PipelineError(RouteResilienceError):
    """Raised when a pipeline step fails in an unrecoverable way."""


class VectorConversionError(RouteResilienceError):
    """Raised when vector ↔ raster conversion fails."""


class GeospatialError(RouteResilienceError):
    """Raised for CRS, transform, or projection errors."""
