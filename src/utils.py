"""
Backward-compatibility shim.
Import from src.core.* directly in new code:
  from src.core.io     import load_image, save_json, ensure_dir
  from src.core.logger import get_logger
  from src.core.config import load_config
"""
from src.core.config import load_config           # noqa: F401
from src.core.logger import get_logger            # noqa: F401
from src.core.io import (                         # noqa: F401
    ensure_dir, load_image, load_mask, save_image,
    save_json, load_json, compute_md5, file_is_readable,
    extract_geospatial_metadata, format_bytes, timer,
)
