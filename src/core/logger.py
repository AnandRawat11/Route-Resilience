"""
Route Resilience — src/core/logger.py

Structured logging factory.
All pipeline modules call get_logger(__name__) for consistent,
colour-coded, file-rotated logging.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

# Colour logging — graceful fallback
try:
    import colorlog  # type: ignore
    _HAS_COLORLOG = True
except ImportError:
    _HAS_COLORLOG = False

# Module-level logger cache — avoids duplicate handlers on repeated calls
_LOGGERS: Dict[str, logging.Logger] = {}


def get_logger(
    name: str,
    config: Optional[Dict[str, Any]] = None,
    log_dir: Optional[Union[str, Path]] = None,
) -> logging.Logger:
    """
    Create or retrieve a named logger.

    Args:
        name:     Logger name (use ``__name__`` in every module).
        config:   Full pipeline config dict (reads logging settings from it).
        log_dir:  Override log directory (takes priority over config).

    Returns:
        Configured ``logging.Logger``.
    """
    if name in _LOGGERS:
        return _LOGGERS[name]

    logger = logging.getLogger(name)

    # Defaults
    level_str    = "INFO"
    log_to_file  = True
    _log_dir     = Path("logs")
    max_bytes    = 10_485_760   # 10 MB
    backup_count = 5
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    if config is not None:
        lc           = config.get("logging", {})
        level_str    = lc.get("level", level_str)
        log_to_file  = lc.get("log_to_file", log_to_file)
        _log_dir     = Path(lc.get("log_dir", str(_log_dir)))
        max_bytes    = lc.get("max_bytes", max_bytes)
        backup_count = lc.get("backup_count", backup_count)
        fmt          = lc.get("format", fmt)

    if log_dir is not None:
        _log_dir = Path(log_dir)

    level = getattr(logging, level_str.upper(), logging.INFO)
    logger.setLevel(level)

    # Skip if already has handlers (e.g. root logger propagation)
    if logger.handlers:
        _LOGGERS[name] = logger
        return logger

    # ── Console handler ───────────────────────────────────────
    if _HAS_COLORLOG:
        console_fmt = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s | %(levelname)-8s%(reset)s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
            log_colors={
                "DEBUG":    "cyan",
                "INFO":     "green",
                "WARNING":  "yellow",
                "ERROR":    "red",
                "CRITICAL": "bold_red",
            },
        )
    else:
        console_fmt = logging.Formatter(fmt, datefmt="%H:%M:%S")

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(console_fmt)
    logger.addHandler(ch)

    # ── Rotating file handler ─────────────────────────────────
    if log_to_file:
        _log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file  = _log_dir / f"pipeline_{timestamp}.log"
        fh = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter(fmt))
        logger.addHandler(fh)

    logger.propagate = False
    _LOGGERS[name] = logger
    return logger
