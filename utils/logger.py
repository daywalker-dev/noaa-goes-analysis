"""Structured logging with rich formatting."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

try:
    from rich.logging import RichHandler
    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False


def get_logger(
    name: str = "goes_forecast",
    level: int = logging.INFO,
    log_file: Optional[str | Path] = None,
) -> logging.Logger:
    """Create a structured logger with optional file output."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    logger.propagate = False

    fmt = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"

    if _HAS_RICH:
        console = RichHandler(rich_tracebacks=True, show_path=False)
        console.setFormatter(logging.Formatter("%(message)s", datefmt=date_fmt))
    else:
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(logging.Formatter(fmt, datefmt=date_fmt))
    logger.addHandler(console)

    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter(fmt, datefmt=date_fmt))
        logger.addHandler(fh)

    return logger
