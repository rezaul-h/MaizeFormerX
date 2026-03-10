"""
Logging utilities for project-wide consistent logging.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from src.utils.io import ensure_dir


_LOGGER_CACHE: dict[str, logging.Logger] = {}


def get_logger(
    name: str,
    level: int | str = logging.INFO,
    log_file: str | Path | None = None,
    propagate: bool = False,
) -> logging.Logger:
    """
    Create or retrieve a configured logger.

    Parameters
    ----------
    name:
        Logger name.
    level:
        Logging level.
    log_file:
        Optional file path for file logging.
    propagate:
        Whether logs should propagate to parent loggers.
    """
    if name in _LOGGER_CACHE:
        return _LOGGER_CACHE[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = propagate

    if logger.handlers:
        _LOGGER_CACHE[name] = logger
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        log_path = Path(log_file)
        ensure_dir(log_path)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _LOGGER_CACHE[name] = logger
    return logger


def set_global_logging_level(level: int | str) -> None:
    """Update the level of all cached loggers."""
    for logger in _LOGGER_CACHE.values():
        logger.setLevel(level)
        for handler in logger.handlers:
            handler.setLevel(level)