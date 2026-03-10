"""
Timing utilities for profiling and structured runtime measurement.
"""

from __future__ import annotations

import time
from contextlib import ContextDecorator


class Timer(ContextDecorator):
    """Context manager and decorator for wall-clock timing."""

    def __init__(self, name: str | None = None) -> None:
        self.name = name
        self.start_time: float | None = None
        self.end_time: float | None = None
        self.elapsed: float | None = None

    def __enter__(self) -> "Timer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time if self.start_time is not None else None
        return None


def time_now() -> float:
    """Return a high-resolution current timestamp."""
    return time.perf_counter()


def elapsed_seconds(start_time: float) -> float:
    """Return elapsed wall-clock seconds since `start_time`."""
    return time.perf_counter() - start_time


def format_seconds(seconds: float) -> str:
    """Format a duration in seconds into a human-readable string."""
    if seconds < 1:
        return f"{seconds * 1000:.2f} ms"
    if seconds < 60:
        return f"{seconds:.2f} s"

    minutes, rem = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {rem:.2f}s"

    hours, rem_minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(rem_minutes)}m {rem:.2f}s"