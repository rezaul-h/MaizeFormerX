"""
I/O utilities for filesystem-safe reading, writing, and directory management.
"""

from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path
from typing import Any

import yaml


def to_path(path: str | Path) -> Path:
    """Convert a string or Path into a Path object."""
    return path if isinstance(path, Path) else Path(path)


def ensure_dir(path: str | Path) -> Path:
    """
    Ensure that a directory exists.

    If a file path is passed, its parent directory is created.
    """
    path = to_path(path)
    target_dir = path if path.suffix == "" else path.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def ensure_dirs(paths: list[str | Path]) -> list[Path]:
    """Ensure multiple directories exist."""
    return [ensure_dir(p) for p in paths]


def path_exists(path: str | Path) -> bool:
    """Return True if the path exists."""
    return to_path(path).exists()


def is_file(path: str | Path) -> bool:
    """Return True if the path is a file."""
    return to_path(path).is_file()


def is_dir(path: str | Path) -> bool:
    """Return True if the path is a directory."""
    return to_path(path).is_dir()


def list_files(
    directory: str | Path,
    suffixes: tuple[str, ...] | None = None,
    recursive: bool = True,
    sort: bool = True,
) -> list[Path]:
    """List files in a directory, optionally filtering by suffix."""
    directory = to_path(directory)
    if not directory.exists():
        return []

    files = directory.rglob("*") if recursive else directory.glob("*")
    results = [p for p in files if p.is_file()]

    if suffixes is not None:
        suffixes_lower = tuple(s.lower() for s in suffixes)
        results = [p for p in results if p.suffix.lower() in suffixes_lower]

    if sort:
        results = sorted(results)

    return results


def read_text(path: str | Path, encoding: str = "utf-8") -> str:
    """Read a text file."""
    return to_path(path).read_text(encoding=encoding)


def write_text(path: str | Path, content: str, encoding: str = "utf-8") -> None:
    """Write a text file."""
    path = to_path(path)
    ensure_dir(path)
    path.write_text(content, encoding=encoding)


def read_json(path: str | Path, encoding: str = "utf-8") -> Any:
    """Read a JSON file."""
    with to_path(path).open("r", encoding=encoding) as f:
        return json.load(f)


def write_json(
    path: str | Path,
    data: Any,
    indent: int = 2,
    ensure_ascii: bool = False,
    encoding: str = "utf-8",
) -> None:
    """Write a JSON file."""
    path = to_path(path)
    ensure_dir(path)
    with path.open("w", encoding=encoding) as f:
        json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)


def read_yaml(path: str | Path, encoding: str = "utf-8") -> dict[str, Any]:
    """Read a YAML file."""
    with to_path(path).open("r", encoding=encoding) as f:
        data = yaml.safe_load(f)
    return data or {}


def write_yaml(
    path: str | Path,
    data: dict[str, Any],
    encoding: str = "utf-8",
    sort_keys: bool = False,
) -> None:
    """Write a YAML file."""
    path = to_path(path)
    ensure_dir(path)
    with path.open("w", encoding=encoding) as f:
        yaml.safe_dump(
            data,
            f,
            sort_keys=sort_keys,
            allow_unicode=True,
            default_flow_style=False,
        )


def read_csv(path: str | Path, encoding: str = "utf-8") -> list[dict[str, str]]:
    """Read a CSV file into a list of dictionaries."""
    with to_path(path).open("r", encoding=encoding, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def write_csv(
    path: str | Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str] | None = None,
    encoding: str = "utf-8",
) -> None:
    """Write a list of dictionaries to CSV."""
    path = to_path(path)
    ensure_dir(path)

    if not rows and fieldnames is None:
        raise ValueError("fieldnames must be provided when writing an empty CSV.")

    if fieldnames is None:
        fieldnames = list(rows[0].keys())

    with path.open("w", encoding=encoding, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_pickle(path: str | Path, obj: Any) -> None:
    """Serialize and save a Python object."""
    path = to_path(path)
    ensure_dir(path)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str | Path) -> Any:
    """Load a serialized Python object."""
    with to_path(path).open("rb") as f:
        return pickle.load(f)