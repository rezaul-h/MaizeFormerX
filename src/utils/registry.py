"""
Simple name-to-object registry utilities.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class Registry:
    """A lightweight registry for mapping string names to callables or objects."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._items: dict[str, Any] = {}

    def register(self, key: str | None = None) -> Callable[[Any], Any]:
        """Decorator-based registration."""

        def decorator(obj: Any) -> Any:
            name = key or obj.__name__.lower()
            if name in self._items:
                raise KeyError(f"{name!r} is already registered in registry {self.name!r}.")
            self._items[name] = obj
            return obj

        return decorator

    def add(self, key: str, value: Any) -> None:
        """Register an object explicitly."""
        if key in self._items:
            raise KeyError(f"{key!r} is already registered in registry {self.name!r}.")
        self._items[key] = value

    def get(self, key: str) -> Any:
        """Retrieve a registered object."""
        if key not in self._items:
            available = ", ".join(sorted(self._items.keys()))
            raise KeyError(
                f"{key!r} is not registered in registry {self.name!r}. "
                f"Available: [{available}]"
            )
        return self._items[key]

    def exists(self, key: str) -> bool:
        """Return True if a key exists."""
        return key in self._items

    def keys(self) -> list[str]:
        """Return registered keys."""
        return sorted(self._items.keys())

    def items(self) -> list[tuple[str, Any]]:
        """Return registered items."""
        return sorted(self._items.items(), key=lambda x: x[0])

    def __contains__(self, key: str) -> bool:
        return key in self._items

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return f"Registry(name={self.name!r}, size={len(self._items)})"