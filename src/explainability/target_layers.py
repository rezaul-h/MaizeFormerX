"""
Target-layer resolution utilities for explainability.
"""

from __future__ import annotations

import re
from typing import Any


_INDEX_PATTERN = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\[(\-?\d+)\]")


def _resolve_token(obj: Any, token: str) -> Any:
    """
    Resolve one path token, supporting forms like:
    - encoder
    - blocks[0]
    - blocks[-1]
    """
    match = _INDEX_PATTERN.fullmatch(token)
    if match:
        attr_name, index_str = match.groups()
        value = getattr(obj, attr_name)
        return value[int(index_str)]
    return getattr(obj, token)


def resolve_layer_path(model: Any, layer_path: str) -> Any:
    """
    Resolve a dotted layer path against a model.

    Example
    -------
    resolve_layer_path(model, "encoder.blocks[-1].norm1")
    """
    current = model
    for token in layer_path.split("."):
        current = _resolve_token(current, token)
    return current


def get_default_target_layer(model_name: str) -> str:
    """
    Return a defensible default target layer for a model family.
    """
    name = model_name.lower()
    if name == "maizeformerx":
        return "encoder.blocks[-1].norm2"
    if name in {"mobilevit", "efficientformer", "tinyvit", "swinv2"}:
        return "model"
    if name in {"shufflenetv2", "ghostnet", "mobilenetv3", "efficientnet_b0", "efficientnet_b1", "mixmobilenet"}:
        return "model" if hasattr(model_name, "model") else ""
    return ""


def resolve_target_layer(model: Any, model_name: str, layer_path: str | None = None) -> Any:
    """
    Resolve either an explicit target layer path or a default one.
    """
    if layer_path is None or layer_path == "":
        layer_path = get_default_target_layer(model_name)
        if not layer_path:
            raise ValueError(f"No default target layer available for model_name={model_name!r}")

    return resolve_layer_path(model, layer_path)