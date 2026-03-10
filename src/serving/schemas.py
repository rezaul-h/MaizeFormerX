"""
Pydantic schemas for the inference service.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool
    device: str


class PredictionItem(BaseModel):
    class_index: int
    class_name: str
    probability: float


class PredictResponse(BaseModel):
    predictions: list[PredictionItem]
    top1_class_index: int
    top1_class_name: str
    top1_probability: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class LoadModelRequest(BaseModel):
    model_name: str
    dataset_name: str
    checkpoint_path: str
    model_config_path: str | None = None
    device: str = "auto"