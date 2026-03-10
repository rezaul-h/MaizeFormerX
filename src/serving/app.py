"""
FastAPI serving app for MaizeFormerX inference.
"""

from __future__ import annotations

from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from PIL import Image

from src.serving.inference_api import InferenceService
from src.serving.schemas import (
    HealthResponse,
    LoadModelRequest,
    PredictResponse,
)

app = FastAPI(title="MaizeFormerX Inference API", version="0.1.0")
service = InferenceService()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    device = str(service.bundle.device) if service.bundle is not None else "uninitialized"
    return HealthResponse(
        model_loaded=service.is_loaded,
        device=device,
    )


@app.post("/load_model")
def load_model(request: LoadModelRequest):
    try:
        service.load_model(
            model_name=request.model_name,
            dataset_name=request.dataset_name,
            checkpoint_path=request.checkpoint_path,
            model_config_path=request.model_config_path,
            device_name=request.device,
        )
        return {
            "status": "ok",
            "model_name": request.model_name,
            "dataset_name": request.dataset_name,
            "checkpoint_path": request.checkpoint_path,
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)) -> PredictResponse:
    if not service.is_loaded:
        raise HTTPException(status_code=400, detail="No model is loaded.")

    try:
        image_bytes = await file.read()
        result = service.predict_from_bytes(image_bytes, topk=3)
        return PredictResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/explain/json")
async def explain_json(file: UploadFile = File(...)):
    if not service.is_loaded:
        raise HTTPException(status_code=400, detail="No model is loaded.")

    try:
        image_bytes = await file.read()
        result = service.explain_from_bytes(image_bytes)
        # Avoid returning full arrays directly.
        return JSONResponse(
            content={
                "top1_class_index": result["top1_class_index"],
                "top1_class_name": result["top1_class_name"],
                "top1_probability": result["top1_probability"],
                "metadata": result["metadata"],
            }
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/explain/overlay")
async def explain_overlay(file: UploadFile = File(...)):
    if not service.is_loaded:
        raise HTTPException(status_code=400, detail="No model is loaded.")

    try:
        image_bytes = await file.read()
        result = service.explain_from_bytes(image_bytes)

        overlay = result["overlay"]
        image = Image.fromarray(overlay)
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return Response(content=buffer.getvalue(), media_type="image/png")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc