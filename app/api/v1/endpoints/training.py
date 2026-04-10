"""Model training endpoints (CSV upload)."""

from __future__ import annotations

import io
from datetime import datetime

import pandas as pd
from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from app.core.config import settings
from app.schemas.schemas import TrainingResponse

router = APIRouter()


@router.post("/churn", response_model=TrainingResponse, summary="Train churn model from CSV")
async def train_churn(request: Request, file: UploadFile = File(...)) -> TrainingResponse:
    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(400, f"Could not parse CSV: {e}")

    if "churned" not in df.columns:
        raise HTTPException(400, "CSV must have a 'churned' column (0=retained, 1=churned).")

    registry = request.app.state.model_registry
    trainer = registry.churn_model

    if not hasattr(trainer, "train"):
        raise HTTPException(
            503,
            "Training requires a trained XGBoost pipeline (no demo). Train with real artifacts or disable demo.",
        )

    try:
        metrics = trainer.train(
            df,
            target_col="churned",
            enable_mlflow=bool(settings.MLFLOW_TRACKING_URI),
        )
        registry._status["churn"] = True
    except Exception as e:
        raise HTTPException(500, f"Training failed: {e}")

    return TrainingResponse(
        model_version=metrics["model_version"],
        accuracy=metrics["accuracy"],
        roc_auc=metrics["roc_auc"],
        f1_score=metrics["f1_score"],
        precision=metrics["precision"],
        recall=metrics["recall"],
        feature_importances=metrics["feature_importances"],
        training_duration_sec=metrics["training_duration_sec"],
        trained_at=metrics["trained_at"],
    )


@router.post("/segment", summary="Train segmentation model")
async def train_segment(request: Request, file: UploadFile = File(...)) -> dict:
    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(400, f"Could not parse CSV: {e}")

    if "segment" not in df.columns:
        raise HTTPException(400, "CSV must include 'segment' column.")

    registry = request.app.state.model_registry
    try:
        registry.segment_model.train(df)
        registry._status["segment"] = True
    except Exception as e:
        raise HTTPException(500, f"Segment training failed: {e}")

    return {"status": "ok", "message": "Segment model trained.", "trained_at": datetime.utcnow()}


@router.post("/anomaly", summary="Train anomaly detector")
async def train_anomaly(request: Request, file: UploadFile = File(...)) -> dict:
    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(400, f"Could not parse CSV: {e}")

    registry = request.app.state.model_registry
    try:
        registry.anomaly_model.train(df)
        registry._status["anomaly"] = True
    except Exception as e:
        raise HTTPException(500, f"Anomaly training failed: {e}")

    return {"status": "ok", "message": "Anomaly model trained.", "trained_at": datetime.utcnow()}
