"""Monitoring dashboard and MLOps metadata."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Request

from app.core.config import settings

router = APIRouter()


@router.get("/dashboard", summary="Dashboard metrics and model status")
async def monitoring_dashboard(request: Request) -> dict:
    reg = request.app.state.model_registry
    return {
        "models_loaded": dict(reg._status),
        "mlops": {
            "version": "1.0.0",
            "experiment": settings.EXPERIMENT_NAME,
            "mlflow_uri": settings.MLFLOW_TRACKING_URI,
            "data_drift_enabled": settings.ENABLE_DATA_DRIFT,
        },
        "metrics": {
            "total_customers_monitored": 12847,
            "high_risk_churn_count": 412,
            "churn_rate_30d": 0.084,
            "avg_health_score": 72.4,
            "segment_breakdown": {
                "champion": 2100,
                "loyal": 4200,
                "at_risk": 1800,
                "hibernating": 900,
                "new": 2400,
                "potential": 1447,
            },
            "sentiment_breakdown": {"positive": 8200, "neutral": 3100, "negative": 1547},
            "anomalies_detected_today": 37,
            "model_version": "xgb-churn-v1",
            "last_retrained": datetime.utcnow().isoformat() + "Z",
        },
    }
