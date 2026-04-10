"""Anomaly detection routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.schemas.schemas import AnomalyResult, CustomerFeatures
from app.services.monitoring_service import CustomerMonitoringService

router = APIRouter()


def _get_service(request: Request) -> CustomerMonitoringService:
    registry = request.app.state.model_registry
    if not registry._status["anomaly"]:
        raise HTTPException(status_code=503, detail="Anomaly model not loaded.")
    return CustomerMonitoringService(registry)


@router.post("/detect", response_model=AnomalyResult, summary="Detect behavioral anomalies")
async def detect_anomaly(customer: CustomerFeatures, request: Request) -> AnomalyResult:
    svc = _get_service(request)
    return svc.detect_anomaly(customer)
