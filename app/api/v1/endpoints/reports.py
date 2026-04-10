"""Unified customer monitoring reports."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.schemas.schemas import CustomerFeatures, CustomerMonitoringReport
from app.services.monitoring_service import CustomerMonitoringService

router = APIRouter()


@router.post("/full", response_model=CustomerMonitoringReport, summary="Full churn + segment + anomaly + sentiment report")
async def full_report(customer: CustomerFeatures, request: Request) -> CustomerMonitoringReport:
    registry = request.app.state.model_registry
    if not registry._status["churn"]:
        raise HTTPException(status_code=503, detail="Churn model not available.")
    svc = CustomerMonitoringService(registry)
    return svc.full_report(customer)
