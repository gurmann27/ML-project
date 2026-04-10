"""Churn prediction routes."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, HTTPException, Request

from app.schemas.schemas import (
    BatchChurnResponse,
    BatchCustomerRequest,
    ChurnPrediction,
    CustomerFeatures,
    RiskLevel,
)
from app.services.monitoring_service import CustomerMonitoringService

router = APIRouter()


def _get_service(request: Request) -> CustomerMonitoringService:
    registry = request.app.state.model_registry
    if not registry._status["churn"]:
        raise HTTPException(
            status_code=503,
            detail="Churn model not loaded. POST /api/v1/train/churn to train first.",
        )
    return CustomerMonitoringService(registry)


@router.post("/predict", response_model=ChurnPrediction, summary="Predict churn for one customer")
async def predict_churn(customer: CustomerFeatures, request: Request) -> ChurnPrediction:
    svc = _get_service(request)
    return svc.predict_churn(customer)


@router.post("/predict/batch", response_model=BatchChurnResponse, summary="Batch churn prediction")
async def predict_churn_batch(body: BatchCustomerRequest, request: Request) -> BatchChurnResponse:
    svc = _get_service(request)
    predictions = svc.predict_churn_batch(body.customers)

    high_risk = sum(
        1 for p in predictions if p.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)
    )
    risk_dist: dict[str, int] = {}
    for p in predictions:
        risk_dist[p.risk_level.value] = risk_dist.get(p.risk_level.value, 0) + 1

    return BatchChurnResponse(
        total_customers=len(predictions),
        high_risk_count=high_risk,
        predictions=predictions,
        summary={
            "risk_distribution": risk_dist,
            "avg_churn_probability": round(
                sum(p.churn_probability for p in predictions) / len(predictions), 4
            ),
        },
        processed_at=datetime.utcnow(),
    )
