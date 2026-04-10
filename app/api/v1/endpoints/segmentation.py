"""Customer segmentation routes."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, HTTPException, Request

from app.schemas.schemas import (
    BatchCustomerRequest,
    BatchSegmentResponse,
    CustomerFeatures,
    SegmentationResult,
)
from app.services.monitoring_service import CustomerMonitoringService

router = APIRouter()


def _get_service(request: Request) -> CustomerMonitoringService:
    registry = request.app.state.model_registry
    if not registry._status["segment"]:
        raise HTTPException(status_code=503, detail="Segmentation model not loaded.")
    return CustomerMonitoringService(registry)


@router.post("/predict", response_model=SegmentationResult, summary="Segment one customer")
async def predict_segment(customer: CustomerFeatures, request: Request) -> SegmentationResult:
    svc = _get_service(request)
    return svc.segment_customer(customer)


@router.post("/predict/batch", response_model=BatchSegmentResponse, summary="Batch segmentation")
async def predict_segment_batch(body: BatchCustomerRequest, request: Request) -> BatchSegmentResponse:
    svc = _get_service(request)
    results = [svc.segment_customer(c) for c in body.customers]
    dist: dict[str, int] = {}
    for r in results:
        dist[r.segment.value] = dist.get(r.segment.value, 0) + 1

    return BatchSegmentResponse(
        total_customers=len(results),
        segment_distribution=dist,
        results=results,
        processed_at=datetime.utcnow(),
    )
