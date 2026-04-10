"""Sentiment analysis (NLP) routes."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Request

from app.schemas.schemas import BatchSentimentRequest, SentimentLabel, SentimentRequest, SentimentResponse
from app.services.monitoring_service import CustomerMonitoringService

router = APIRouter()


def _get_service(request: Request) -> CustomerMonitoringService:
    registry = request.app.state.model_registry
    if not registry._status["nlp"]:
        raise HTTPException(status_code=503, detail="NLP service not loaded.")
    return CustomerMonitoringService(registry)


@router.post("/analyze", response_model=SentimentResponse, summary="Analyze sentiment of one text")
async def analyze_sentiment(body: SentimentRequest, request: Request) -> SentimentResponse:
    svc = _get_service(request)
    return svc.analyze_sentiment(body.customer_id, body.text, body.source)


@router.post("/analyze/batch", summary="Batch sentiment analysis")
async def analyze_sentiment_batch(body: BatchSentimentRequest, request: Request) -> Dict[str, Any]:
    svc = _get_service(request)
    registry = request.app.state.model_registry
    raw_items = [item.model_dump() for item in body.items]
    results_dicts: List[dict] = registry.nlp_service.analyze_batch(raw_items)
    out: List[SentimentResponse] = []
    for d in results_dicts:
        out.append(
            SentimentResponse(
                customer_id=d["customer_id"],
                text=d["text"],
                sentiment=SentimentLabel(d["sentiment"]),
                confidence=d["confidence"],
                scores=d["scores"],
                keywords=d["keywords"],
                analyzed_at=d["analyzed_at"],
            )
        )
    return {"total": len(out), "results": out, "processed_at": datetime.utcnow()}
