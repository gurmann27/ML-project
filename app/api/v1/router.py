"""Aggregate API v1 routes."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.endpoints import anomaly, churn, health, monitoring, reports, segmentation, sentiment, training

api_router = APIRouter()
api_router.include_router(health.router, tags=["Health"])
api_router.include_router(churn.router, prefix="/churn", tags=["Churn"])
api_router.include_router(segmentation.router, prefix="/segment", tags=["Segmentation"])
api_router.include_router(anomaly.router, prefix="/anomaly", tags=["Anomaly"])
api_router.include_router(sentiment.router, prefix="/sentiment", tags=["Sentiment"])
api_router.include_router(reports.router, prefix="/report", tags=["Reports"])
api_router.include_router(training.router, prefix="/train", tags=["Training"])
api_router.include_router(monitoring.router, prefix="/monitoring", tags=["Monitoring"])
