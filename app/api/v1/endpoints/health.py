"""Health check."""

from __future__ import annotations

import time

from fastapi import APIRouter, Request

from app.schemas.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse, summary="Service health and model status")
async def health(request: Request) -> HealthResponse:
    reg = request.app.state.model_registry
    started = getattr(request.app.state, "started_at", time.time())
    return HealthResponse(
        status="ok",
        models_loaded=dict(reg._status),
        uptime_seconds=round(time.time() - started, 2),
        version="1.0.0",
    )
