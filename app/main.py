"""
Customer Monitoring System - Backend
ML Problem Type: Classification (XGBoost) + NLP (text feature extraction)
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_router
from app.core.config import settings
from app.core.model_registry import ModelRegistry

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all ML models on startup."""
    logger.info("Initializing Customer Monitoring System...")
    app.state.started_at = time.time()
    app.state.model_registry = ModelRegistry()
    await app.state.model_registry.load_all()
    logger.info("All models ready.")
    yield
    logger.info("Shutting down, releasing resources...")
    app.state.model_registry.cleanup()


app = FastAPI(
    title="Customer Monitoring System API",
    description=(
        "ML-powered customer monitoring: churn prediction, sentiment analysis, "
        "anomaly detection, segmentation — powered by XGBoost + NLP."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")


@app.get("/", tags=["Root"])
async def root():
    return {
        "service": "Customer Monitoring System",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health",
    }
