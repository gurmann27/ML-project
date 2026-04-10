"""Application settings from environment variables."""

from __future__ import annotations

import json
from typing import Any, List, Optional, Union

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    APP_NAME: str = "Customer Monitoring System"
    DEBUG: bool = False
    SECRET_KEY: str = "change-me-in-development"

    ALLOWED_ORIGINS: List[str] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://127.0.0.1:3000",
        ]
    )

    DATABASE_URL: str = "sqlite:///./customer_monitoring.db"
    MODEL_DIR: str = "models/artifacts"
    CHURN_MODEL_PATH: str = "models/artifacts/churn_xgb.joblib"
    SEGMENT_MODEL_PATH: str = "models/artifacts/segment_xgb.joblib"
    ANOMALY_MODEL_PATH: str = "models/artifacts/anomaly_iso.joblib"
    SCALER_PATH: str = "models/artifacts/feature_scaler.joblib"
    ENCODER_PATH: str = "models/artifacts/label_encoder.joblib"

    SENTIMENT_MODEL: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"

    XGB_N_ESTIMATORS: int = 300
    XGB_MAX_DEPTH: int = 6
    XGB_LEARNING_RATE: float = 0.05
    XGB_SUBSAMPLE: float = 0.8
    XGB_COLSAMPLE_BYTREE: float = 0.8
    XGB_EVAL_METRIC: str = "logloss"
    XGB_USE_GPU: bool = False

    CHURN_THRESHOLD: float = 0.5
    ANOMALY_CONTAMINATION: float = 0.05

    MLFLOW_TRACKING_URI: Optional[str] = None
    EXPERIMENT_NAME: str = "customer-monitoring"
    ENABLE_DATA_DRIFT: bool = True

    MODEL_STORAGE_BACKEND: str = "local"

    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def parse_allowed_origins(cls, v: Union[str, List[str]]) -> Any:
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return [s.strip() for s in v.split(",") if s.strip()]
        return v


settings = Settings()
