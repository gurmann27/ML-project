"""
Pydantic schemas for Customer Monitoring System.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CustomerSegment(str, Enum):
    CHAMPION = "champion"
    LOYAL = "loyal"
    AT_RISK = "at_risk"
    HIBERNATING = "hibernating"
    NEW = "new"
    POTENTIAL = "potential"


class SentimentLabel(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class CustomerFeatures(BaseModel):
    customer_id: str = Field(..., description="Unique customer identifier")
    age: int = Field(..., ge=18, le=100)
    gender: str = Field(..., pattern="^(M|F|Other)$")
    location: str
    tenure_months: int = Field(..., ge=0, description="Months as a customer")
    subscription_plan: str = Field(..., description="Basic | Standard | Premium")
    monthly_charge: float = Field(..., ge=0)
    total_charges: float = Field(..., ge=0)
    num_products: int = Field(..., ge=1)
    contract_type: str = Field(..., description="Month-to-Month | One Year | Two Year")
    payment_method: str
    login_frequency_30d: int = Field(..., ge=0)
    support_tickets_90d: int = Field(..., ge=0)
    avg_session_duration_min: float = Field(..., ge=0)
    last_activity_days_ago: int = Field(..., ge=0)
    nps_score: Optional[int] = Field(None, ge=0, le=10)
    late_payments_count: int = Field(..., ge=0)
    discount_used: bool = False
    referrals_made: int = Field(0, ge=0)
    last_feedback: Optional[str] = Field(None, max_length=2000)


class BatchCustomerRequest(BaseModel):
    customers: List[CustomerFeatures] = Field(..., min_length=1, max_length=1000)


class ChurnPrediction(BaseModel):
    customer_id: str
    churn_probability: float = Field(..., ge=0.0, le=1.0)
    risk_level: RiskLevel
    top_risk_factors: List[Dict[str, Any]]
    recommended_actions: List[str]
    predicted_at: datetime


class BatchChurnResponse(BaseModel):
    total_customers: int
    high_risk_count: int
    predictions: List[ChurnPrediction]
    summary: Dict[str, Any]
    processed_at: datetime


class SegmentationResult(BaseModel):
    customer_id: str
    segment: CustomerSegment
    segment_confidence: float
    rfm_scores: Dict[str, float]
    segment_description: str
    upsell_opportunities: List[str]


class BatchSegmentResponse(BaseModel):
    total_customers: int
    segment_distribution: Dict[str, int]
    results: List[SegmentationResult]
    processed_at: datetime


class AnomalyResult(BaseModel):
    customer_id: str
    is_anomaly: bool
    anomaly_score: float
    anomaly_type: Optional[str]
    flagged_features: List[str]
    severity: RiskLevel
    detected_at: datetime


class SentimentRequest(BaseModel):
    customer_id: str
    text: str = Field(..., min_length=1, max_length=5000)
    source: str = Field("feedback", description="feedback | review | support_ticket | chat")


class SentimentResponse(BaseModel):
    customer_id: str
    text: str
    sentiment: SentimentLabel
    confidence: float
    scores: Dict[str, float]
    keywords: List[str]
    analyzed_at: datetime


class BatchSentimentRequest(BaseModel):
    items: List[SentimentRequest] = Field(..., min_length=1, max_length=500)


class CustomerMonitoringReport(BaseModel):
    customer_id: str
    churn: ChurnPrediction
    segment: Optional[SegmentationResult] = None
    anomaly: Optional[AnomalyResult] = None
    sentiment: Optional[SentimentResponse] = None
    health_score: float = Field(..., ge=0, le=100)
    priority_alert: bool
    generated_at: datetime


class DashboardMetrics(BaseModel):
    total_customers_monitored: int
    high_risk_churn_count: int
    churn_rate_30d: float
    avg_health_score: float
    segment_breakdown: Dict[str, int]
    sentiment_breakdown: Dict[str, int]
    anomalies_detected_today: int
    model_version: str
    last_retrained: Optional[datetime]


class TrainingRequest(BaseModel):
    dataset_path: str
    target_column: str = "churned"
    test_size: float = Field(0.2, ge=0.1, le=0.4)
    cv_folds: int = Field(5, ge=3, le=10)
    enable_mlflow: bool = True


class TrainingResponse(BaseModel):
    model_version: str
    accuracy: float
    roc_auc: float
    f1_score: float
    precision: float
    recall: float
    feature_importances: Dict[str, float]
    training_duration_sec: float
    trained_at: datetime


class DriftReport(BaseModel):
    drift_detected: bool
    drifted_features: List[str]
    drift_scores: Dict[str, float]
    recommendation: str
    checked_at: datetime


class HealthResponse(BaseModel):
    status: str
    models_loaded: Dict[str, bool]
    uptime_seconds: float
    version: str
