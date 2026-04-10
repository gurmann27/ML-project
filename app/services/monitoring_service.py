"""
CustomerMonitoringService — orchestrates XGBoost + NLP for all predictions.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Optional

from app.core.model_registry import ModelRegistry
from app.schemas.schemas import (
    AnomalyResult,
    ChurnPrediction,
    CustomerFeatures,
    CustomerMonitoringReport,
    CustomerSegment,
    RiskLevel,
    SegmentationResult,
    SentimentLabel,
    SentimentResponse,
)

logger = logging.getLogger(__name__)


def _churn_risk_level(prob: float) -> RiskLevel:
    if prob >= 0.75:
        return RiskLevel.CRITICAL
    if prob >= 0.5:
        return RiskLevel.HIGH
    if prob >= 0.25:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


def _recommended_actions(risk: RiskLevel, segment: str, factors: List[Dict]) -> List[str]:
    actions = []
    factor_names = [f["feature"] for f in factors]

    if risk in (RiskLevel.HIGH, RiskLevel.CRITICAL):
        actions.append("Trigger immediate retention outreach via email/call.")
    if "support_tickets_90d" in factor_names:
        actions.append("Escalate open support tickets to senior tier.")
    if "nps_score" in factor_names:
        actions.append("Send personalized NPS follow-up survey.")
    if "last_activity_days_ago" in factor_names:
        actions.append("Launch re-engagement campaign with incentive.")
    if "late_payments_count" in factor_names:
        actions.append("Offer flexible payment plan or billing reminder.")
    if segment == "at_risk":
        actions.append("Assign dedicated customer success manager.")
    if segment in ("champion", "loyal"):
        actions.append("Offer loyalty reward or upgrade promotion.")
    if not actions:
        actions.append("Monitor customer health score weekly.")
    return actions


def _compute_health_score(
    churn_prob: float,
    sentiment: Optional[str],
    anomaly: bool,
    nps: Optional[int],
) -> float:
    score = 100.0
    score -= churn_prob * 50
    if anomaly:
        score -= 15
    if sentiment == "negative":
        score -= 15
    elif sentiment == "neutral":
        score -= 5
    if nps is not None:
        score += (nps - 5) * 2
    return max(0.0, min(100.0, round(score, 1)))


class CustomerMonitoringService:

    def __init__(self, registry: ModelRegistry):
        self.registry = registry

    def predict_churn(self, customer: CustomerFeatures) -> ChurnPrediction:
        features = customer.model_dump(exclude={"customer_id", "last_feedback"})

        if customer.last_feedback and self.registry._status["nlp"]:
            text_features = self.registry.nlp_service.text_to_features(customer.last_feedback)
            features.update(text_features)

        result = self.registry.churn_model.predict_single(features)
        prob = result["churn_probability"]
        risk = _churn_risk_level(prob)

        segment_name = "unknown"
        if self.registry._status["segment"]:
            try:
                seg = self.registry.segment_model.predict(features)
                segment_name = seg["segment"]
            except Exception:
                pass

        return ChurnPrediction(
            customer_id=customer.customer_id,
            churn_probability=round(prob, 4),
            risk_level=risk,
            top_risk_factors=result["top_risk_factors"],
            recommended_actions=_recommended_actions(risk, segment_name, result["top_risk_factors"]),
            predicted_at=datetime.utcnow(),
        )

    def predict_churn_batch(self, customers: List[CustomerFeatures]) -> List[ChurnPrediction]:
        return [self.predict_churn(c) for c in customers]

    def segment_customer(self, customer: CustomerFeatures) -> SegmentationResult:
        features = customer.model_dump(exclude={"customer_id", "last_feedback"})
        result = self.registry.segment_model.predict(features)

        upsell_map = {
            "champion": ["Annual plan upgrade", "Referral bonus program"],
            "loyal": ["Premium tier promotion", "Exclusive features preview"],
            "at_risk": ["Discount offer", "Personalized check-in call"],
            "hibernating": ["Win-back campaign", "Feature highlight email"],
            "new": ["Onboarding guide", "30-day check-in"],
            "potential": ["Feature demo", "Case study sharing"],
        }

        return SegmentationResult(
            customer_id=customer.customer_id,
            segment=CustomerSegment(result["segment"]),
            segment_confidence=result["segment_confidence"],
            rfm_scores=result["rfm_scores"],
            segment_description=result["segment_description"],
            upsell_opportunities=upsell_map.get(result["segment"], []),
        )

    def detect_anomaly(self, customer: CustomerFeatures) -> AnomalyResult:
        features = customer.model_dump(exclude={"customer_id", "last_feedback"})
        result = self.registry.anomaly_model.detect(features)

        anomaly_type = None
        if result["is_anomaly"]:
            if "frequent_late_payments" in result["flagged_features"]:
                anomaly_type = "payment_anomaly"
            elif "unusually_high_support_tickets" in result["flagged_features"]:
                anomaly_type = "support_abuse_pattern"
            elif "very_long_inactivity" in result["flagged_features"]:
                anomaly_type = "inactivity_anomaly"
            else:
                anomaly_type = "behavioral_anomaly"

        return AnomalyResult(
            customer_id=customer.customer_id,
            is_anomaly=result["is_anomaly"],
            anomaly_score=result["anomaly_score"],
            anomaly_type=anomaly_type,
            flagged_features=result["flagged_features"],
            severity=RiskLevel(result["severity"]),
            detected_at=datetime.utcnow(),
        )

    def analyze_sentiment(self, customer_id: str, text: str, source: str = "feedback") -> SentimentResponse:
        result = self.registry.nlp_service.analyze(customer_id, text, source)
        return SentimentResponse(
            customer_id=customer_id,
            text=text,
            sentiment=SentimentLabel(result["sentiment"]),
            confidence=result["confidence"],
            scores=result["scores"],
            keywords=result["keywords"],
            analyzed_at=result["analyzed_at"],
        )

    def full_report(self, customer: CustomerFeatures) -> CustomerMonitoringReport:
        churn = self.predict_churn(customer)
        segment = self.segment_customer(customer) if self.registry._status["segment"] else None
        anomaly = self.detect_anomaly(customer) if self.registry._status["anomaly"] else None
        sentiment = None
        if customer.last_feedback and self.registry._status["nlp"]:
            sentiment = self.analyze_sentiment(customer.customer_id, customer.last_feedback)

        sentiment_str = sentiment.sentiment.value if sentiment else None
        health = _compute_health_score(
            churn_prob=churn.churn_probability,
            sentiment=sentiment_str,
            anomaly=anomaly.is_anomaly if anomaly else False,
            nps=customer.nps_score,
        )

        return CustomerMonitoringReport(
            customer_id=customer.customer_id,
            churn=churn,
            segment=segment,
            anomaly=anomaly,
            sentiment=sentiment,
            health_score=health,
            priority_alert=churn.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL),
            generated_at=datetime.utcnow(),
        )
