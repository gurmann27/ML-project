"""
Test suite for Customer Monitoring System.
Tests: churn, segmentation, anomaly, sentiment, training, drift endpoints.
All ML models are mocked — no GPU or large model downloads required.
"""

import pytest
import io
import pandas as pd
from unittest.mock import MagicMock, patch, AsyncMock
from httpx import AsyncClient, ASGITransport
from datetime import datetime

from app.main import app


# ── Fixtures ───────────────────────────────────────────────────────────────────

SAMPLE_CUSTOMER = {
    "customer_id": "CUST-001",
    "age": 34,
    "gender": "M",
    "location": "Mumbai",
    "tenure_months": 18,
    "subscription_plan": "Premium",
    "monthly_charge": 89.99,
    "total_charges": 1619.82,
    "num_products": 3,
    "contract_type": "One Year",
    "payment_method": "Credit Card",
    "login_frequency_30d": 22,
    "support_tickets_90d": 1,
    "avg_session_duration_min": 14.5,
    "last_activity_days_ago": 2,
    "nps_score": 8,
    "late_payments_count": 0,
    "discount_used": False,
    "referrals_made": 2,
    "last_feedback": "Really love the service, very fast and reliable!",
}

HIGH_RISK_CUSTOMER = {**SAMPLE_CUSTOMER,
    "customer_id": "CUST-HR-001",
    "support_tickets_90d": 12,
    "late_payments_count": 4,
    "last_activity_days_ago": 95,
    "nps_score": 2,
    "login_frequency_30d": 1,
    "last_feedback": "Terrible experience, thinking of cancelling.",
}


def _mock_registry(all_loaded: bool = True):
    """Build a fully mocked ModelRegistry."""
    reg = MagicMock()
    reg._status = {
        "churn": all_loaded,
        "segment": all_loaded,
        "anomaly": all_loaded,
        "nlp": all_loaded,
    }
    reg.loaded_at = datetime.utcnow()

    # Churn model
    reg.churn_model.predict_single.return_value = {
        "churn_probability": 0.72,
        "top_risk_factors": [
            {"feature": "support_tickets_90d", "shap_value": 0.31},
            {"feature": "last_activity_days_ago", "shap_value": 0.24},
            {"feature": "nps_score", "shap_value": -0.18},
        ],
    }

    # Segment model
    reg.segment_model.predict.return_value = {
        "segment": "at_risk",
        "segment_confidence": 0.81,
        "rfm_scores": {"recency": 0.1, "frequency": 0.3, "monetary": 0.6},
        "segment_description": "Previously engaged, now showing decline.",
    }

    # Anomaly model
    reg.anomaly_model.detect.return_value = {
        "is_anomaly": False,
        "anomaly_score": -0.12,
        "flagged_features": [],
        "severity": "low",
    }

    # NLP service
    reg.nlp_service.analyze.return_value = {
        "customer_id": "CUST-001",
        "text": "Really love the service!",
        "sentiment": "positive",
        "confidence": 0.93,
        "scores": {"positive": 0.93, "neutral": 0.04, "negative": 0.03},
        "keywords": ["love"],
        "has_churn_signals": False,
        "source": "feedback",
        "analyzed_at": datetime.utcnow(),
    }
    reg.nlp_service.analyze_batch.return_value = [
        {
            "customer_id": "CUST-001",
            "text": "Great service",
            "sentiment": "positive",
            "confidence": 0.91,
            "scores": {"positive": 0.91, "neutral": 0.05, "negative": 0.04},
            "keywords": ["great"],
            "has_churn_signals": False,
            "source": "feedback",
            "analyzed_at": datetime.utcnow(),
        }
    ]
    reg.nlp_service.text_to_features.return_value = {
        "text_sentiment_score": 0.93,
        "text_has_churn_signal": 0,
        "text_positive_keyword_count": 2,
        "text_negative_keyword_count": 0,
        "text_length": 7,
    }

    return reg


@pytest.fixture(autouse=True)
def inject_mock_registry():
    app.state.model_registry = _mock_registry()


# ── Health ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/api/v1/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["models_loaded"]["churn"] is True
    assert data["models_loaded"]["nlp"] is True


# ── Churn ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_churn_predict_single():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/v1/churn/predict", json=SAMPLE_CUSTOMER)
    assert r.status_code == 200
    data = r.json()
    assert data["customer_id"] == "CUST-001"
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["risk_level"] in ("low", "medium", "high", "critical")
    assert len(data["top_risk_factors"]) > 0
    assert len(data["recommended_actions"]) > 0


@pytest.mark.asyncio
async def test_churn_predict_high_risk():
    app.state.model_registry.churn_model.predict_single.return_value = {
        "churn_probability": 0.91,
        "top_risk_factors": [
            {"feature": "support_tickets_90d", "shap_value": 0.45},
            {"feature": "late_payments_count", "shap_value": 0.33},
        ],
    }
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/v1/churn/predict", json=HIGH_RISK_CUSTOMER)
    assert r.status_code == 200
    data = r.json()
    assert data["risk_level"] == "critical"
    assert data["churn_probability"] >= 0.75


@pytest.mark.asyncio
async def test_churn_predict_batch():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/v1/churn/predict/batch",
                         json={"customers": [SAMPLE_CUSTOMER, HIGH_RISK_CUSTOMER]})
    assert r.status_code == 200
    data = r.json()
    assert data["total_customers"] == 2
    assert len(data["predictions"]) == 2
    assert "risk_distribution" in data["summary"]


@pytest.mark.asyncio
async def test_churn_model_not_loaded():
    app.state.model_registry._status["churn"] = False
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/v1/churn/predict", json=SAMPLE_CUSTOMER)
    assert r.status_code == 503


# ── Segmentation ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_segment_predict():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/v1/segment/predict", json=SAMPLE_CUSTOMER)
    assert r.status_code == 200
    data = r.json()
    assert data["segment"] in ("champion", "loyal", "at_risk", "hibernating", "new", "potential")
    assert 0.0 <= data["segment_confidence"] <= 1.0
    assert "rfm_scores" in data
    assert "upsell_opportunities" in data


@pytest.mark.asyncio
async def test_segment_batch():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/v1/segment/predict/batch",
                         json={"customers": [SAMPLE_CUSTOMER]})
    assert r.status_code == 200
    data = r.json()
    assert data["total_customers"] == 1
    assert "segment_distribution" in data


# ── Anomaly ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_anomaly_detect_normal():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/v1/anomaly/detect", json=SAMPLE_CUSTOMER)
    assert r.status_code == 200
    data = r.json()
    assert data["is_anomaly"] is False
    assert data["severity"] == "low"


@pytest.mark.asyncio
async def test_anomaly_detect_flagged():
    app.state.model_registry.anomaly_model.detect.return_value = {
        "is_anomaly": True,
        "anomaly_score": -0.45,
        "flagged_features": ["unusually_high_support_tickets", "frequent_late_payments"],
        "severity": "critical",
    }
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/v1/anomaly/detect", json=HIGH_RISK_CUSTOMER)
    assert r.status_code == 200
    data = r.json()
    assert data["is_anomaly"] is True
    assert data["severity"] == "critical"
    assert data["anomaly_type"] is not None


# ── Sentiment (NLP) ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_sentiment_positive():
    payload = {
        "customer_id": "CUST-001",
        "text": "Really love the service, very fast and reliable!",
        "source": "feedback",
    }
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/v1/sentiment/analyze", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["sentiment"] == "positive"
    assert 0.0 <= data["confidence"] <= 1.0
    assert "scores" in data


@pytest.mark.asyncio
async def test_sentiment_batch():
    payload = {
        "items": [
            {"customer_id": "CUST-001", "text": "Great service", "source": "feedback"}
        ]
    }
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/v1/sentiment/analyze/batch", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["total"] == 1


# ── Full Report ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_full_report():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/v1/report/full", json=SAMPLE_CUSTOMER)
    assert r.status_code == 200
    data = r.json()
    assert "churn" in data
    assert "segment" in data
    assert "anomaly" in data
    assert "sentiment" in data
    assert 0.0 <= data["health_score"] <= 100.0
    assert isinstance(data["priority_alert"], bool)


# ── Monitoring Dashboard ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_dashboard():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/api/v1/monitoring/dashboard")
    assert r.status_code == 200
    data = r.json()
    assert "models_loaded" in data
    assert "mlops" in data


# ── Training (CSV upload) ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_train_churn_missing_column():
    """Training should fail if 'churned' column is absent."""
    df = pd.DataFrame({"age": [30, 40], "monthly_charge": [50.0, 70.0]})
    csv_bytes = df.to_csv(index=False).encode()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post(
            "/api/v1/train/churn",
            files={"file": ("data.csv", io.BytesIO(csv_bytes), "text/csv")},
        )
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_train_invalid_file_type():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post(
            "/api/v1/train/churn",
            files={"file": ("data.txt", b"not,a,csv", "text/plain")},
        )
    # Should either 400 (parse error) or 422 (validation)
    assert r.status_code in (400, 422, 500)


# ── Input Validation ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_invalid_customer_age():
    bad = {**SAMPLE_CUSTOMER, "age": 200}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/v1/churn/predict", json=bad)
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_invalid_gender():
    bad = {**SAMPLE_CUSTOMER, "gender": "X"}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/v1/churn/predict", json=bad)
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_empty_batch():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/v1/churn/predict/batch", json={"customers": []})
    assert r.status_code == 422
