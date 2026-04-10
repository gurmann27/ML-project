"""
Lightweight inference when trained artifacts are not on disk.
Heuristic outputs match the shapes expected by the API layer.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List

import numpy as np


class DemoChurnModel:
    """Rule-based churn proxy with SHAP-shaped explanations."""

    def predict_single(self, features: Dict[str, Any]) -> Dict[str, Any]:
        tickets = float(features.get("support_tickets_90d", 0))
        late = float(features.get("late_payments_count", 0))
        inactive = float(features.get("last_activity_days_ago", 0))
        nps = features.get("nps_score")
        nps_term = (10 - float(nps)) * 0.02 if nps is not None else 0.1

        prob = (
            0.08
            + min(tickets, 15) * 0.028
            + min(late, 6) * 0.045
            + min(inactive, 120) / 400.0
            + nps_term
        )
        prob = float(np.clip(prob + random.uniform(-0.02, 0.02), 0.02, 0.97))

        factors: List[Dict[str, Any]] = [
            {"feature": "support_tickets_90d", "shap_value": round(min(tickets * 0.03, 0.45), 4)},
            {"feature": "last_activity_days_ago", "shap_value": round(min(inactive / 200.0, 0.35), 4)},
            {"feature": "nps_score", "shap_value": round(-0.12 if nps is not None and nps < 5 else 0.05, 4)},
        ]
        factors.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

        return {"churn_probability": round(prob, 4), "top_risk_factors": factors[:5]}


class DemoSegmentModel:
    """Maps RFM-style signals to a segment label."""

    SEGMENT_DESCRIPTIONS = {
        "champion": "High-value, highly engaged customers.",
        "loyal": "Consistent buyers with strong retention.",
        "at_risk": "Previously engaged, now showing decline.",
        "hibernating": "Long inactive, likely to churn.",
        "new": "Recently acquired, still forming habits.",
        "potential": "Good signals but not yet fully committed.",
    }

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        last = float(features.get("last_activity_days_ago", 0))
        login = float(features.get("login_frequency_30d", 0))
        charge = float(features.get("monthly_charge", 0))
        tenure = float(features.get("tenure_months", 0))

        rfm_recency = 1.0 / (last + 1.0)
        rfm_frequency = login
        rfm_monetary = charge

        if last > 60 and login < 3:
            seg = "hibernating"
        elif last > 30 and charge < 40:
            seg = "at_risk"
        elif tenure < 3:
            seg = "new"
        elif charge > 75 and login > 15:
            seg = "champion"
        elif login > 10:
            seg = "loyal"
        else:
            seg = "potential"

        conf = round(random.uniform(0.72, 0.94), 4)

        return {
            "segment": seg,
            "segment_confidence": conf,
            "rfm_scores": {
                "recency": round(rfm_recency, 4),
                "frequency": round(rfm_frequency, 4),
                "monetary": round(rfm_monetary, 4),
            },
            "segment_description": self.SEGMENT_DESCRIPTIONS[seg],
        }


class DemoAnomalyDetector:
    """Isolation-Forest-like scores using simple thresholds."""

    def detect(self, features: Dict[str, Any]) -> Dict[str, Any]:
        tickets = int(features.get("support_tickets_90d", 0))
        late = int(features.get("late_payments_count", 0))
        inactive = int(features.get("last_activity_days_ago", 0))

        flagged: List[str] = []
        if tickets > 10:
            flagged.append("unusually_high_support_tickets")
        if late > 3:
            flagged.append("frequent_late_payments")
        if inactive > 90:
            flagged.append("very_long_inactivity")

        is_anomaly = len(flagged) >= 2 or (tickets > 12 and late > 2)
        score = -0.45 if is_anomaly else 0.08
        severity = "critical" if is_anomaly and inactive > 80 else ("high" if is_anomaly else "low")

        return {
            "is_anomaly": is_anomaly,
            "anomaly_score": round(score + random.uniform(-0.05, 0.05), 4),
            "flagged_features": flagged,
            "severity": severity,
        }
