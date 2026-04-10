"""
Data Drift Detection Service.
Compares incoming feature distributions to reference training data.
Uses statistical tests (KS test, PSI) to flag distribution shifts.
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime
from scipy import stats

logger = logging.getLogger(__name__)

NUMERIC_FEATURES = [
    "age", "tenure_months", "monthly_charge", "total_charges",
    "login_frequency_30d", "support_tickets_90d", "last_activity_days_ago",
    "late_payments_count",
]


def _psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """Population Stability Index — measures distribution shift."""
    def _scale_range(x, min_val, max_val):
        x = np.clip(x, min_val, max_val)
        return x

    min_val = min(expected.min(), actual.min())
    max_val = max(expected.max(), actual.max()) + 1e-6
    breakpoints = np.linspace(min_val, max_val, buckets + 1)

    expected_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected) + 1e-6
    actual_pct = np.histogram(actual, bins=breakpoints)[0] / len(actual) + 1e-6

    psi_val = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi_val)


class DataDriftService:

    def __init__(self, reference_path: Optional[str] = None):
        self.reference_data: Optional[pd.DataFrame] = None
        self.reference_path = reference_path

    def load_reference(self):
        if self.reference_path:
            try:
                self.reference_data = pd.read_parquet(self.reference_path)
                logger.info(f"Reference data loaded: {len(self.reference_data)} rows")
            except Exception as e:
                logger.warning(f"Could not load reference data: {e}")

    def set_reference(self, df: pd.DataFrame):
        self.reference_data = df.copy()
        logger.info(f"Reference data set: {len(df)} rows")

    def check_drift(self, current_df: pd.DataFrame) -> Dict:
        if self.reference_data is None:
            return {
                "drift_detected": False,
                "drifted_features": [],
                "drift_scores": {},
                "recommendation": "No reference data set. Train a model first.",
                "checked_at": datetime.utcnow(),
            }

        drifted = []
        scores = {}

        for feature in NUMERIC_FEATURES:
            if feature not in self.reference_data.columns or feature not in current_df.columns:
                continue

            ref = self.reference_data[feature].dropna().values
            cur = current_df[feature].dropna().values

            if len(ref) < 10 or len(cur) < 10:
                continue

            # KS Test
            ks_stat, p_value = stats.ks_2samp(ref, cur)
            # PSI
            psi_score = _psi(ref, cur)

            scores[feature] = {
                "ks_statistic": round(ks_stat, 4),
                "ks_p_value": round(p_value, 4),
                "psi": round(psi_score, 4),
            }

            # Flag if KS significant (p < 0.05) or PSI > 0.2 (industry threshold)
            if p_value < 0.05 or psi_score > 0.2:
                drifted.append(feature)

        drift_detected = len(drifted) > 0
        recommendation = (
            "Model retraining recommended — significant drift detected in: "
            + ", ".join(drifted)
            if drift_detected
            else "No significant drift detected. Model is stable."
        )

        return {
            "drift_detected": drift_detected,
            "drifted_features": drifted,
            "drift_scores": scores,
            "recommendation": recommendation,
            "checked_at": datetime.utcnow(),
        }
