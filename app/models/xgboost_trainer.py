"""
XGBoost Trainer for Customer Churn Classification + Segmentation.

ML Problem Type: Classification (binary churn + multiclass segmentation)
Algorithm: XGBoost with SHAP explainability, cross-validation, MLflow tracking.
"""

import os
import time
import logging
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from datetime import datetime

import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, classification_report,
)
from sklearn.ensemble import IsolationForest
import shap
import joblib

from app.core.config import settings

logger = logging.getLogger(__name__)


class XGBoostChurnTrainer:
    """
    Trains an XGBoost classifier for customer churn prediction.
    - Binary classification: churned (1) vs retained (0)
    - SHAP for explainability
    - MLflow tracking (optional)
    - Stratified K-Fold CV
    """

    def __init__(self):
        self.model: Optional[xgb.XGBClassifier] = None
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.feature_names: list = []
        self.shap_explainer = None

        self.xgb_params = {
            "n_estimators": settings.XGB_N_ESTIMATORS,
            "max_depth": settings.XGB_MAX_DEPTH,
            "learning_rate": settings.XGB_LEARNING_RATE,
            "subsample": settings.XGB_SUBSAMPLE,
            "colsample_bytree": settings.XGB_COLSAMPLE_BYTREE,
            "eval_metric": settings.XGB_EVAL_METRIC,
            "use_label_encoder": False,
            "random_state": 42,
            "n_jobs": -1,
            "tree_method": "gpu_hist" if settings.XGB_USE_GPU else "hist",
        }

    def preprocess(self, df: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """Encode categoricals, scale numerics, return feature matrix."""
        df = df.copy()

        categorical_cols = ["gender", "location", "subscription_plan",
                            "contract_type", "payment_method"]
        numeric_cols = [
            "age", "tenure_months", "monthly_charge", "total_charges",
            "num_products", "login_frequency_30d", "support_tickets_90d",
            "avg_session_duration_min", "last_activity_days_ago",
            "nps_score", "late_payments_count", "referrals_made",
        ]

        # Encode categoricals
        for col in categorical_cols:
            if col not in df.columns:
                continue
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders.get(col)
                if le:
                    df[col] = df[col].astype(str).map(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )

        # Boolean → int
        if "discount_used" in df.columns:
            df["discount_used"] = df["discount_used"].astype(int)

        # Derived features
        if "total_charges" in df.columns and "tenure_months" in df.columns:
            df["avg_monthly_revenue"] = df["total_charges"] / (df["tenure_months"] + 1)
        if "support_tickets_90d" in df.columns and "login_frequency_30d" in df.columns:
            df["engagement_ratio"] = df["login_frequency_30d"] / (df["support_tickets_90d"] + 1)

        all_features = numeric_cols + categorical_cols + ["avg_monthly_revenue", "engagement_ratio",
                                                           "discount_used"]
        available = [c for c in all_features if c in df.columns]
        self.feature_names = available

        X = df[available].fillna(0).values
        if fit:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)

        return X

    def train(
        self,
        df: pd.DataFrame,
        target_col: str = "churned",
        test_size: float = 0.2,
        cv_folds: int = 5,
        enable_mlflow: bool = True,
    ) -> Dict:
        """Full training pipeline with CV, SHAP, and optional MLflow logging."""
        start_time = time.time()
        logger.info("Starting XGBoost churn model training...")

        # Split
        from sklearn.model_selection import train_test_split
        y = df[target_col].values
        X = self.preprocess(df.drop(columns=[target_col]), fit=True)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )

        # Cross-validation
        logger.info(f"Running {cv_folds}-fold stratified CV...")
        cv_model = xgb.XGBClassifier(**self.xgb_params)
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(cv_model, X_train, y_train, cv=skf, scoring="roc_auc")
        logger.info(f"CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # Final fit with early stopping
        self.model = xgb.XGBClassifier(**self.xgb_params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=50,
        )

        # Evaluation
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
            "f1_score": round(f1_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred), 4),
            "recall": round(recall_score(y_test, y_pred), 4),
            "cv_roc_auc_mean": round(cv_scores.mean(), 4),
            "cv_roc_auc_std": round(cv_scores.std(), 4),
        }

        logger.info(f"Test metrics: {metrics}")
        logger.info("\n" + classification_report(y_test, y_pred))

        # SHAP explainability
        logger.info("Computing SHAP explainer...")
        self.shap_explainer = shap.TreeExplainer(self.model)

        # Feature importances (gain-based)
        fi = dict(zip(self.feature_names, self.model.feature_importances_))
        metrics["feature_importances"] = dict(
            sorted(fi.items(), key=lambda x: x[1], reverse=True)
        )

        # MLflow logging
        if enable_mlflow and settings.MLFLOW_TRACKING_URI:
            self._log_to_mlflow(metrics, X_train, y_train)

        # Save artifacts
        self._save_artifacts()

        metrics["training_duration_sec"] = round(time.time() - start_time, 2)
        metrics["trained_at"] = datetime.utcnow()
        metrics["model_version"] = f"xgb-churn-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

        return metrics

    def _log_to_mlflow(self, metrics: Dict, X_train, y_train):
        try:
            import mlflow
            mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
            mlflow.set_experiment(settings.EXPERIMENT_NAME)
            with mlflow.start_run():
                mlflow.log_params(self.xgb_params)
                mlflow.log_metrics({k: v for k, v in metrics.items()
                                    if isinstance(v, (int, float))})
                mlflow.xgboost.log_model(self.model, "churn_xgb_model")
                logger.info("MLflow run logged.")
        except Exception as e:
            logger.warning(f"MLflow logging failed (non-fatal): {e}")

    def _save_artifacts(self):
        os.makedirs(settings.MODEL_DIR, exist_ok=True)
        joblib.dump(self.model, settings.CHURN_MODEL_PATH)
        joblib.dump(self.scaler, settings.SCALER_PATH)
        joblib.dump(self.label_encoders, settings.ENCODER_PATH)
        logger.info(f"Artifacts saved to {settings.MODEL_DIR}")

    def predict_single(self, features: dict) -> Dict:
        """Predict churn for one customer with SHAP explanation."""
        df = pd.DataFrame([features])
        X = self.preprocess(df, fit=False)
        prob = float(self.model.predict_proba(X)[0][1])

        # SHAP explanation
        shap_values = self.shap_explainer.shap_values(X)[0]
        top_factors = sorted(
            [{"feature": f, "shap_value": round(float(v), 4)}
             for f, v in zip(self.feature_names, shap_values)],
            key=lambda x: abs(x["shap_value"]),
            reverse=True,
        )[:5]

        return {"churn_probability": prob, "top_risk_factors": top_factors}

    def load(self):
        """Load saved artifacts from disk."""
        self.model = joblib.load(settings.CHURN_MODEL_PATH)
        self.scaler = joblib.load(settings.SCALER_PATH)
        self.label_encoders = joblib.load(settings.ENCODER_PATH)
        self.shap_explainer = shap.TreeExplainer(self.model)
        logger.info("Churn XGBoost model loaded from disk.")


class XGBoostSegmentTrainer:
    """
    Multiclass XGBoost classifier for customer segmentation.
    Segments: Champion, Loyal, At-Risk, Hibernating, New, Potential
    Uses RFM (Recency, Frequency, Monetary) features.
    """

    SEGMENT_LABELS = {
        0: "champion", 1: "loyal", 2: "at_risk",
        3: "hibernating", 4: "new", 5: "potential",
    }

    SEGMENT_DESCRIPTIONS = {
        "champion": "High-value, highly engaged customers.",
        "loyal": "Consistent buyers with strong retention.",
        "at_risk": "Previously engaged, now showing decline.",
        "hibernating": "Long inactive, likely to churn.",
        "new": "Recently acquired, still forming habits.",
        "potential": "Good signals but not yet fully committed.",
    }

    def __init__(self):
        self.model: Optional[xgb.XGBClassifier] = None
        self.scaler = StandardScaler()

        self.xgb_params = {
            "n_estimators": 200,
            "max_depth": 5,
            "learning_rate": 0.08,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
            "use_label_encoder": False,
            "random_state": 42,
            "n_jobs": -1,
        }

    def compute_rfm(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive RFM scores from raw features."""
        df = df.copy()
        df["rfm_recency"] = 1 / (df["last_activity_days_ago"] + 1)
        df["rfm_frequency"] = df["login_frequency_30d"]
        df["rfm_monetary"] = df["monthly_charge"]
        return df

    def train(self, df: pd.DataFrame, target_col: str = "segment") -> None:
        df = self.compute_rfm(df)
        feature_cols = ["rfm_recency", "rfm_frequency", "rfm_monetary",
                        "tenure_months", "num_products", "support_tickets_90d"]
        X = self.scaler.fit_transform(df[feature_cols].fillna(0))
        y = LabelEncoder().fit_transform(df[target_col])

        self.model = xgb.XGBClassifier(**self.xgb_params)
        self.model.fit(X, y)
        joblib.dump(self.model, settings.SEGMENT_MODEL_PATH)
        logger.info("Segment XGBoost model trained and saved.")

    def predict(self, features: dict) -> Dict:
        df = pd.DataFrame([features])
        df = self.compute_rfm(df)
        feature_cols = ["rfm_recency", "rfm_frequency", "rfm_monetary",
                        "tenure_months", "num_products", "support_tickets_90d"]
        available = [c for c in feature_cols if c in df.columns]
        X = self.scaler.transform(df[available].fillna(0))

        probs = self.model.predict_proba(X)[0]
        pred_class = int(np.argmax(probs))
        segment = self.SEGMENT_LABELS[pred_class]

        return {
            "segment": segment,
            "segment_confidence": round(float(probs[pred_class]), 4),
            "rfm_scores": {
                "recency": round(float(df["rfm_recency"].iloc[0]), 4),
                "frequency": round(float(df["rfm_frequency"].iloc[0]), 4),
                "monetary": round(float(df["rfm_monetary"].iloc[0]), 4),
            },
            "segment_description": self.SEGMENT_DESCRIPTIONS[segment],
        }

    def load(self):
        self.model = joblib.load(settings.SEGMENT_MODEL_PATH)
        logger.info("Segment XGBoost model loaded from disk.")


class AnomalyDetector:
    """
    Isolation Forest for detecting anomalous customer behavior.
    Flags unusual activity patterns that may indicate fraud or data issues.
    """

    def __init__(self):
        self.model: Optional[IsolationForest] = None
        self.scaler = StandardScaler()
        self.feature_cols = [
            "monthly_charge", "support_tickets_90d", "login_frequency_30d",
            "last_activity_days_ago", "late_payments_count", "total_charges",
        ]

    def train(self, df: pd.DataFrame) -> None:
        X = self.scaler.fit_transform(df[self.feature_cols].fillna(0))
        self.model = IsolationForest(
            contamination=settings.ANOMALY_CONTAMINATION,
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X)
        joblib.dump(self.model, settings.ANOMALY_MODEL_PATH)
        logger.info("Anomaly detection model trained and saved.")

    def detect(self, features: dict) -> Dict:
        df = pd.DataFrame([features])
        available = [c for c in self.feature_cols if c in df.columns]
        X = self.scaler.transform(df[available].fillna(0))

        score = float(self.model.score_samples(X)[0])
        is_anomaly = self.model.predict(X)[0] == -1

        flagged = []
        if features.get("support_tickets_90d", 0) > 10:
            flagged.append("unusually_high_support_tickets")
        if features.get("late_payments_count", 0) > 3:
            flagged.append("frequent_late_payments")
        if features.get("last_activity_days_ago", 0) > 90:
            flagged.append("very_long_inactivity")

        severity_map = {
            True: "critical" if score < -0.3 else "high",
            False: "low",
        }

        return {
            "is_anomaly": is_anomaly,
            "anomaly_score": round(score, 4),
            "flagged_features": flagged,
            "severity": severity_map[is_anomaly],
        }

    def load(self):
        self.model = joblib.load(settings.ANOMALY_MODEL_PATH)
        logger.info("Anomaly detection model loaded from disk.")
