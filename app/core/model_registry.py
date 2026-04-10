"""
Loads ML artifacts on startup. Falls back to heuristic demo models when files
or dependencies (e.g. XGBoost native lib) are unavailable.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from app.core.config import settings
from app.core.demo_ml import DemoAnomalyDetector, DemoChurnModel, DemoSegmentModel
from app.services.nlp_keyword import KeywordNLPService

logger = logging.getLogger(__name__)


def _try_import_trainers() -> Tuple[Any, Any, Any]:
    """Import XGBoost trainers only when the native library loads."""
    try:
        from app.models.xgboost_trainer import AnomalyDetector, XGBoostChurnTrainer, XGBoostSegmentTrainer

        return XGBoostChurnTrainer, XGBoostSegmentTrainer, AnomalyDetector
    except Exception as e:
        logger.warning("XGBoost trainer stack unavailable (%s); using demo models only.", e)
        return None, None, None


class ModelRegistry:
    """Owns churn, segment, anomaly, and NLP services."""

    def __init__(self) -> None:
        self.churn_model: Any = None
        self.segment_model: Any = None
        self.anomaly_model: Any = None
        self.nlp_service: Any = None
        self._status: Dict[str, bool] = {
            "churn": False,
            "segment": False,
            "anomaly": False,
            "nlp": False,
        }
        self.loaded_at: Optional[datetime] = None

    async def load_all(self) -> None:
        XGBChurn, XGBSeg, XGBAno = _try_import_trainers()

        # Churn
        if XGBChurn is not None:
            trainer = XGBChurn()
            try:
                trainer.load()
                self.churn_model = trainer
                self._status["churn"] = True
                logger.info("Churn model loaded from disk.")
            except Exception as e:
                logger.warning("Churn artifacts not available (%s); using demo churn model.", e)
                self.churn_model = DemoChurnModel()
                self._status["churn"] = True
        else:
            self.churn_model = DemoChurnModel()
            self._status["churn"] = True

        # Segmentation
        if XGBSeg is not None:
            seg = XGBSeg()
            try:
                seg.load()
                self.segment_model = seg
                self._status["segment"] = True
                logger.info("Segment model loaded from disk.")
            except Exception as e:
                logger.warning("Segment artifacts not available (%s); using demo segment model.", e)
                self.segment_model = DemoSegmentModel()
                self._status["segment"] = True
        else:
            self.segment_model = DemoSegmentModel()
            self._status["segment"] = True

        # Anomaly
        if XGBAno is not None:
            ano = XGBAno()
            try:
                ano.load()
                self.anomaly_model = ano
                self._status["anomaly"] = True
                logger.info("Anomaly model loaded from disk.")
            except Exception as e:
                logger.warning("Anomaly artifacts not available (%s); using demo anomaly model.", e)
                self.anomaly_model = DemoAnomalyDetector()
                self._status["anomaly"] = True
        else:
            self.anomaly_model = DemoAnomalyDetector()
            self._status["anomaly"] = True

        # NLP (lazy import so environments without transformers still start)
        try:
            from app.services.nlp_service import NLPService

            nlp = NLPService(model_name=settings.SENTIMENT_MODEL)
            nlp.load()
            self.nlp_service = nlp
            self._status["nlp"] = True
            logger.info("Transformer NLP loaded.")
        except Exception as e:
            logger.warning("Transformer NLP unavailable (%s); using keyword NLP fallback.", e)
            self.nlp_service = KeywordNLPService()
            self._status["nlp"] = True

        self.loaded_at = datetime.utcnow()

    def cleanup(self) -> None:
        self.churn_model = None
        self.segment_model = None
        self.anomaly_model = None
        self.nlp_service = None
