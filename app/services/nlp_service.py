"""
NLP Service for Customer Monitoring System.

Capabilities:
  - Sentiment analysis on feedback/reviews using transformer model
  - Keyword extraction
  - Text feature engineering for XGBoost pipeline
  - Batch processing
"""

import logging
import re
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

logger = logging.getLogger(__name__)


# Common customer-related keywords to track
POSITIVE_KEYWORDS = [
    "great", "excellent", "amazing", "love", "helpful", "fast", "easy",
    "reliable", "recommend", "satisfied", "perfect", "wonderful",
]
NEGATIVE_KEYWORDS = [
    "slow", "broken", "terrible", "awful", "cancel", "refund", "disappointed",
    "useless", "bug", "error", "crash", "unhelpful", "rude", "never again",
]
CHURN_SIGNAL_KEYWORDS = [
    "cancel", "unsubscribe", "leaving", "switching", "competitor",
    "refund", "terrible", "never again", "worst",
]


class NLPService:
    """
    Transformer-based NLP pipeline for customer text analysis.
    Uses cardiffnlp/twitter-roberta-base-sentiment-latest by default.
    Falls back to a lightweight distilbert model if the primary fails.
    """

    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
        self.model_name = model_name
        self.sentiment_pipeline = None
        self.device = 0 if torch.cuda.is_available() else -1
        self._label_map = {}

    def load(self):
        """Load the sentiment pipeline."""
        try:
            logger.info(f"Loading NLP model: {self.model_name}")
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                tokenizer=self.model_name,
                device=self.device,
                truncation=True,
                max_length=512,
            )
            # Warm up
            self.sentiment_pipeline("test")
            logger.info("NLP model loaded successfully.")
        except Exception as e:
            logger.warning(f"Primary NLP model failed ({e}), falling back to distilbert...")
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=self.device,
                truncation=True,
                max_length=512,
            )
            logger.info("Fallback NLP model loaded.")

    def _normalize_label(self, raw_label: str) -> str:
        """Normalize model-specific labels to positive/neutral/negative."""
        raw = raw_label.upper()
        if "POS" in raw or raw in ("LABEL_2", "POSITIVE"):
            return "positive"
        elif "NEG" in raw or raw in ("LABEL_0", "NEGATIVE"):
            return "negative"
        else:
            return "neutral"

    def extract_keywords(self, text: str) -> List[str]:
        """Extract customer-relevant keywords from text."""
        text_lower = text.lower()
        found = []
        for kw in POSITIVE_KEYWORDS + NEGATIVE_KEYWORDS + CHURN_SIGNAL_KEYWORDS:
            if kw in text_lower:
                found.append(kw)
        return list(set(found))

    def has_churn_signals(self, text: str) -> bool:
        """Check if text contains language indicating intent to churn."""
        text_lower = text.lower()
        return any(kw in text_lower for kw in CHURN_SIGNAL_KEYWORDS)

    def analyze(self, customer_id: str, text: str, source: str = "feedback") -> Dict:
        """Analyze sentiment of a single text."""
        if not self.sentiment_pipeline:
            raise RuntimeError("NLP model not loaded. Call load() first.")

        # Get model output
        result = self.sentiment_pipeline(text[:512])[0]
        raw_label = result["label"]
        raw_score = result["score"]

        sentiment = self._normalize_label(raw_label)

        # Build score dict (approximated when model only returns top label)
        if sentiment == "positive":
            scores = {"positive": raw_score, "neutral": (1 - raw_score) / 2,
                      "negative": (1 - raw_score) / 2}
        elif sentiment == "negative":
            scores = {"negative": raw_score, "neutral": (1 - raw_score) / 2,
                      "positive": (1 - raw_score) / 2}
        else:
            scores = {"neutral": raw_score, "positive": (1 - raw_score) / 2,
                      "negative": (1 - raw_score) / 2}

        keywords = self.extract_keywords(text)

        return {
            "customer_id": customer_id,
            "text": text,
            "sentiment": sentiment,
            "confidence": round(raw_score, 4),
            "scores": {k: round(v, 4) for k, v in scores.items()},
            "keywords": keywords,
            "has_churn_signals": self.has_churn_signals(text),
            "source": source,
            "analyzed_at": datetime.utcnow(),
        }

    def analyze_batch(self, items: List[Dict]) -> List[Dict]:
        """Analyze a batch of texts efficiently."""
        results = []
        texts = [item["text"][:512] for item in items]

        # Run batch inference
        raw_results = self.sentiment_pipeline(texts, batch_size=16)

        for item, result in zip(items, raw_results):
            sentiment = self._normalize_label(result["label"])
            score = result["score"]

            if sentiment == "positive":
                scores = {"positive": score, "neutral": (1 - score) / 2, "negative": (1 - score) / 2}
            elif sentiment == "negative":
                scores = {"negative": score, "neutral": (1 - score) / 2, "positive": (1 - score) / 2}
            else:
                scores = {"neutral": score, "positive": (1 - score) / 2, "negative": (1 - score) / 2}

            keywords = self.extract_keywords(item["text"])

            results.append({
                "customer_id": item["customer_id"],
                "text": item["text"],
                "sentiment": sentiment,
                "confidence": round(score, 4),
                "scores": {k: round(v, 4) for k, v in scores.items()},
                "keywords": keywords,
                "has_churn_signals": self.has_churn_signals(item["text"]),
                "source": item.get("source", "feedback"),
                "analyzed_at": datetime.utcnow(),
            })

        return results

    def text_to_features(self, text: Optional[str]) -> Dict:
        """
        Convert free-text to numerical features for XGBoost pipeline.
        Returns engineered features that can be appended to structured data.
        """
        if not text:
            return {
                "text_sentiment_score": 0.0,
                "text_has_churn_signal": 0,
                "text_positive_keyword_count": 0,
                "text_negative_keyword_count": 0,
                "text_length": 0,
            }

        result = self.analyze("_feature_eng_", text)
        positive_kws = sum(1 for kw in result["keywords"] if kw in POSITIVE_KEYWORDS)
        negative_kws = sum(1 for kw in result["keywords"] if kw in NEGATIVE_KEYWORDS)

        # Signed sentiment score: +1.0 = fully positive, -1.0 = fully negative
        score_map = {"positive": result["confidence"], "neutral": 0.0,
                     "negative": -result["confidence"]}
        sentiment_score = score_map[result["sentiment"]]

        return {
            "text_sentiment_score": round(sentiment_score, 4),
            "text_has_churn_signal": int(result["has_churn_signals"]),
            "text_positive_keyword_count": positive_kws,
            "text_negative_keyword_count": negative_kws,
            "text_length": len(text.split()),
        }
