"""Keyword-based sentiment when transformer models cannot be loaded."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Dict, List

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


class KeywordNLPService:
    """Lightweight sentiment + text features for offline / CPU-only setups."""

    def analyze(self, customer_id: str, text: str, source: str = "feedback") -> Dict:
        text_lower = text.lower()
        pos_hits = sum(1 for w in POSITIVE_KEYWORDS if w in text_lower)
        neg_hits = sum(1 for w in NEGATIVE_KEYWORDS if w in text_lower)

        if neg_hits > pos_hits:
            sentiment = "negative"
            confidence = min(0.55 + 0.05 * neg_hits, 0.95)
        elif pos_hits > neg_hits:
            sentiment = "positive"
            confidence = min(0.55 + 0.05 * pos_hits, 0.95)
        else:
            sentiment = "neutral"
            confidence = 0.5

        scores = {
            "positive": round(confidence if sentiment == "positive" else (1 - confidence) / 2, 4),
            "neutral": round(confidence if sentiment == "neutral" else (1 - confidence) / 3, 4),
            "negative": round(confidence if sentiment == "negative" else (1 - confidence) / 2, 4),
        }

        keywords = [w for w in POSITIVE_KEYWORDS + NEGATIVE_KEYWORDS + CHURN_SIGNAL_KEYWORDS if w in text_lower]

        return {
            "customer_id": customer_id,
            "text": text,
            "sentiment": sentiment,
            "confidence": round(confidence, 4),
            "scores": scores,
            "keywords": list(set(keywords))[:20],
            "has_churn_signals": any(k in text_lower for k in CHURN_SIGNAL_KEYWORDS),
            "source": source,
            "analyzed_at": datetime.utcnow(),
        }

    def analyze_batch(self, items: List[Dict]) -> List[Dict]:
        return [self.analyze(item["customer_id"], item["text"], item.get("source", "feedback")) for item in items]

    def text_to_features(self, text: str | None) -> Dict:
        if not text:
            return {
                "text_sentiment_score": 0.0,
                "text_has_churn_signal": 0,
                "text_positive_keyword_count": 0,
                "text_negative_keyword_count": 0,
                "text_length": 0,
            }
        r = self.analyze("_fe_", text)
        pos = sum(1 for k in r["keywords"] if k in POSITIVE_KEYWORDS)
        neg = sum(1 for k in r["keywords"] if k in NEGATIVE_KEYWORDS)
        score_map = {"positive": r["confidence"], "neutral": 0.0, "negative": -r["confidence"]}
        return {
            "text_sentiment_score": round(score_map.get(r["sentiment"], 0.0), 4),
            "text_has_churn_signal": int(r["has_churn_signals"]),
            "text_positive_keyword_count": pos,
            "text_negative_keyword_count": neg,
            "text_length": len(re.split(r"\s+", text.strip())) if text.strip() else 0,
        }
