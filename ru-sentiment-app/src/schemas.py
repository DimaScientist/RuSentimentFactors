"""PyDantic Schemas for service."""
from pydantic import BaseModel
from typing import List, Optional


class HealthCheck(BaseModel):
    """HealthCheck schema."""

    service: bool


class SentimentPrediction(BaseModel):
    """Sentiment prediction schema."""

    positive: float
    neutral: float
    negative: float


class PredictionProbabilities(BaseModel):
    """Prediction probabilities schema."""

    predict_proba: SentimentPrediction


class PredictionDetails(BaseModel):
    """Prediction details schema."""

    text_sentiment: Optional[PredictionProbabilities] = None
    image_sentiment: Optional[List[PredictionProbabilities]] = None


class PredictionResult(BaseModel):
    """Prediction result schema."""

    prediction_id: int
    prediction_result: str
    prediction_details: Optional[PredictionDetails] = None
