"""PyDantic Schemas for service."""
from uuid import UUID

from pydantic import BaseModel, root_validator
from typing import Any, Dict, List, Optional


class ErrorOut(BaseModel):
    """Error response body with only "detail" field."""

    detail: str


class HealthCheck(BaseModel):
    """HealthCheck schema."""

    service: bool


class SentimentPrediction(BaseModel):
    """Sentiment prediction schema."""

    positive: float = 0
    neutral: float = 0
    negative: float = 0

    @root_validator
    def round_values(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Round values."""
        values["positive"] = round(values["positive"], 4)
        values["neutral"] = round(values["neutral"], 4)
        values["negative"] = round(values["negative"], 4)
        return values


class PredictionProbabilities(BaseModel):
    """Prediction probabilities schema."""

    predict_proba: SentimentPrediction


class PredictionDetails(BaseModel):
    """Prediction details schema."""

    text_sentiment: Optional[SentimentPrediction] = None
    image_sentiment: Optional[List[SentimentPrediction]] = None


class PredictionResult(BaseModel):
    """Prediction result schema."""

    prediction_id: Optional[UUID] = None
    prediction_result: str
    prediction_details: Optional[PredictionDetails] = None


class ModelSentimentSimplePrediction(PredictionProbabilities):
    """Schema for sentiment simple prediction."""

    sentiment: str


class ImageTable(BaseModel):
    """Schema for image table in ClickHouse."""

    id: UUID
    image_url: Optional[str] = None
    bucket: str
    key: str
    caption: Optional[str] = None
    prediction_id: UUID
    prediction_details_id: Optional[UUID] = None


class PredictionDetailsTable(BaseModel):
    """Schema for prediction details table in ClickHouse."""

    id: UUID
    negative: float
    neutral: float
    positive: float


class PredictionTable(BaseModel):
    """Schema for prediction table in ClickHouse."""

    id: UUID
    post_id: Optional[str] = None
    post_url: Optional[str] = None
    text: Optional[str] = None
    clean_text: Optional[str] = None
    predicted_value: str
    labeled_value: Optional[str] = None
    text_prediction_details_id: Optional[UUID] = None


class SentimentFeatures(BaseModel):
    """Schema for sentiment features."""

    negative: List[str]
    neutral: List[str]
    positive: List[str]


class ShortPrediction(BaseModel):
    """Schema for short prediction data."""

    id: UUID
    post_id: Optional[str] = None
    post_url: Optional[str] = None
    predicted_value: str


class Summary(SentimentPrediction):
    """Schema for sentiment summary."""

    features: Optional[SentimentFeatures] = None
    predictions: Optional[List[ShortPrediction]] = None


class SavedMinioImage(BaseModel):
    """Schema for saved image MinIO."""

    bucket: str
    key: str
    image_url: Optional[str] = None
    caption: str
    filename: str


class DetailedImageData(BaseModel):
    """Schema for detailed images."""

    id: UUID
    image_url: Optional[str] = None
    filename: str
    prediction_details: SentimentPrediction


class DetailedPredictionData(BaseModel):
    """Schema for detailed predictions."""

    id: UUID
    post_id: Optional[str] = None
    post_url: Optional[str] = None
    text: Optional[str] = None
    predicted_value: str
    text_prediction_details: Optional[SentimentPrediction] = None
    image_info: Optional[List[DetailedImageData]] = None
