"""Services for ru-sentiment service."""
from __future__ import annotations

import io

from typing import TYPE_CHECKING
from PIL import Image

from models import predict_captions
from src.schemas import PredictionResult
from src.ml_sentiment_model import MLSentimentModel

if TYPE_CHECKING:
    from fastapi import UploadFile
    from typing import List, Optional
    from uuid import UUID
    from src.clickhouse_client import ClickHouse
    from src.minio_client import Minio
    from src.schemas import DetailedPredictionData, Summary


ml_sentiment_model = MLSentimentModel()


def get_prediction_by_text_and_images(
    store: bool,
    text: Optional[str],
    images: Optional[List[UploadFile]],
    click_house: ClickHouse,
    minio: Minio,
) -> PredictionResult:
    """Get sentiment prediction by images and text."""
    pil_images = None
    image_captions = None

    if images:
        pil_images = []
        for image in images:
            image_bytes = io.BytesIO(image.file.read())
            pil_images.append(Image.open(image_bytes))
        image_captions = predict_captions(pil_images)

    prediction_result = ml_sentiment_model.predict(text, pil_images, image_captions)

    if store:
        prediction_id = click_house.insert_prediction(
            prediction_result,
            text=text,
        )
        prediction_result.prediction_id = prediction_id
        if images:
            saved_images = minio.save_images(
                images,
                image_captions,
            )

            click_house.insert_images(
                prediction_id,
                prediction_result.prediction_details.image_sentiment,
                saved_images,
            )

    return prediction_result


def set_labeled_sentiment_to_prediction(
    prediction_id: UUID,
    sentiment: str,
    click_house: ClickHouse,
) -> None:
    """Set labeled sentiment to prediction."""
    return click_house.set_labeled_prediction(prediction_id, sentiment)


def get_prediction_summary(
    features: Optional[bool],
    expand: Optional[bool],
    click_house: ClickHouse,
) -> Summary:
    """Get prediction summary."""
    return click_house.prediction_summary(features, expand)


def get_prediction_details(
    prediction_id: UUID,
    click_house: ClickHouse,
) -> DetailedPredictionData:
    """Get prediction details."""
    return click_house.get_prediction_by_id(prediction_id)
