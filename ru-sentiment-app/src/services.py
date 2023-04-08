"""Services for ru-sentiment service."""
from __future__ import annotations

import io
import uuid

from loguru import logger
from typing import TYPE_CHECKING
from tqdm import tqdm
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
    from src.schemas import DetailedPredictionData, DownloadedPostFromVK, Summary
    from src.vk_api import VKAPI


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


def predict_and_store_result_for_post(post: DownloadedPostFromVK, click_house: ClickHouse) -> uuid.UUID:
    """Predict sentiment for post and store result."""
    text = post.text
    images = []
    captions = []
    for image_info in post.saved_images:
        pil_image = image_info.image
        caption = image_info.caption or predict_captions([pil_image])[0]

        images.append(pil_image)
        captions.append(caption)

    prediction_result = ml_sentiment_model.predict(text, images, captions)

    prediction_id = click_house.insert_prediction(
        prediction=prediction_result,
        post_id=post.post_id,
        text=text,
    )

    if images:
        click_house.insert_images(
            prediction_id=prediction_id,
            image_details=prediction_result.prediction_details.image_sentiment,
            images=post.saved_images,
        )

    return prediction_id


def get_prediction_for_vk_post(
    post_url: str,
    vk_api: VKAPI,
    click_house: ClickHouse,
    minio: Minio,
) -> DetailedPredictionData:
    """Get prediction for VK post."""
    post_id = post_url.split("wall")[-1]
    post = vk_api.get_post_by_id(post_id, minio)

    prediction_id = predict_and_store_result_for_post(post, click_house)

    result = get_prediction_details(prediction_id, click_house)

    return result


def get_summary_prediction_for_vk_wall(
    owner_url: str,
    vk_api: VKAPI,
    click_house: ClickHouse,
    minio: Minio,
    features: bool = False,
    expand: bool = False,
    post_count: int = 10,
) -> Summary:
    """Get summary for wall."""
    owner = owner_url.split("/")[-1]
    posts = vk_api.get_posts_by_wall(owner, minio, post_count)

    prediction_ids = []
    logger.info(f"Predict sentiment for posts from owner {owner_url} wall:")
    for post in tqdm(posts):
        prediction_id = predict_and_store_result_for_post(post, click_house)
        prediction_ids.append(prediction_id)

    result = click_house.prediction_summary(features, expand, prediction_ids)
    return result
