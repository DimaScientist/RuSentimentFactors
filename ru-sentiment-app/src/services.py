"""Services for ru-sentiment service."""
from __future__ import annotations

import io
import uuid

from loguru import logger
from typing import TYPE_CHECKING
from PIL import Image

from models import predict_captions
from src.errors import BadRequestException
from src.schemas import PredictionResult
from src.text_preprocessing import preprocess_text
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

    clean_text = preprocess_text(text, use_lemmatization=True)

    if images:
        pil_images = []
        for image in images:
            image_bytes = io.BytesIO(image.file.read())
            pil_images.append(Image.open(image_bytes))
        image_captions = predict_captions(pil_images)

    prediction_result = ml_sentiment_model.predict(clean_text, pil_images, image_captions)

    if store:
        prediction_id = click_house.insert_prediction(
            prediction_result,
            text=text,
            clean_text=clean_text,
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

    logger.info(f"Prediction by image and text result: {prediction_result.dict()}.")

    return prediction_result


def set_labeled_sentiment_to_prediction(
    prediction_id: UUID,
    sentiment: str,
    click_house: ClickHouse,
) -> None:
    """Set labeled sentiment to prediction."""
    logger.info(f"Set human sentiment to {str(prediction_id)} is {sentiment}.")
    return click_house.set_labeled_prediction(prediction_id, sentiment)


def get_prediction_summary(
    features: Optional[bool],
    expand: Optional[bool],
    click_house: ClickHouse,
) -> Summary:
    """Get prediction summary."""
    logger.info(f"Get prediction summary with options:\nadd features{features}\nadd predictions:{expand}.")
    return click_house.prediction_summary(features, expand)


def get_prediction_details(
    prediction_id: UUID,
    click_house: ClickHouse,
) -> DetailedPredictionData:
    """Get prediction details."""
    logger.info(f"Get prediction details for {str(prediction_id)}.")
    return click_house.get_prediction_by_id(prediction_id)


def predict_and_store_result_for_post(post: DownloadedPostFromVK, click_house: ClickHouse) -> uuid.UUID:
    """Predict sentiment for post and store result."""
    text = post.text
    images = []
    captions = []

    logger.info(f"Start sentiment analysis for post {post.post_id}...")
    for image_info in post.saved_images:
        pil_image = image_info.image
        caption = image_info.caption or predict_captions([pil_image])[0]

        images.append(pil_image)
        captions.append(caption)

    logger.info(
        f"""
    Post info: id={post.post_id}
    has text={text is not None}
    has images={post.saved_images is not None and len(post.saved_images) > 0}.
    """
    )

    clean_text = preprocess_text(text, use_lemmatization=True)
    prediction_result = ml_sentiment_model.predict(clean_text, images, captions)
    logger.info(f"Prediction result {prediction_result.dict()}.")

    prediction_id = click_house.insert_prediction(
        prediction=prediction_result,
        post_id=post.post_id,
        text=text,
        clean_text=clean_text,
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
    logger.info(f"Predict sentiment for post: {post_url}...")
    post_id = post_url.split("wall")[-1]
    post = vk_api.get_post_by_id(post_id, minio)

    if post.text or post.saved_images:
        prediction_id = predict_and_store_result_for_post(post, click_house)

        result = get_prediction_details(prediction_id, click_house)

        logger.info("Sentiment analysis for post complete.")
        return result
    else:
        raise BadRequestException(message=f"Post {post.post_id} hasn't images and text.")


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
    for i, post in enumerate(posts, start=1):
        logger.info(f"{i} / {len(posts)}: {post.post_id}.")
        if post.text or post.saved_images:
            prediction_id = predict_and_store_result_for_post(post, click_house)
            prediction_ids.append(prediction_id)
        else:
            logger.info(f"Post {post.post_id} hasn't images and text.")

    logger.info("Sentiment analysis for wall complete.")
    result = click_house.prediction_summary(features, expand, prediction_ids)
    return result
