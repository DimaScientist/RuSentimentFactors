"""Enums for service."""
from __future__ import annotations

import os
import pickle
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from loguru import logger

from models import (
    visual_sentiment_model,
    text_sentiment_model,
)

if TYPE_CHECKING:
    from typing import Any, List


SERIALIZED_MODELS_PATH = Path(os.getcwd()) / "models" / "serialized"


class Sentiment(Enum):
    """Sentiment enum."""

    negative = 0
    neutral = 1
    positive = 2


class Tables(Enum):
    """Tables enum."""

    prediction = "prediction"
    image = "image"
    prediction_details = "prediction_details"

    @classmethod
    def __get_prediction_table_script(cls) -> str:
        """Get prediction table script."""
        script = """
        create table if not exists default.prediction
        (
            id             UUID comment 'Primary Key',
            post_id Nullable(String) comment 'Post ID in social media',
            post_url Nullable(String) comment 'Post URL in social media',
            text Nullable(String) comment 'Text in post',
            clean_text Nullable(String) comment 'Clean post text',
            predicted_value String comment 'Predicted sentiment value',
            labeled_value Nullable(String) comment 'Human labeled value',
            text_prediction_details_id Nullable(UUID) comment 'Text prediction details'
        )
         engine = Memory;
        """
        return script

    @classmethod
    def __get_image_table_script(cls) -> str:
        """Get image table script."""
        script = """
        create table if not exists default.image
        (
            id            UUID comment 'Primary Key',
            image_url Nullable(String) comment 'Image URL in social media',
            bucket        String comment 'MinIO bucket',
            key           String comment 'MinIO key',
            caption Nullable(String) comment 'Image caption',
            prediction_id UUID comment 'Prediction Foreign Key caption',
            prediction_details_id Nullable(UUID) comment 'Prediction details',
            filename String comment 'Image file name'
        )
            engine = Memory;
        """
        return script

    @classmethod
    def __get_prediction_details_table_script(cls) -> str:
        """Fet prediction details table script."""
        script = """
        create table if not exists default.prediction_details
        (
            id       UUID comment 'Primary Key',
            negative Float64 comment 'Negative class probability',
            neutral  Float64 comment 'Neutral class probability',
            positive Float64 comment 'Positive class probability'
        )
            engine = Memory;
        """
        return script

    def get_creation_table_script(self) -> str:
        """Create table."""
        if self == self.prediction:
            return self.__get_prediction_table_script()
        elif self == self.image:
            return self.__get_image_table_script()
        elif self == self.prediction_details:
            return self.__get_prediction_details_table_script()

    @classmethod
    def get_list(cls) -> List[str]:
        """Get ClickHouse tables."""
        return [cls.image.name, cls.prediction.name, cls.prediction_details.name]


class Models(Enum):
    """Enum for models."""

    label_encoder = "label_encoder.pkl"
    sentiment_text_model = "sentiment_text_model.cfg"
    sentiment_visual_model = "sentiment_visual_model.cfg"

    def load(self) -> Any:
        """Load model."""
        from config import configurations

        logger.info(f"Download model: {self.name}.")

        path = SERIALIZED_MODELS_PATH / self.value

        if self == self.sentiment_visual_model:
            state_dict = torch.load(path)
            model = visual_sentiment_model
            model.load_state_dict(state_dict)
        elif self == self.sentiment_text_model:
            state_dict = torch.load(path)
            model = text_sentiment_model
            model.load_state_dict(state_dict)
        elif self == self.label_encoder:
            with open(path, "rb") as file:
                model = pickle.load(file)
        else:
            raise ValueError(
                f"Model settings file extension should be "
                f"{configurations.PYTORCH_MODEL_FORMAT} (PyTorch save) "
                f"or {configurations.PICKLE_MODEL_FORMAT} (pickle)."
            )

        logger.info(f"{self.name} downloaded.")
        return model
