"""Enums for service."""
from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import List


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
            int             UInt256 comment 'Primary Key',
            post_id Nullable(String) comment 'Post ID in social media',
            post_url Nullable(String) comment 'Post URL in social media',
            text Nullable(String) comment 'Text in post',
            clean_text Nullable(String) comment 'Clean post text',
            predicted_value String comment 'Predicted sentiment value',
            labeled_value Nullable(String) comment 'Human labeled value',
            text_prediction_details_id Nullable(UInt256) comment 'Text prediction details'
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
            id            UInt256 comment 'Primary Key',
            image_url Nullable(String) comment 'Image URL in social media',
            bucket        String comment 'MinIO bucket',
            key           String comment 'MinIO key',
            caption Nullable(String) comment 'Image caption',
            prediction_id UInt256 comment 'Prediction Foreign Key caption',
            prediction_details_id Nullable(UInt256) comment 'Prediction details'
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
            id       UInt256 comment 'Primary Key',
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
