"""Configurations for RuSentimentFactors service."""
import os
from typing import Any, Dict, Optional

import clickhouse_connect
from minio import Minio
from pathlib import Path
from pydantic import BaseSettings, ValidationError, root_validator, validator

import torch

from loguru import logger

from src.enums import Tables


class BaseConfig(BaseSettings):
    """Config that gets options from environment variables."""

    def __init__(self) -> None:  # pragma: no cover
        """Override __init__ from pydantic.BaseSettings to add more verbosity to 'value_error.missing' errors.

        Call init from pydantic.BaseSettings, catch all errors and if it was 'missing value' error then reraise it
        with more readable error message. Any other exception will be reraised as is.

        :raises AssertionError: can not find environment variable
        :raises ValidationError: any other pydantic's validation error
        """
        try:
            super().__init__()
        except ValidationError as validation_error:
            for error in validation_error.errors():
                if error["type"] == "value_error.missing":
                    raise AssertionError(
                        f"Set '{error['loc'][0]}' environment variable."
                    )
            raise validation_error


class Config(BaseConfig):
    """Configuration for service."""

    DEBUG: bool = False
    VK_TOKEN: str = ""
    HOST: str = "0.0.0.0"
    PORT: int = 5000
    ROOT_PATH: str = ""

    """MINIO CONFIGURATIONS."""
    MINIO_BUCKET: str = "images"
    MINIO_HOST: str
    MINIO_ROOT_USER: str
    MINIO_ROOT_PASSWORD: str

    """CLICKHOUSE CONFIGURATIONS"""
    CLICKHOUSE_HOST: str
    CLICKHOUSE_PORT: int
    CLICKHOUSE_ROOT_USER: str
    CLICKHOUSE_ROOT_PASSWORD: Optional[str] = None

    """MODELS CONFIGURATIONS."""
    BERT_TOKENIZER_CONFIG_PATH: Path = (
            Path(os.getcwd()) / "models" / "bert_tokenizer.cfg"
    )
    VISUAL_FEATURE_EXTRACTOR_CONFIG_PATH: Path = (
            Path(os.getcwd()) / "models" / "visual_feature_extractor.cfg"
    )
    LABEL_ENCODER_PATH: Path = Path(os.getcwd()) / "models" / "label_encoder.pkl"
    SENTIMENT_TEXT_MODEL_CONFIG_PATH: Path = (
            Path(os.getcwd()) / "models" / "sentiment_text_model.cfg"
    )
    SENTIMENT_VISUAL_MODEL_CONFIG_PATH: Path = (
            Path(os.getcwd()) / "models" / "sentiment_visual_model.cfg"
    )
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    IMAGE_CAPTION_GENERATION_MODEL: str = "tuman/vit-rugpt2-image-captioning"

    @validator("BERT_TOKENIZER_CONFIG_PATH")
    def check_bert_tokenizer_config_on_existing(cls, path_to_tokenizer: Path) -> Path:
        """Check on BERT tokenizer existing."""
        if not os.path.exists(path_to_tokenizer):
            raise ValueError("BERT_TOKENIZER_CONFIG_PATH is not exists.")
        return path_to_tokenizer

    @validator("VISUAL_FEATURE_EXTRACTOR_CONFIG_PATH")
    def check_visual_feature_extractor_config_on_existing(
            cls,
            path_to_visual_feature_extractor: Path,
    ) -> Path:
        """Check on Visual feature extractor existing."""
        if not os.path.exists(path_to_visual_feature_extractor):
            raise ValueError("VISUAL_FEATURE_EXTRACTOR_CONFIG_PATH is not exists.")
        return path_to_visual_feature_extractor

    @validator("LABEL_ENCODER_PATH")
    def check_label_encoder_config_on_existing(
            cls,
            path_to_label_encoder: Path,
    ) -> Path:
        """Check on label encoder existing."""
        if not os.path.exists(path_to_label_encoder):
            raise ValueError("LABEL_ENCODER_PATH is not exists.")
        return path_to_label_encoder

    @validator("SENTIMENT_TEXT_MODEL_CONFIG_PATH")
    def check_bert_model_config_existing(cls, path_to_bert_model: Path) -> Path:
        """Check on BERT model existing."""
        if not os.path.exists(path_to_bert_model):
            raise ValueError("SENTIMENT_TEXT_MODEL_CONFIG_PATH is not exists.")
        return path_to_bert_model

    @validator("SENTIMENT_VISUAL_MODEL_CONFIG_PATH")
    def check_visual_multimodal_config_existing(
            cls,
            path_to_visual_multimodal_model: Path,
    ) -> Path:
        """Check on multimodal model existing."""
        if not os.path.exists(path_to_visual_multimodal_model):
            raise ValueError("SENTIMENT_VISUAL_MODEL_CONFIG_PATH is not exists.")
        return path_to_visual_multimodal_model

    @root_validator
    def check_and_create_necessary_objects(
            cls,
            values: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Check and create bucket, tables and other."""
        # Check Minio buckets.
        minio = Minio(
            values.get("MINIO_HOST"),
            values.get("MINIO_ROOT_USER"),
            values.get("MINIO_ROOT_PASSWORD"),
            secure=False,
        )
        if not minio.bucket_exists(values.get("MINIO_BUCKET")):
            minio.make_bucket(values.get("MINIO_BUCKET"))
            logger.info(f"Create bucket: {values.get('MINIO_BUCKET')}")

        # Check ClickHouse tables
        clickhouse_settings = dict(
            host=values.get("CLICKHOUSE_HOST"),
            port=values.get("CLICKHOUSE_PORT"),
            username=values.get("CLICKHOUSE_ROOT_USER"),
        )
        if values.get("CLICKHOUSE_ROOT_PASSWORD"):
            clickhouse_settings["password"] = values.get("CLICKHOUSE_ROOT_PASSWORD")

        clickhouse_client = clickhouse_connect.get_client(**clickhouse_settings)
        tables = clickhouse_client.command("SHOW TABLES").split("\n")
        for table in Tables:
            if table.name not in tables:
                clickhouse_client.command(table.get_creation_table_script())
                logger.info(f"Create table: {table.name}")

        return values
