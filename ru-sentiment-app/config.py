"""Configurations for RuSentimentFactors service."""
import os
from typing import Any, Dict, Optional

import clickhouse_connect
from dotenv import load_dotenv
from loguru import logger
from minio import Minio
from pydantic import BaseSettings, ValidationError, root_validator

from src.enums import Tables

if os.environ.get("LOAD_ENV") and bool(os.environ.get("LOAD_ENV")):
    load_dotenv(".env")


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
                    raise AssertionError(f"Set '{error['loc'][0]}' environment variable.")
            raise validation_error


class Config(BaseConfig):
    """Configuration for service."""

    DEBUG: bool = True
    VK_TOKEN: str
    HOST: str = "0.0.0.0"
    PORT: int = 5000
    ROOT_PATH: str = ""

    """MINIO CONFIGURATIONS."""
    MINIO_BUCKET: str = "images"
    MINIO_HOST: str
    MINIO_ROOT_USER: str
    MINIO_ROOT_PASSWORD: str

    """CLICKHOUSE CONFIGURATIONS."""
    CLICKHOUSE_HOST: str
    CLICKHOUSE_PORT: int
    CLICKHOUSE_USER: str
    CLICKHOUSE_PASSWORD: Optional[str] = None

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
            username=values.get("CLICKHOUSE_USER"),
        )
        if values.get("CLICKHOUSE_PASSWORD"):
            clickhouse_settings["password"] = values.get("CLICKHOUSE_PASSWORD")

        clickhouse_client = clickhouse_connect.get_client(**clickhouse_settings)
        tables = clickhouse_client.command("SHOW TABLES").split("\n")
        for table in Tables:
            if table.name not in tables:
                clickhouse_client.command(table.get_creation_table_script())
                logger.info(f"Create table: {table.name}")

        return values


configurations = Config()
