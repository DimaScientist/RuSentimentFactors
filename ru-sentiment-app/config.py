"""Configurations for RuSentimentFactors service."""
from pydantic import BaseSettings, ValidationError


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

    DEBUG: bool = False
    VK_TOKEN: str = ""
    HOST: str = "0.0.0.0"
    PORT: int = 5000
    ROOT_PATH: str = ""
