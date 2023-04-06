"""Endpoints."""
from __future__ import annotations

import traceback
from typing import List, Optional

from uuid import UUID
from fastapi import Depends, Form, HTTPException, UploadFile, Query
from loguru import logger
from requests import Request
from starlette.responses import JSONResponse

from src import app, services
from src.clickhouse_client import ClickHouse, get_clickhouse
from src.errors import NotAllowedException, NotFoundException, BadRequestException
from src.minio_client import Minio, get_minio
from src.schemas import DetailedPredictionData, HealthCheck, PredictionResult, Summary


@app.exception_handler(NotAllowedException)
def unhandled_permission_denied_exception(
    request: Request,
    exc: NotAllowedException,
) -> JSONResponse:
    """Return json response for raised PermissionDeniedException."""
    return JSONResponse(status_code=403, content={"detail": exc.message})


@app.exception_handler(NotFoundException)
def unhandled_not_found_exception(
    request: Request,
    exc: NotFoundException,
) -> JSONResponse:
    """Return json response for raised NotFoundException."""
    return JSONResponse(status_code=404, content={"detail": exc.message})


@app.exception_handler(BadRequestException)
def unhandled_bad_request_exception(
    request: Request,
    exc: BadRequestException,
) -> JSONResponse:
    """Return json response for raised BadRequestException."""
    return JSONResponse(status_code=400, content={"detail": exc.message})


@app.exception_handler(Exception)
def unhandled_internal_error(
    request: Request,
    exc: Exception,
) -> Optional[JSONResponse]:
    """Return json response for raised exceptions."""
    if not isinstance(exc, HTTPException):
        logger.error(exc)
        content = {"detail": "Internal error."}
        tr = traceback.format_exception(etype=type(exc), value=exc, tb=exc.__traceback__)
        logger.error(" ".join(tr))
        return JSONResponse(status_code=500, content=content)
    return None


@app.get("/healthcheck")
def healthcheck() -> HealthCheck:
    """Healthcheck."""
    return HealthCheck(service=True)


@app.post("/common/prediction")
def get_prediction_by_text_and_images(
    store: bool = False,
    images: Optional[List[UploadFile]] = None,
    text: Optional[str] = Form(default=None),
    click_house: ClickHouse = Depends(get_clickhouse),
    minio: Minio = Depends(get_minio),
) -> PredictionResult:
    """Get prediction by text and images."""
    prediction_result = services.get_prediction_by_text_and_images(store, text, images, click_house, minio)
    return prediction_result


@app.get("/common/prediction/{prediction_id}")
def get_prediction_details(
    prediction_id: UUID,
    click_house: ClickHouse = Depends(get_clickhouse),
) -> DetailedPredictionData:
    """Get prediction details."""
    return services.get_prediction_details(prediction_id, click_house)


@app.put("/common/prediction/{prediction_id}")
def set_labeled_prediction(
    prediction_id: UUID,
    sentiment: str = Form(...),
    click_house: ClickHouse = Depends(get_clickhouse),
) -> None:
    """Set human label to prediction."""
    return services.set_labeled_sentiment_to_prediction(
        prediction_id,
        sentiment,
        click_house,
    )


@app.get("/common/summary")
def get_prediction_summary_result(
    features: Optional[bool] = Query(None),
    expand: Optional[bool] = Query(None),
    click_house: ClickHouse = Depends(get_clickhouse),
) -> Summary:
    """Get sentiment statistic by data."""
    summary_result = services.get_prediction_summary(features, expand, click_house)
    return summary_result
