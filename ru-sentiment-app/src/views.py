"""Endpoints."""
from __future__ import annotations

import traceback
from typing import Optional

from fastapi import HTTPException
from loguru import logger
from requests import Request
from starlette.responses import JSONResponse

from src import app
from src.errors import NotAllowedException, NotFoundException, BadRequestException
from src.schemas import HealthCheck, PredictionResult


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
        tr = traceback.format_exception(
            etype=type(exc), value=exc, tb=exc.__traceback__
        )
        logger.error(" ".join(tr))
        return JSONResponse(status_code=500, content=content)
    return None


@app.get("/healthcheck")
def healthcheck() -> HealthCheck:
    """Healthcheck."""
    return HealthCheck(service=True)


@app.post("/common/predict")
def get_prediction_by_text_and_images() -> PredictionResult:
    """Get prediction by text and images."""
    return None
