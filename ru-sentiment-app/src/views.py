from __future__ import annotations

from src import app
from src.schemas import HealthCheck


@app.get("/healthcheck")
def healthcheck() -> HealthCheck:
    """Healthcheck."""
    return HealthCheck(service=True)
