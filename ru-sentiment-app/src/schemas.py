"""PyDantic Schemas for service."""
from pydantic import BaseModel


class HealthCheck(BaseModel):
    """HealthCheck schema."""
    service: bool
