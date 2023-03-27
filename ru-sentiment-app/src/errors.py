"""Internal exceptions for service."""

from __future__ import annotations


class InternalException(Exception):
    """Base class for internal exceptions."""

    def __init__(self, message: str) -> None:
        """Init method."""
        super().__init__()
        self.message = message


class NotFoundException(InternalException):
    """Raised when something can not be retrieved because does not exist."""

    pass


class NotAllowedException(InternalException):
    """Raised when user has no permissions to access something."""

    pass


class BadRequestException(InternalException):
    """Raised when something can not be performed for some reason."""

    pass
