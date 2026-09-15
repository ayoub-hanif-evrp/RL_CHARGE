"""Errors raised by the physical model. Missing scientific values fail loudly."""

from data.errors import EVRPTWGRError


class PhysicsError(EVRPTWGRError):
    """Base class for physical-model failures."""


class UnknownPhysicsProfileError(PhysicsError):
    """The named physics profile does not exist on disk."""


class MissingPhysicsParameterError(PhysicsError):
    """A required physical parameter is absent or has unknown units."""


class InvalidPhysicsParameterError(PhysicsError):
    """A physical parameter is present but scientifically unusable."""
