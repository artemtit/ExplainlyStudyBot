from __future__ import annotations


class DomainError(Exception):
    """Base class for domain-level errors."""


class MaterialValidationError(DomainError):
    """Raised when material payload cannot be validated."""
