"""Custom exceptions for the givp package.

All givp-specific errors derive from :class:`GivpError`, which itself inherits
from :class:`Exception`.  Each concrete subclass also inherits from a relevant
built-in exception (``ValueError`` / ``RuntimeError``) so existing
``except ValueError:`` clauses in user code keep working.
"""

from __future__ import annotations


class GivpError(Exception):
    """Base class for all givp errors."""


class InvalidBoundsError(GivpError, ValueError):
    """Raised when the lower/upper bound vectors are inconsistent."""


class InvalidInitialGuessError(GivpError, ValueError):
    """Raised when the initial guess is incompatible with the bounds."""


class InvalidConfigError(GivpError, ValueError):
    """Raised when a :class:`GraspIlsVndConfig` field has an invalid value."""


class EvaluatorError(GivpError, RuntimeError):
    """Raised when the user-supplied evaluator misbehaves in a fatal way."""


class EmptyPoolError(GivpError, RuntimeError):
    """Raised when an operation is attempted on an empty elite pool."""


__all__ = [
    "EmptyPoolError",
    "EvaluatorError",
    "GivpError",
    "InvalidBoundsError",
    "InvalidConfigError",
    "InvalidInitialGuessError",
]
