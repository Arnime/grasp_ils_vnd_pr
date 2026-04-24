# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT
"""Pure-utility helpers shared across the ``givp._core`` submodules.

This module deliberately has zero internal dependencies (other than ``numpy``)
so it can be imported from every other ``_core`` submodule without risking
circular imports.
"""

from __future__ import annotations

import logging
import secrets
import time as _time_mod
from collections.abc import Callable
from contextvars import ContextVar

import numpy as np

# Type alias for the user-supplied objective function.
EvaluatorFn = Callable[[np.ndarray], float]

logger = logging.getLogger("givp._core")
_VERBOSE_HANDLER_ATTACHED: list[bool] = [False]


def _ensure_verbose_handler() -> None:
    """Attach a stdout handler to the ``givp._core`` logger so verbose=True
    actually prints to the console even when the application has not
    configured ``logging`` itself.

    Idempotent: safe to call repeatedly; only the first call adds a handler.
    """
    if _VERBOSE_HANDLER_ATTACHED[0]:
        logger.setLevel(logging.INFO)
        return
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("[givp] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    _VERBOSE_HANDLER_ATTACHED[0] = True


# Per-context runtime configuration shared across helpers.
#
# Stored in :class:`contextvars.ContextVar` so concurrent calls (threads /
# asyncio tasks) cannot clobber each other's settings.
_INTEGER_SPLIT: ContextVar[int | None] = ContextVar("givp_integer_split", default=None)
_GROUP_SIZE: ContextVar[int | None] = ContextVar("givp_group_size", default=None)
# Master RNG used by ``_new_rng`` to spawn child generators when a seed has
# been pinned via :func:`_set_seed`. ``None`` means "use OS entropy", which
# is the legacy non-deterministic behaviour. We store a :class:`SeedSequence`
# (rather than a :class:`Generator`) so child seeds are produced via the
# numpy-recommended :meth:`SeedSequence.spawn` API, which is the canonical
# way to derive statistically independent streams from a single root seed.
_MASTER_SEED_SEQ: ContextVar[np.random.SeedSequence | None] = ContextVar(
    "givp_master_seed_seq", default=None
)


def _set_seed(seed: int | None) -> None:
    """Pin the master :class:`SeedSequence` used to spawn per-call generators.

    Pass ``None`` to restore the default non-deterministic behaviour. When
    a seed is pinned, every subsequent :func:`_new_rng` call (without an
    explicit seed) gets a deterministic, statistically independent child
    seed via :meth:`SeedSequence.spawn`.
    """
    if seed is None:
        _MASTER_SEED_SEQ.set(None)
    else:
        _MASTER_SEED_SEQ.set(np.random.SeedSequence(seed))


def _get_half(n: int) -> int:
    """Return the index where integer variables begin, given vector length ``n``."""
    split = _INTEGER_SPLIT.get()
    if split is not None and 0 <= split <= n:
        return split
    return n // 2


def _set_integer_split(split: int | None) -> None:
    """Set the integer split used by the helpers below (per ContextVar)."""
    _INTEGER_SPLIT.set(split)


def _set_group_size(size: int | None) -> None:
    """Set the number of steps per group for the group/block neighbourhoods."""
    _GROUP_SIZE.set(size)


def _get_group_size() -> int | None:
    """Return the configured group size, if any."""
    return _GROUP_SIZE.get()


def _new_rng(seed: int | None = None) -> np.random.Generator:
    """Create a RNG using an explicit seed to satisfy static-analysis rules.

    When a master seed has been pinned via :func:`_set_seed`, an independent
    child :class:`SeedSequence` is spawned from the master and used to seed
    the new generator. Otherwise OS entropy is used.
    """
    if seed is not None:
        return np.random.default_rng(seed)
    master = _MASTER_SEED_SEQ.get()
    if master is not None:
        # ``spawn(1)`` mutates the master to advance its internal counter,
        # so successive calls return statistically independent streams.
        (child,) = master.spawn(1)
        return np.random.default_rng(child)
    return np.random.default_rng(secrets.randbits(64))


def _expired(deadline: float) -> bool:
    """Retorna True se o deadline foi atingido (0 = sem limite).

    Uses :func:`time.monotonic` so suspended/sleeping systems do not skew
    the deadline.
    """
    return deadline > 0 and _time_mod.monotonic() >= deadline


def _safe_evaluate(evaluator: EvaluatorFn, candidate: np.ndarray) -> float:
    """Call the user evaluator and coerce the result to a finite float.

    Returns ``np.inf`` on any failure (treated as an infeasible candidate).
    Logs a warning with traceback so silent bugs in the evaluator are visible.
    """
    try:
        cost = float(evaluator(candidate))
    except Exception:  # pylint: disable=broad-except
        logger.warning(
            "evaluator raised an exception; treating candidate as infeasible",
            exc_info=True,
        )
        return np.inf
    if not np.isfinite(cost):
        return np.inf
    return cost
