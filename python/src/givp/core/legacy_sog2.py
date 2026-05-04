# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT
"""Deprecated compatibility shim for legacy SOG2 helpers."""

from __future__ import annotations

import warnings

from givp.legacy.sog2 import evaluate_candidates

warnings.warn(
    "givp.core.legacy_sog2 is deprecated and has moved to givp.legacy.sog2. "
    "Import from givp.legacy.sog2 to silence this warning.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["evaluate_candidates"]
