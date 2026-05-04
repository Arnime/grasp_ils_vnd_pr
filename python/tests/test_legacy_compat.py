# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT
"""Backward-compatibility tests for the legacy SOG2 import path."""

from __future__ import annotations

import importlib
import warnings


def test_legacy_module_import_path_emits_deprecation_warning() -> None:
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", DeprecationWarning)
        module = importlib.import_module("givp.core.legacy_sog2")

    assert hasattr(module, "evaluate_candidates")
    assert any("moved to givp.legacy.sog2" in str(item.message) for item in captured)


def test_core_re_export_points_to_new_legacy_namespace() -> None:
    from givp.core import evaluate_candidates as from_core
    from givp.legacy.sog2 import evaluate_candidates as from_new_namespace

    assert from_core is from_new_namespace
