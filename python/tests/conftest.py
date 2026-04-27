# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT
"""Pytest fixtures for benchmark functions used in examples and tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from givp import benchmarks

from tests.fixtures import problems as _problems


@pytest.fixture
def sphere():
    """Return the classic Sphere benchmark callable."""
    return benchmarks.sphere


@pytest.fixture
def sphere_bounds_1d():
    """Return a typical 1D bound for the Sphere function."""
    return [(-5.12, 5.12)]


@pytest.fixture
def sphere_bounds_4d():
    """Return a typical 4D bound set for the Sphere function."""
    return [(-5.12, 5.12)] * 4


@pytest.fixture
def sphere_file():
    """Path to the example sphere function file in `tests/fixtures`."""
    return str(Path(__file__).parent / "fixtures" / "sphere.py")


@pytest.fixture
def knapsack_values():
    """Return knapsack values from the fixtures module."""
    return list(_problems.KNAP_VALUES)


@pytest.fixture
def knapsack_weights():
    """Return knapsack weights from the fixtures module."""
    return list(_problems.KNAP_WEIGHTS)


@pytest.fixture
def knapsack_capacity():
    """Return knapsack capacity from the fixtures module."""
    return int(_problems.KNAP_CAPACITY)


@pytest.fixture
def qap_flow():
    """Return the flow matrix for a small QAP instance."""
    return _problems.QAP_FLOW.copy()


@pytest.fixture
def qap_dist():
    """Return the distance matrix for a small QAP instance."""
    return _problems.QAP_DIST.copy()
