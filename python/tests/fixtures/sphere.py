"""Simple sphere function fixture for CLI integration tests."""

import numpy as np


def sphere(x: np.ndarray) -> float:
    return float(np.sum(x**2))


def neg_sphere(x: np.ndarray) -> float:
    return -float(np.sum(x**2))
