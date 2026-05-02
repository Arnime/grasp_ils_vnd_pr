# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT
"""Small problem instances and data used by tests fixtures.

This module centralizes small, canonical instances for knapsack and QAP
used by the benchmark tests.
"""

import numpy as np

# Knapsack small instance
KNAP_VALUES = [60, 100, 120]
KNAP_WEIGHTS = [10, 20, 30]
KNAP_CAPACITY = 50

# QAP small instance
QAP_FLOW = np.array([[0, 5], [5, 0]], dtype=float)
QAP_DIST = np.array([[0, 1], [1, 0]], dtype=float)
