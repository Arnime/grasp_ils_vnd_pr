# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT
"""Fuzz tests for the givp public API.

Two backends are supported:
- **atheris** (Linux only, coverage-guided): used in CI / OSS-Fuzz.
- **hypothesis** (Windows / macOS / Linux): used for local development.

Run with atheris on Linux:
    pip install atheris
    python fuzz/fuzz_givp.py -atheris_runs=10000

Run with hypothesis on any platform (hypothesis is a dev dependency):
    python fuzz/fuzz_givp.py --hypothesis

Or via ClusterFuzzLite / OSS-Fuzz:
    https://google.github.io/clusterfuzzlite/
"""

from __future__ import annotations

import struct
import sys

try:
    import atheris  # type: ignore[import-untyped]

    HAS_ATHERIS = True
except ImportError:
    atheris = None  # type: ignore[assignment]
    HAS_ATHERIS = False

import numpy as np
from givp import GIVPConfig, givp
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

_FAST_CFG = GIVPConfig(
    max_iterations=2,
    vnd_iterations=3,
    ils_iterations=1,
    early_stop_threshold=2,
    use_convergence_monitor=False,
)


def _sphere(x: np.ndarray) -> float:
    return float(np.sum(x**2))


def _fuzz_bytes(data: bytes) -> None:
    """Core fuzz logic shared by both pythonfuzz and atheris targets."""
    if len(data) < 4:
        return

    # First byte encodes ndim (1-6) and direction
    ndim = max(1, min(6, data[0] % 6 + 1))
    direction = "maximize" if data[1] % 2 else "minimize"
    payload = data[2:]

    # Each dimension needs 2 x float64 = 16 bytes
    if len(payload) < ndim * 16:
        return

    bounds = []
    for i in range(ndim):
        chunk = payload[i * 16 : i * 16 + 16]
        lo, width = struct.unpack("dd", chunk)
        if not (np.isfinite(lo) and np.isfinite(width) and abs(width) >= 1e-4):
            return
        bounds.append((lo, lo + abs(width)))

    result = givp(
        _sphere,
        bounds,
        direction=direction,
        config=_FAST_CFG,
    )
    assert result.x.shape == (ndim,)
    assert np.isfinite(result.fun)


# ---------------------------------------------------------------------------
# hypothesis entry point (cross-platform: Windows / macOS / Linux)
# ---------------------------------------------------------------------------

_bounds_st = st.lists(
    st.tuples(
        st.floats(
            min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False
        ),
        st.floats(
            min_value=1e-3, max_value=10.0, allow_nan=False, allow_infinity=False
        ),
    ),
    min_size=1,
    max_size=6,
).map(lambda pairs: [(lo, lo + w) for lo, w in pairs])


@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(bounds=_bounds_st, direction=st.sampled_from(["minimize", "maximize"]))
def fuzz_with_hypothesis(bounds: list[tuple[float, float]], direction: str) -> None:
    """Hypothesis-based fuzz target: works on Windows, macOS and Linux."""
    result = givp(
        _sphere,
        bounds,
        direction=direction,
        config=_FAST_CFG,
    )
    assert result.x.shape == (len(bounds),)
    assert np.isfinite(result.fun)


# ---------------------------------------------------------------------------
# atheris entry point (Linux / OSS-Fuzz)
# ---------------------------------------------------------------------------


def _get_atheris_func():
    """Return the instrumented atheris fuzz target (Linux only)."""
    if not HAS_ATHERIS:
        return None

    @atheris.instrument_func  # type: ignore[attr-defined]
    def fuzz_atheris(data: bytes) -> None:
        """atheris entry point: drive givp with arbitrary bytes."""
        _fuzz_bytes(data)

    return fuzz_atheris


def main() -> None:
    """Launch the appropriate fuzzer.

    - ``--hypothesis``: run the hypothesis-based target (cross-platform).
    - default on Linux with atheris installed: run atheris.
    - default elsewhere: fall back to hypothesis.
    """
    if "--hypothesis" in sys.argv or not HAS_ATHERIS:
        sys.argv = [a for a in sys.argv if a != "--hypothesis"]
        fuzz_with_hypothesis()  # pylint: disable=no-value-for-parameter
    else:
        atheris.Setup(sys.argv, _get_atheris_func())  # type: ignore[attr-defined]
        atheris.Fuzz()  # type: ignore[attr-defined]


if __name__ == "__main__":
    main()
