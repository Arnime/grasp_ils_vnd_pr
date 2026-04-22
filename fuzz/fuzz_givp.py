"""Fuzz tests for the givp public API using atheris.

Run with atheris (requires a separate install):
    python fuzz/fuzz_givp.py -atheris_runs=10000

Or via ClusterFuzzLite / OSS-Fuzz:
    https://google.github.io/clusterfuzzlite/
"""

from __future__ import annotations

import sys

import atheris
import numpy as np

from givp import GraspIlsVndConfig, grasp_ils_vnd_pr

_FAST_CFG = GraspIlsVndConfig(
    max_iterations=2,
    vnd_iterations=3,
    ils_iterations=1,
    early_stop_threshold=2,
    use_convergence_monitor=False,
)


def _sphere(x: np.ndarray) -> float:
    return float(np.sum(x**2))


@atheris.instrument_func
def TestOneInput(data: bytes) -> None:
    fdp = atheris.FuzzedDataProvider(data)

    # Derive dimensionality (1-6) and bounds from fuzz bytes
    ndim = fdp.ConsumeIntInRange(1, 6)
    bounds = []
    for _ in range(ndim):
        lo = fdp.ConsumeFloat()
        width = abs(fdp.ConsumeFloat())
        if not (np.isfinite(lo) and np.isfinite(width) and width >= 1e-4):
            return
        bounds.append((lo, lo + width))

    direction = fdp.PickValueInList(["minimize", "maximize"])

    try:
        result = grasp_ils_vnd_pr(
            _sphere,
            bounds,
            direction=direction,
            config=_FAST_CFG,
        )
        assert result.x.shape == (ndim,)
        assert np.isfinite(result.fun)
    except Exception:
        # Unexpected exception — re-raise so atheris records the crash
        raise


def main() -> None:
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
