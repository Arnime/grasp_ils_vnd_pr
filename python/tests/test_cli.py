# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT
"""Tests for the ``givp`` CLI (``givp run``)."""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

SPHERE_FILE = str(Path(__file__).parent / "fixtures" / "sphere.py")
BOUNDS_1D = "[[-5,5]]"
BOUNDS_4D = "[[-5,5],[-5,5],[-5,5],[-5,5]]"
FAST_CONFIG = json.dumps(
    {
        "max_iterations": 3,
        "vnd_iterations": 4,
        "ils_iterations": 2,
        "elite_size": 3,
        "path_relink_frequency": 2,
        "num_candidates_per_step": 4,
        "early_stop_threshold": 10,
        "use_convergence_monitor": False,
    }
)


def _run(
    *extra_args: str, stdin: str | None = None
) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, "-m", "givp.cli", *extra_args]
    return subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        input=stdin,
    )


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


def test_run_outputs_json_to_stdout():
    proc = _run(
        "run",
        "--func-file",
        SPHERE_FILE,
        "--func-name",
        "sphere",
        "--bounds",
        BOUNDS_1D,
        "--config",
        FAST_CONFIG,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert isinstance(data, dict)


def test_run_json_schema_fields():
    proc = _run(
        "run",
        "--func-file",
        SPHERE_FILE,
        "--func-name",
        "sphere",
        "--bounds",
        BOUNDS_1D,
        "--config",
        FAST_CONFIG,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert set(data.keys()) == {
        "x",
        "fun",
        "nit",
        "nfev",
        "success",
        "termination",
        "direction",
    }
    assert isinstance(data["x"], list)
    assert isinstance(data["fun"], float)
    assert isinstance(data["nit"], int)
    assert isinstance(data["nfev"], int)
    assert isinstance(data["success"], bool)
    assert isinstance(data["termination"], str)
    assert isinstance(data["direction"], str)


def test_run_termination_is_closed_enum():
    from givp.result import TerminationReason

    valid_values = {r.value for r in TerminationReason}
    proc = _run(
        "run",
        "--func-file",
        SPHERE_FILE,
        "--func-name",
        "sphere",
        "--bounds",
        BOUNDS_1D,
        "--config",
        FAST_CONFIG,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["termination"] in valid_values


def test_run_direction_minimize():
    proc = _run(
        "run",
        "--func-file",
        SPHERE_FILE,
        "--func-name",
        "sphere",
        "--bounds",
        BOUNDS_1D,
        "--config",
        FAST_CONFIG,
        "--direction",
        "minimize",
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["direction"] == "minimize"


def test_run_direction_maximize():
    proc = _run(
        "run",
        "--func-file",
        SPHERE_FILE,
        "--func-name",
        "neg_sphere",
        "--bounds",
        BOUNDS_4D,
        "--config",
        FAST_CONFIG,
        "--direction",
        "maximize",
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["direction"] == "maximize"


def test_run_4d_bounds_produces_correct_x_shape():
    proc = _run(
        "run",
        "--func-file",
        SPHERE_FILE,
        "--func-name",
        "sphere",
        "--bounds",
        BOUNDS_4D,
        "--config",
        FAST_CONFIG,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert len(data["x"]) == 4


def test_run_seed_produces_output():
    proc = _run(
        "run",
        "--func-file",
        SPHERE_FILE,
        "--func-name",
        "sphere",
        "--bounds",
        BOUNDS_1D,
        "--config",
        FAST_CONFIG,
        "--seed",
        "42",
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert "x" in data


def test_run_via_json_flag():
    payload = json.dumps(
        {
            "func_file": SPHERE_FILE,
            "func_name": "sphere",
            "bounds": [[-5, 5]],
            "config": json.loads(FAST_CONFIG),
        }
    )
    proc = _run("run", "--json", payload)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert "x" in data


def test_run_via_json_stdin():
    payload = json.dumps(
        {
            "func_file": SPHERE_FILE,
            "func_name": "sphere",
            "bounds": [[-5, 5]],
            "config": json.loads(FAST_CONFIG),
        }
    )
    proc = _run("run", "--json", "-", stdin=payload)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert "x" in data


def test_run_explicit_flags_override_json():
    """--bounds on the command line should take precedence over --json contents."""
    payload = json.dumps(
        {
            "func_file": SPHERE_FILE,
            "func_name": "sphere",
            "bounds": [[-5, 5], [-5, 5]],  # 2D in JSON
            "config": json.loads(FAST_CONFIG),
        }
    )
    proc = _run("run", "--json", payload, "--bounds", BOUNDS_1D)  # 1D override
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert len(data["x"]) == 1  # CLI flag wins


def test_errors_go_to_stderr():
    proc = _run(
        "run",
        "--func-file",
        "nonexistent.py",
        "--func-name",
        "sphere",
        "--bounds",
        BOUNDS_1D,
    )
    assert proc.returncode != 0
    assert "error" in proc.stderr.lower()
    assert proc.stdout == ""


def test_missing_func_name_error():
    proc = _run(
        "run",
        "--func-file",
        SPHERE_FILE,
        "--func-name",
        "no_such_function",
        "--bounds",
        BOUNDS_1D,
        "--config",
        FAST_CONFIG,
    )
    assert proc.returncode != 0
    assert proc.stdout == ""


def test_missing_required_args_gives_exit_2():
    # No func-file/func-name/bounds and no --json
    proc = _run("run")
    assert proc.returncode == 2


def test_invalid_bounds_json_gives_exit_2():
    proc = _run(
        "run",
        "--func-file",
        SPHERE_FILE,
        "--func-name",
        "sphere",
        "--bounds",
        "not-json",
        "--config",
        FAST_CONFIG,
    )
    assert proc.returncode != 0


# ---------------------------------------------------------------------------
# Unit tests for _result helpers
# ---------------------------------------------------------------------------


def test_termination_reason_from_message_converged():
    from givp.result import TerminationReason

    assert (
        TerminationReason.from_message("converged after 10 iter")
        == TerminationReason.CONVERGED
    )


def test_termination_reason_from_message_max_iterations():
    from givp.result import TerminationReason

    assert (
        TerminationReason.from_message("reached max iterations")
        == TerminationReason.MAX_ITERATIONS
    )


def test_termination_reason_from_message_time_limit():
    from givp.result import TerminationReason

    assert (
        TerminationReason.from_message("time limit exceeded")
        == TerminationReason.TIME_LIMIT
    )


def test_termination_reason_from_message_early_stop():
    from givp.result import TerminationReason

    assert (
        TerminationReason.from_message("early stop threshold reached")
        == TerminationReason.EARLY_STOP
    )


def test_termination_reason_from_message_unknown():
    from givp.result import TerminationReason

    assert (
        TerminationReason.from_message("something completely unrecognized")
        == TerminationReason.UNKNOWN
    )


def test_optimize_result_to_dict_types():
    import numpy as np
    from givp.result import OptimizeResult

    result = OptimizeResult(
        x=np.array([1.0, 2.0]),
        fun=5.0,
        nit=3,
        nfev=42,
        success=True,
        message="reached max iterations",
        direction="minimize",
    )
    d = result.to_dict()
    assert d["x"] == [1.0, 2.0]
    assert d["fun"] == pytest.approx(5.0)
    assert d["nit"] == 3
    assert d["nfev"] == 42
    assert d["success"] is True
    assert d["termination"] == "max_iterations"
    assert d["direction"] == "minimize"
    # Must be JSON-serializable
    json.dumps(d)


def test_optimize_result_to_dict_no_meta():
    """to_dict() must not leak the meta dict (internal details)."""
    import numpy as np
    from givp.result import OptimizeResult

    result = OptimizeResult(
        x=np.array([0.0]),
        fun=0.0,
        nit=1,
        nfev=10,
        success=True,
        message="",
        meta={"max_iterations": 99},
    )
    d = result.to_dict()
    assert "meta" not in d


def test_termination_reason_from_message_no_feasible():
    from givp.result import TerminationReason

    assert (
        TerminationReason.from_message("no feasible solution found")
        == TerminationReason.NO_FEASIBLE
    )
    assert (
        TerminationReason.from_message("no solution exists")
        == TerminationReason.NO_FEASIBLE
    )


# ---------------------------------------------------------------------------
# In-process unit tests for CLI helpers (needed for coverage tracking)
# ---------------------------------------------------------------------------


def test_parse_bounds_valid():
    from givp.cli import _parse_bounds

    bounds = _parse_bounds("[[-5,5],[0,10]]")
    assert bounds == [(-5.0, 5.0), (0.0, 10.0)]


def test_parse_bounds_invalid_json_raises():
    from givp.cli import _parse_bounds

    with pytest.raises(ValueError, match="valid JSON"):
        _parse_bounds("not-json")


def test_parse_bounds_not_list_raises():
    from givp.cli import _parse_bounds

    with pytest.raises(ValueError, match="JSON array"):
        _parse_bounds('{"a":1}')


def test_parse_bounds_bad_pair_raises():
    from givp.cli import _parse_bounds

    with pytest.raises(ValueError, match="\\[low, high\\]"):
        _parse_bounds("[[1,2,3]]")


def test_parse_config_none_returns_default():
    from givp.cli import _parse_config
    from givp.config import GIVPConfig

    cfg = _parse_config(None)
    assert isinstance(cfg, GIVPConfig)


def test_parse_config_valid_json():
    from givp.cli import _parse_config

    cfg = _parse_config('{"max_iterations": 10}')
    assert cfg.max_iterations == 10


def test_parse_config_invalid_json_raises():
    from givp.cli import _parse_config

    with pytest.raises(ValueError, match="valid JSON"):
        _parse_config("bad")


def test_parse_config_not_dict_raises():
    from givp.cli import _parse_config

    with pytest.raises(ValueError, match="JSON object"):
        _parse_config("[1,2,3]")


def test_load_func_valid():
    from givp.cli import _load_func

    func = _load_func(SPHERE_FILE, "sphere")
    import numpy as np

    assert func(np.array([3.0, 4.0])) == pytest.approx(25.0)


def test_load_func_file_not_found():
    from givp.cli import _load_func

    with pytest.raises(FileNotFoundError):
        _load_func("nonexistent.py", "sphere")


def test_load_func_attr_missing():
    from givp.cli import _load_func

    with pytest.raises(AttributeError, match="no_such_func"):
        _load_func(SPHERE_FILE, "no_such_func")


def test_resolve_args_json_merged():
    from givp.cli import _build_parser, _resolve_args

    parser = _build_parser()
    ns = parser.parse_args(
        [
            "run",
            "--json",
            json.dumps(
                {"func_file": SPHERE_FILE, "func_name": "sphere", "bounds": [[-5, 5]]}
            ),
        ]
    )
    args = _resolve_args(ns)
    assert args["func_file"] == SPHERE_FILE
    assert args["func_name"] == "sphere"


def test_resolve_args_explicit_overrides_json():
    from givp.cli import _build_parser, _resolve_args

    parser = _build_parser()
    ns = parser.parse_args(
        [
            "run",
            "--json",
            json.dumps(
                {"func_file": "wrong.py", "func_name": "sphere", "bounds": [[-5, 5]]}
            ),
            "--func-file",
            SPHERE_FILE,
        ]
    )
    args = _resolve_args(ns)
    assert args["func_file"] == SPHERE_FILE


def test_resolve_args_invalid_json_raises():
    from givp.cli import _build_parser, _resolve_args

    parser = _build_parser()
    ns = parser.parse_args(["run", "--json", "bad-json"])
    with pytest.raises(ValueError, match="valid JSON"):
        _resolve_args(ns)


def test_cmd_run_missing_args_returns_2():
    from givp.cli import _build_parser, _cmd_run

    parser = _build_parser()
    ns = parser.parse_args(["run", "--json", "{}"])
    code = _cmd_run(ns)
    assert code == 2


def test_cmd_run_invalid_json_in_json_flag_returns_2():
    from givp.cli import _build_parser, _cmd_run

    parser = _build_parser()
    ns = parser.parse_args(["run", "--json", "not-json"])
    code = _cmd_run(ns)
    assert code == 2


def test_cmd_run_success_returns_0(capsys):
    from givp.cli import _build_parser, _cmd_run

    parser = _build_parser()
    ns = parser.parse_args(
        [
            "run",
            "--func-file",
            SPHERE_FILE,
            "--func-name",
            "sphere",
            "--bounds",
            BOUNDS_1D,
            "--config",
            FAST_CONFIG,
        ]
    )
    code = _cmd_run(ns)
    assert code == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert "x" in data


def test_cmd_run_file_not_found_returns_1():
    from givp.cli import _build_parser, _cmd_run

    parser = _build_parser()
    ns = parser.parse_args(
        [
            "run",
            "--func-file",
            "nonexistent.py",
            "--func-name",
            "sphere",
            "--bounds",
            BOUNDS_1D,
        ]
    )
    code = _cmd_run(ns)
    assert code == 1


def test_cmd_run_bad_bounds_returns_nonzero():
    from givp.cli import _build_parser, _cmd_run

    parser = _build_parser()
    ns = parser.parse_args(
        [
            "run",
            "--func-file",
            SPHERE_FILE,
            "--func-name",
            "sphere",
            "--bounds",
            "not-json",
            "--config",
            FAST_CONFIG,
        ]
    )
    code = _cmd_run(ns)
    assert code != 0


def test_cmd_run_dict_config_from_json_flag(capsys):
    """When --json carries a dict 'config', it should be used as GIVPConfig kwargs."""
    from givp.cli import _build_parser, _cmd_run

    payload = json.dumps(
        {
            "func_file": SPHERE_FILE,
            "func_name": "sphere",
            "bounds": [[-5, 5]],
            "config": json.loads(FAST_CONFIG),
        }
    )
    parser = _build_parser()
    ns = parser.parse_args(["run", "--json", payload])
    code = _cmd_run(ns)
    assert code == 0
    captured = capsys.readouterr()
    assert "x" in json.loads(captured.out)


def test_load_func_spec_none_raises():
    """Cover the ImportError branch when spec_from_file_location returns None."""
    from givp.cli import _load_func

    with (
        patch("importlib.util.spec_from_file_location", return_value=None),
        pytest.raises(ImportError, match="Cannot load module"),
    ):
        _load_func(SPHERE_FILE, "sphere")


def test_load_func_spec_loader_none_raises():
    """Cover the ImportError branch when spec.loader is None."""
    from givp.cli import _load_func

    fake_spec = MagicMock()
    fake_spec.loader = None
    with (
        patch("importlib.util.spec_from_file_location", return_value=fake_spec),
        pytest.raises(ImportError, match="Cannot load module"),
    ):
        _load_func(SPHERE_FILE, "sphere")


def test_resolve_args_stdin_dash(monkeypatch):
    """Cover the sys.stdin.read() branch when json_input == '-'."""
    from givp.cli import _build_parser, _resolve_args

    payload = json.dumps(
        {"func_file": SPHERE_FILE, "func_name": "sphere", "bounds": [[-5, 5]]}
    )
    monkeypatch.setattr("sys.stdin", io.StringIO(payload))

    parser = _build_parser()
    ns = parser.parse_args(["run", "--json", "-"])
    args = _resolve_args(ns)
    assert args["func_file"] == SPHERE_FILE


def test_resolve_args_bounds_and_direction_flags():
    """Cover the bounds, direction, config and seed namespace branches in _resolve_args."""
    from givp.cli import _build_parser, _resolve_args

    parser = _build_parser()
    ns = parser.parse_args(
        [
            "run",
            "--func-file",
            SPHERE_FILE,
            "--func-name",
            "sphere",
            "--bounds",
            BOUNDS_1D,
            "--direction",
            "maximize",
            "--config",
            FAST_CONFIG,
            "--seed",
            "7",
        ]
    )
    args = _resolve_args(ns)
    assert args["bounds"] == BOUNDS_1D
    assert args["direction"] == "maximize"
    assert args["config"] == FAST_CONFIG
    assert args["seed"] == 7


def test_cmd_run_generic_exception_returns_1():
    """Cover the bare `except Exception` handler in _cmd_run."""
    from givp.cli import _build_parser, _cmd_run

    parser = _build_parser()
    ns = parser.parse_args(
        [
            "run",
            "--func-file",
            SPHERE_FILE,
            "--func-name",
            "sphere",
            "--bounds",
            BOUNDS_1D,
            "--config",
            FAST_CONFIG,
        ]
    )
    with patch("givp.cli.givp", side_effect=RuntimeError("boom")):
        code = _cmd_run(ns)
    assert code == 1


def test_main_calls_sys_exit(monkeypatch):
    """Cover the main() function body."""
    from givp.cli import main

    mock_ns = MagicMock()
    mock_ns.func.return_value = 0

    with patch("givp.cli._build_parser") as mock_build:
        mock_build.return_value.parse_args.return_value = mock_ns
        with pytest.raises(SystemExit) as exc_info:
            main()
    assert exc_info.value.code == 0
