# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT
"""
Command-line interface for the ``givp`` optimizer.

Usage examples
--------------
Run from a file that defines the objective function::

	givp run --func-file path/to/objective.py --func-name my_func \\
        --bounds "[[-5,5],[-5,5]]"

Pass all arguments as a single JSON blob (useful for agent/MCP integration)::

	givp run --json '{"func_file":"objective.py","func_name":"my_func","bounds":[[-5,5],[-5,5]]}'

Pipe JSON input::

	echo '{"func_file":"objective.py","func_name":"my_func","bounds":[[-5,5]]}' | givp run --json -

The output is always a JSON object printed to stdout. Errors go to stderr.
Exit code 0 on success, non-zero on any failure.

Output schema
-------------
{
  "x":           [float, ...],   # best solution vector
  "fun":         float,          # objective value at x
  "nit":         int,            # outer iterations executed
  "nfev":        int,            # total function evaluations
  "success":     bool,
  "termination": str, # one of: converged | max_iterations |
                      #   time_limit | early_stop | no_feasible | unknown
  "direction":   str             # "minimize" or "maximize"
}
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any

from givp.api import givp
from givp.config import GIVPConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_func(func_file: str, func_name: str) -> Any:
    """Load *func_name* from *func_file* using importlib (no exec/eval)."""
    path = Path(func_file).resolve()
    if not path.exists():
        raise FileNotFoundError(f"func-file not found: {path}")

    spec = importlib.util.spec_from_file_location("_givp_user_module", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from: {path}")

    module: types.ModuleType = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]

    if not hasattr(module, func_name):
        raise AttributeError(
            f"Function '{func_name}' not found in '{path}'. "
            f"Available names: {[n for n in dir(module) if not n.startswith('_')]}"
        )
    return getattr(module, func_name)


def _parse_bounds(raw: str) -> list[tuple[float, float]]:
    """Parse a JSON array of [low, high] pairs into a list of (low, high) tuples."""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--bounds must be valid JSON, got: {raw!r}") from exc

    if not isinstance(parsed, list):
        raise ValueError("--bounds must be a JSON array, e.g. [[-5,5],[-5,5]]")

    result: list[tuple[float, float]] = []
    for item in parsed:
        if not (isinstance(item, (list, tuple)) and len(item) == 2):
            raise ValueError(f"Each bound must be [low, high], got: {item!r}")
        result.append((float(item[0]), float(item[1])))
    return result


def _parse_config(raw: str | None) -> GIVPConfig:
    """Parse an optional JSON object into a GIVPConfig."""
    if raw is None:
        return GIVPConfig()
    try:
        kwargs = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--config must be valid JSON, got: {raw!r}") from exc
    if not isinstance(kwargs, dict):
        raise ValueError("--config must be a JSON object")
    return GIVPConfig(**kwargs)


def _resolve_args(namespace: argparse.Namespace) -> dict[str, Any]:
    """Merge --json input with explicit flags (explicit flags take precedence)."""
    merged: dict[str, Any] = {}

    if namespace.json_input is not None:
        raw = namespace.json_input
        if raw == "-":
            raw = sys.stdin.read()
        try:
            merged = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"--json must be valid JSON: {exc}") from exc

    # Explicit CLI flags override anything from --json
    if namespace.func_file is not None:
        merged["func_file"] = namespace.func_file
    if namespace.func_name is not None:
        merged["func_name"] = namespace.func_name
    if namespace.bounds is not None:
        merged["bounds"] = namespace.bounds
    if namespace.direction is not None:
        merged["direction"] = namespace.direction
    if namespace.config is not None:
        merged["config"] = namespace.config
    if namespace.seed is not None:
        merged["seed"] = namespace.seed

    return merged


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


def _cmd_run(namespace: argparse.Namespace) -> int:
    try:
        args = _resolve_args(namespace)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # Validate required fields
    missing = [k for k in ("func_file", "func_name", "bounds") if k not in args]
    if missing:
        print(
            f"error: missing required argument(s): {', '.join(missing)}\n"
            "       Provide via --func-file/--func-name/--bounds or --json.",
            file=sys.stderr,
        )
        return 2

    try:
        func = _load_func(args["func_file"], args["func_name"])
        bounds = (
            _parse_bounds(args["bounds"])
            if isinstance(args["bounds"], str)
            else [tuple(b) for b in args["bounds"]]
        )
        config = _parse_config(
            args.get("config") if isinstance(args.get("config"), str) else None
        )
        if isinstance(args.get("config"), dict):
            config = GIVPConfig(**args["config"])

        result = givp(
            func,
            bounds,
            direction=args.get("direction", "minimize"),
            config=config,
            seed=args.get("seed"),
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (AttributeError, ImportError, ValueError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pylint: disable=broad-exception-caught
        # Catch any unexpected exception and surface as a CLI error code.
        print(f"error: unexpected failure — {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result.to_dict()))
    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="givp",
        description="GRASP-ILS-VND-PR optimizer CLI",
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    sub.required = True

    run_p = sub.add_parser("run", help="Run the optimizer on an objective function")
    run_p.add_argument(
        "--func-file",
        metavar="PATH",
        help="Path to a .py file containing the objective function",
    )
    run_p.add_argument(
        "--func-name",
        metavar="NAME",
        help="Name of the callable inside --func-file",
    )
    run_p.add_argument(
        "--bounds",
        metavar="JSON",
        help="Variable bounds as JSON, e.g. '[[-5,5],[-5,5]]'",
    )
    run_p.add_argument(
        "--direction",
        choices=["minimize", "maximize"],
        default=None,
        help="Optimization direction (default: minimize)",
    )
    run_p.add_argument(
        "--config",
        metavar="JSON",
        default=None,
        help="GIVPConfig fields as JSON, e.g. '{\"max_iterations\":200}'",
    )
    run_p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    run_p.add_argument(
        "--json",
        dest="json_input",
        metavar="JSON|-",
        default=None,
        help="All arguments as a JSON object; use - to read from stdin",
    )
    run_p.set_defaults(func=_cmd_run)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for the givp CLI — parse args and run subcommand."""
    parser = _build_parser()
    namespace = parser.parse_args()
    sys.exit(namespace.func(namespace))


if __name__ == "__main__":  # pragma: no cover
    main()
