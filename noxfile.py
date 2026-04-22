"""Nox sessions: ``pip install nox`` then run ``nox`` locally to mirror CI."""

from __future__ import annotations

import nox

nox.options.sessions = ["lint", "typecheck", "tests", "docs"]
nox.options.reuse_existing_virtualenvs = True

PY_VERSIONS = ["3.10", "3.11", "3.12", "3.13"]
_DEV = ".[dev]"


@nox.session(python=PY_VERSIONS)
def tests(session: nox.Session) -> None:
    """Run the test suite with coverage."""
    session.install("-e", _DEV)
    session.run("pytest", *session.posargs)


@nox.session
def lint(session: nox.Session) -> None:
    """Run ruff."""
    session.install("ruff>=0.6")
    session.run("ruff", "check", "src", "tests")


@nox.session
def typecheck(session: nox.Session) -> None:
    """Run mypy."""
    session.install("-e", _DEV)
    session.run("mypy")


@nox.session
def docs(session: nox.Session) -> None:
    """Build docs in strict mode."""
    session.install("-e", ".[docs]")
    session.run("mkdocs", "build", "--strict")


@nox.session
def benchmarks(session: nox.Session) -> None:
    """Run the performance benchmarks."""
    session.install("-e", _DEV)
    session.run(
        "pytest",
        "benchmarks/",
        "--benchmark-only",
        "--benchmark-autosave",
        *session.posargs,
    )


@nox.session
def audit(session: nox.Session) -> None:
    """Audit dependencies for known CVEs."""
    session.install("pip-audit>=2.7")
    session.run("pip-audit", "--strict")
