# Contributing to givp

Thanks for your interest in improving `givp`! This document describes the
preferred workflow for changes.

## Development setup

```bash
git clone https://github.com/Arnime/grasp_ils_vnd_pr
cd grasp_ils_vnd_pr
python -m venv .venv
.venv/Scripts/activate          # Windows
# source .venv/bin/activate     # Linux/macOS
pip install -e .[dev,docs]
pre-commit install
```

## Quality gates

Before opening a PR, please confirm locally that:

```bash
ruff check src tests
mypy
pytest
mkdocs build --strict
```

Or run all of the above with `nox`:

```bash
pip install nox
nox
```

## Pull request checklist

- [ ] Tests added or updated for the change.
- [ ] `pytest` passes locally with coverage ≥ 95%.
- [ ] `mypy` reports no errors.
- [ ] `ruff` is clean.
- [ ] Public API additions are documented in `docs/`.
- [ ] If user-visible, the change is mentioned in the PR description for
      the changelog.

## Developer Certificate of Origin (DCO)

By submitting a Pull Request you certify that you have the right to
contribute the code under the project's MIT licence and you agree to the
[Developer Certificate of Origin v1.1](https://developercertificate.org/).

Add a `Signed-off-by` trailer to every commit in your PR:

```bash
git commit -s -m "your commit message"
```

This appends a line of the form:

```
Signed-off-by: Your Name <your.email@example.com>
```

The CI DCO check will fail if any commit in the PR lacks this trailer.

## Test policy

**All Pull Requests that add or change observable behaviour MUST include
tests.** Specifically:

- New features require at least one positive test and one negative/edge
  case test.
- Bug fixes require a regression test that would have caught the bug.
- The coverage gate (`--cov-fail-under=95`) is enforced in CI and will
  fail the PR if coverage drops below 95%.

PRs that reduce coverage or omit tests for new code will not be merged.

## Commit messages

Please follow the [Conventional Commits](https://www.conventionalcommits.org)
format, which enables automated CHANGELOG generation:

```text
feat(api): expose `seed` parameter for reproducibility
fix(core): guard against empty bounds list
docs: add quickstart for warm-start
```

## Repository policy

The repository enforces branch protection: all Pull Requests require at
least one approving review and passing CI checks (lint, type checks and
tests) before merging. Please ensure your PR satisfies the checklist above
and wait for required status checks to pass before requesting a merge.

If you are a maintainer and need branch protection changes, manage them via
the repository settings on GitHub.

## Reporting bugs

Open a GitHub issue with:

- `givp` version, Python version, OS;
- minimal reproducer (function + bounds + config);
- expected vs actual behaviour;
- traceback if any.
