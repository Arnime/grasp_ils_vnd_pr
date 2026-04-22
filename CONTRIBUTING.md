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

## Commit messages

Please follow the [Conventional Commits](https://www.conventionalcommits.org)
format, which enables automated CHANGELOG generation:

```
feat(api): expose `seed` parameter for reproducibility
fix(core): guard against empty bounds list
docs: add quickstart for warm-start
```

## Reporting bugs

Open a GitHub issue with:
- `givp` version, Python version, OS;
- minimal reproducer (function + bounds + config);
- expected vs actual behaviour;
- traceback if any.
