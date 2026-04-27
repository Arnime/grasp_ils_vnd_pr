# Roadmap

This document describes the planned direction for `givp` over the next
twelve months. Items are subject to change based on user feedback and
available contributor time.

## Current version

**v0.3.x** — stable, feature-complete implementation of the GRASP + ILS +
VND + Path Relinking metaheuristic for continuous black-box optimization.

## Short-term (next 3 months)

- **Parallel neighbourhood evaluation**: allow users to provide a
  parallelism hint so VND neighbourhoods can be evaluated concurrently
  on multi-core machines.
- **Progress callback**: expose an optional `callback` parameter that
  receives the best solution found after each iteration, enabling
  custom early-stopping and progress bars.
- **Expanded examples**: add worked examples for combinatorial
  problems (e.g., TSP-style discretised objective) and multi-objective
  scalarisation.

## Medium-term (3–6 months)

- **Warm-start initialisation**: allow callers to seed the elite pool
  with known-good solutions to accelerate convergence.
- **Configurable path-relinking strategies**: expose `forward`,
  `backward`, and `random` PR direction as an explicit option.
- **Documentation improvements**: add a dedicated Architecture page
  and benchmark comparison charts.

## Long-term (6–12 months)

- **Optional scikit-learn integration**: expose `givp` as a
  scikit-learn-compatible `BaseEstimator` for hyper-parameter tuning
  workflows.
- **Type-safe bounds specification**: accept named-parameter bounds via
  a mapping in addition to the current sequence-of-pairs format.
- **Async support**: explore asyncio-compatible runner for use in
  Jupyter and async frameworks.

## Out of scope

The following are explicitly out of scope for this project:

- **Gradient-based optimisation** — use SciPy or PyTorch for that.
- **Integer / mixed-integer programming** — dedicated MIP solvers
  (e.g., PuLP, OR-Tools) are better suited.
- **GPU acceleration** — not currently planned.

## Feedback

If a feature you need is missing, please open a
[GitHub Issue](https://github.com/Arnime/givp/issues) with the
label `enhancement`.
