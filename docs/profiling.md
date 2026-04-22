# Profiling givp

When investigating performance regressions or proposing optimizations to the
hot path (`_neighborhood_*`, `local_search_vnd`, `_run_grasp_loop`), please
attach a profile so reviewers can verify the impact.

## Quick CPU profile with `py-spy`

[`py-spy`](https://github.com/benfred/py-spy) is a sampling profiler that
attaches to a running Python process without instrumentation overhead.

```powershell
pip install py-spy
py-spy record -o profile.svg -- python -m pytest benchmarks/ --benchmark-only -k "sphere and 30"
```

Open `profile.svg` in a browser to inspect the flamegraph.

## Line-level profile with `scalene`

[`scalene`](https://github.com/plasma-umass/scalene) reports CPU **and**
memory usage per line, which is useful for spotting accidental allocations
in the hot path.

```powershell
pip install scalene
scalene --html --outfile scalene.html benchmarks/runner.py
```

## Reproducible benchmark runs

The repository ships a `pytest-benchmark` suite under `benchmarks/`. Use
`--benchmark-autosave` to compare runs over time:

```powershell
pytest benchmarks/ --benchmark-only --benchmark-autosave
pytest-benchmark compare 0001 0002 --columns=mean,stddev,ops
```

Pin the master RNG via the public `seed=` parameter to make timing
comparisons deterministic across runs.
