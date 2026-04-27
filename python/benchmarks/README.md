# Benchmarks

Performance benchmarks for `givp`, executed on demand (excluded from the
default test run).

## Run locally

```bash
pip install -e .[dev]
pytest benchmarks/ --benchmark-only --benchmark-autosave
```

Results are stored under `.benchmarks/`. To compare against a saved baseline:

```bash
pytest benchmarks/ --benchmark-compare=0001 --benchmark-only
```

## Functions covered

| Function    | Domain          | Global optimum |
|-------------|-----------------|----------------|
| Sphere      | `[-5, 5]^n`     | 0 at origin    |
| Rosenbrock  | `[-2, 2]^n`     | 0 at (1,...,1) |
| Rastrigin   | `[-5.12, 5.12]` | 0 at origin    |
| Ackley      | `[-5, 5]^n`     | 0 at origin    |
