# Benchmark Results

Run the GIVPOptimizer performance benchmarks with:

```bash
cd julia
julia --project=. benchmarks/benchmarks.jl
```

## Test Functions

| Function   | Bounds        | Description                                |
|------------|---------------|--------------------------------------------|
| sphere     | [-5, 5]       | Sum of squares; minimum 0 at origin        |
| rosenbrock | [-2, 2]       | Banana function; minimum 0 at (1,…,1)      |
| rastrigin  | [-5.12, 5.12] | Highly multimodal; minimum 0 at origin     |
| ackley     | [-5, 5]       | Multimodal with flat regions; min 0 at 0   |

Each function is benchmarked at dimensions **5** and **10** (8 total cases).

## Regression Tracking

Results are saved to `benchmarks/results.json` after each run. On subsequent runs,
the script automatically compares against the previous results and flags regressions
(>10% time increase).
