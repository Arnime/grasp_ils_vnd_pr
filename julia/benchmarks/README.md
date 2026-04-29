# Benchmark Results

## Performance benchmarks

Run the GIVPOptimizer micro-benchmarks with:

```bash
cd julia
julia --project=. benchmarks/benchmarks.jl
```

### Test functions

| Function   | Bounds        | Description                                |
|------------|---------------|--------------------------------------------|
| sphere     | [-5, 5]       | Sum of squares; minimum 0 at origin        |
| rosenbrock | [-2, 2]       | Banana function; minimum 0 at (1,…,1)      |
| rastrigin  | [-5.12, 5.12] | Highly multimodal; minimum 0 at origin     |
| ackley     | [-5, 5]       | Multimodal with flat regions; min 0 at 0   |

Each function is benchmarked at dimensions **5** and **10** (8 total cases).
Results are saved to `benchmarks/results.json`. Subsequent runs compare
against the previous results and flag regressions (>10% time increase).

---

## Literature comparison experiment

Reproducible multi-run experiment comparing GIVP against baselines on six
standard benchmark functions. Produces statistics and Wilcoxon-signed-rank
tables ready for SBPO / BRACIS papers.

```bash
# 30 runs × 10-D × GIVP-full + GRASP-only (all 6 functions)
julia --project=julia julia/benchmarks/run_literature_comparison.jl \
    --n-runs 30 --dims 10 --output results.json --verbose

# Include BlackBoxOptim.jl baselines (Differential Evolution and XNES)
# Requires: julia -e 'using Pkg; Pkg.add("BlackBoxOptim")'
julia --project=julia julia/benchmarks/run_literature_comparison.jl \
    --algorithms GIVP-full GRASP-only BBO-DE BBO-XNES

# Capture per-iteration convergence traces
julia --project=julia julia/benchmarks/run_literature_comparison.jl --traces

# Resume a partial run (checkpoint after every completed seed)
julia --project=julia julia/benchmarks/run_literature_comparison.jl --resume
```

### Generating reports

```bash
# Markdown + LaTeX tables with Wilcoxon tests (reads results.json)
julia --project=julia julia/benchmarks/generate_report.jl \
    --input results.json --format both

# With convergence curves (only if --traces was used above)
julia --project=julia julia/benchmarks/generate_report.jl \
    --input results.json --convergence --checkpoints 1 5 10 25 50 75 100
```

Outputs: `results_report.md` and `results_report.tex` in the same directory.

### Six benchmark functions

| Function   | Domain                     | Known optimum |
|------------|----------------------------|---------------|
| Sphere     | $[-5.12,\,5.12]^n$         | 0             |
| Rosenbrock | $[-5,\,10]^n$              | 0             |
| Rastrigin  | $[-5.12,\,5.12]^n$         | 0             |
| Ackley     | $[-32.768,\,32.768]^n$     | 0             |
| Griewank   | $[-600,\,600]^n$           | 0             |
| Schwefel   | $[-500,\,500]^n$           | 0             |

### Interactive notebook

`Notebooks/Julia/benchmark_literature_comparison_julia.ipynb` — run the full
experiment and generate all tables interactively in Jupyter.
