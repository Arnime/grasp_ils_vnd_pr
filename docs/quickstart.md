# Quickstart

## Python

### Minimize a function

```python
import numpy as np
from givp import givp, GIVPConfig

def rosenbrock(x: np.ndarray) -> float:
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))

cfg = GIVPConfig(max_iterations=20, vnd_iterations=30)
result = givp(rosenbrock, [(-2.0, 2.0)] * 5, config=cfg)
print(result)
```

### Maximize

```python
result = givp(my_score, bounds, direction="maximize")
```

### Object-oriented API

`GIVPOptimizer` keeps the run history when you call `.run()` repeatedly:

```python
from givp import GIVPOptimizer

opt = GIVPOptimizer(rosenbrock, [(-2.0, 2.0)] * 5)
opt.run()
opt.run()
print(opt.best_x, opt.best_fun, len(opt.history))
```

### Warm start

```python
result = givp(
    rosenbrock,
    [(-2.0, 2.0)] * 5,
    initial_guess=[1.0, 1.0, 1.0, 1.0, 1.0],
)
```

### Stopping by wall clock

```python
cfg = GIVPConfig(time_limit=2.0)  # seconds
result = givp(rosenbrock, [(-2.0, 2.0)] * 5, config=cfg)
```

## Julia

### Minimize a function (Julia)

```julia
using GIVPOptimizer

function rosenbrock(x::Vector{Float64})::Float64
    return sum(
        100.0 .* (x[2:end] .- x[1:end-1] .^ 2) .^ 2
        .+ (1.0 .- x[1:end-1]) .^ 2
    )
end

cfg = GIVPConfig(; max_iterations=20, vnd_iterations=30)
result = givp(rosenbrock, [(-2.0, 2.0) for _ in 1:5]; config=cfg)
println(result)
```

### Maximize (Julia)

```julia
result = givp(my_score, bounds; direction=maximize)
```

### Warm start (Julia)

```julia
result = givp(
    rosenbrock,
    [(-2.0, 2.0) for _ in 1:5];
    initial_guess=[1.0, 1.0, 1.0, 1.0, 1.0],
)
```

### Stopping by wall clock (Julia)

```julia
cfg = GIVPConfig(; time_limit=2.0)
result = givp(rosenbrock, [(-2.0, 2.0) for _ in 1:5]; config=cfg)
```

### Running benchmarks

```bash
cd julia
julia --project=. benchmarks/benchmarks.jl
```

## Serializing the result

`OptimizeResult.to_dict()` converts the result to a JSON-safe dictionary
with typed primitives. The `termination` field is a closed enum value from
`TerminationReason` — safe to pass directly to an LLM or write to a log:

```python
import json
from givp import givp, TerminationReason

result = givp(rosenbrock, [(-2.0, 2.0)] * 5)
d = result.to_dict()
print(json.dumps(d, indent=2))
# {
#   "x": [1.002, 1.004, 1.008, 1.016, 1.032],
#   "fun": 0.0000321,
#   "nit": 100,
#   "nfev": 48300,
#   "success": true,
#   "termination": "max_iterations",
#   "direction": "minimize"
# }

# Check termination reason
if d["termination"] == TerminationReason.CONVERGED.value:
    print("fully converged!")
```

The possible `termination` values are:

| Value | Meaning |
|---|---|
| `converged` | The convergence monitor detected no meaningful improvement |
| `max_iterations` | The outer iteration budget was exhausted |
| `time_limit` | The wall-clock time limit was reached |
| `early_stop` | The early-stop threshold triggered |
| `no_feasible` | No feasible solution was found |
| `unknown` | Could not map the internal message to a known reason |

## Running from the command line

Install and call via the `givp` entry point — no driver script required:

```bash
givp run \
  --func-file objective.py \
  --func-name rosenbrock \
  --bounds "[[-2,2],[-2,2],[-2,2],[-2,2],[-2,2]]" \
  --config '{"max_iterations": 20}'
```

See the full [CLI Reference](cli.md) for all flags, JSON input, and agent
integration patterns.
