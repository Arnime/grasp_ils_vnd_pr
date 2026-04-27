# CLI Reference

`givp` ships a command-line entry point that lets you run the optimizer
without writing a single line of Python driver code. It is particularly
useful for shell scripts, CI pipelines, and **AI agent / LLM tool-call
integrations** where the caller speaks JSON.

## Installation

The entry point is registered automatically when you install the package:

```bash
pip install givp
givp --help
```

---

## `givp run`

```text
givp run --func-file <path.py> --func-name <name> --bounds <json>
         [--direction minimize|maximize]
         [--config <json>]
         [--seed <int>]
         [--json <json>|-]
```

| Flag | Description |
|---|---|
| `--func-file PATH` | Path to a `.py` file that defines the objective function |
| `--func-name NAME` | Name of the callable inside `--func-file` |
| `--bounds JSON` | Variable bounds as a JSON array of `[low, high]` pairs |
| `--direction` | `minimize` *(default)* or `maximize` |
| `--config JSON` | `GIVPConfig` fields as a JSON object |
| `--seed INT` | Random seed for reproducibility |
| `--json JSON\|-` | Pass all arguments as a single JSON object; use `-` to read from stdin |

### Output

JSON printed to **stdout**. Errors go to **stderr**. Exit code `0` on
success, non-zero on any failure.

```json
{
  "x":           [float, ...],
  "fun":         float,
  "nit":         int,
  "nfev":        int,
  "success":     bool,
  "termination": "converged | max_iterations | time_limit | early_stop | 
  no_feasible | unknown",
  "direction":   "minimize | maximize"
}
```

!!! note "Why a closed enum for `termination`?"
    The `termination` field is a **closed set** of known values (see
    [`TerminationReason`](api/result.md)). This is intentional: free-form
    strings from the optimizer's internal state would be a prompt-injection
    vector when the JSON output is fed directly to an LLM. Enumerating the
    set of possible values eliminates that risk.

---

## Examples

### Minimize a simple function

Given `objective.py`:

```python
# objective.py
import numpy as np

def sphere(x: np.ndarray) -> float:
    return float(np.sum(x ** 2))
```

Run the optimizer:

```bash
givp run \
  --func-file objective.py \
  --func-name sphere \
  --bounds "[[-5,5],[-5,5],[-5,5],[-5,5]]"
```

Example output:

```json
{
  "x": [-0.0023, 0.0011, -0.0008, 0.0014],
  "fun": 0.0000082,
  "nit": 100,
  "nfev": 48320,
  "success": true,
  "termination": "max_iterations",
  "direction": "minimize"
}
```

---

### Maximize

```bash
givp run \
  --func-file objective.py \
  --func-name neg_sphere \
  --bounds "[[-5,5],[-5,5]]" \
  --direction maximize
```

---

### Custom configuration

Pass any `GIVPConfig` field as a JSON object:

```bash
givp run \
  --func-file objective.py \
  --func-name sphere \
  --bounds "[[-5,5]]" \
  --config '{"max_iterations": 200, "time_limit": 5.0, "use_cache": true}'
```

---

### Reproducible runs

```bash
givp run \
  --func-file objective.py \
  --func-name sphere \
  --bounds "[[-5,5]]" \
  --seed 42
```

---

### All arguments as a JSON blob (`--json`)

Useful when calling from an LLM tool or orchestrator that builds the
parameters programmatically:

```bash
givp run --json '{
  "func_file": "objective.py",
  "func_name": "sphere",
  "bounds": [[-5, 5], [-5, 5]],
  "direction": "minimize",
  "config": {"max_iterations": 300},
  "seed": 7
}'
```

Explicit flags always **override** keys from `--json`:

```bash
# JSON says 2-D, --bounds overrides to 1-D
givp run --json '{"func_file":"objective.py","func_name":"sphere","bounds": \
         [[-5,5],[-5,5]]}--bounds "[[-5,5]]"
```

---

### Read from stdin (`--json -`)

Pipe JSON from another process:

```bash
echo '{"func_file":"objective.py","func_name":"sphere","bounds":[[-5,5]]}' \
  | givp run --json -
```

Or from a file:

```bash
cat params.json | givp run --json -
```

---

## Agent / LLM integration

The CLI is designed for safe use as an **LLM tool call**. The recommended
pattern is:

1. The agent generates a JSON object with `func_file`, `func_name`,
   `bounds`, and optional fields.
2. It calls `givp run --json '<payload>'` and captures stdout.
3. It parses the JSON output — all fields are typed primitives, and
   `termination` is a closed enum, so the agent can safely include it in a
   prompt without prompt-injection risk.

```python
import json, subprocess

payload = json.dumps({
    "func_file": "objective.py",
    "func_name": "sphere",
    "bounds":    [[-5, 5]] * 4,
    "config":    {"max_iterations": 200},
    "seed":      42,
})

proc = subprocess.run(
    ["givp", "run", "--json", payload],
    capture_output=True,
    text=True,
)

if proc.returncode != 0:
    raise RuntimeError(proc.stderr)

result = json.loads(proc.stdout)
# result["termination"] is always one of the known TerminationReason values
print(result["x"], result["fun"])
```

---

## Exit codes

| Code | Meaning |
|---|---|
| `0` | Optimization completed; JSON written to stdout |
| `1` | Runtime error (file not found, bad function, optimizer failure) |
| `2` | Invalid or missing arguments |

---

## Security considerations

The CLI loads the objective function via `importlib.util.spec_from_file_location`,
which executes the file as a Python module. This is equivalent to running
`python objective.py` directly — **only pass files you trust**. The
function file path is always resolved to an absolute path before loading,
and the result schema never includes the raw `meta` dict or any
free-form internal state.
