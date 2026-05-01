# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT

"""GIVP command-line interface (Julia port of the Python `givp run` command).

Runs the GRASP-ILS-VND-PR optimizer and writes a JSON result to stdout.

Usage
-----
    # Load function from a Julia source file
    julia julia/cli.jl run \\
        --func-file examples/sphere.jl --func-name sphere \\
        --bounds '[[-5.12,-5.12,-5.12],[5.12,5.12,5.12]]' \\
        --direction minimize --seed 42 --verbose

    # Inline lambda (simple functions)
    julia julia/cli.jl run \\
        --func '(x) -> sum(x .^ 2)' \\
        --bounds '[[-5.0,-5.0],[5.0,5.0]]' \\
        --seed 0

    # JSON mode — mirrors Python CLI
    julia julia/cli.jl run --json '{
        "func_file": "examples/sphere.jl",
        "func_name": "sphere",
        "bounds": [[-5.12,-5.12],[5.12,5.12]],
        "direction": "minimize",
        "seed": 42
    }'

    # Version
    julia julia/cli.jl --version

Output (stdout, JSON)
---------------------
    {
      "x":       [best solution vector],
      "fun":     best objective value,
      "nit":     number of iterations,
      "nfev":    number of function evaluations,
      "success": true/false,
      "message": "termination reason",
      "time_s":  wall-clock seconds,
      "givp_version": "1.0.0"
    }

Bounds format
-------------
Two accepted forms (same as Python CLI):
    - [[lo1, lo2, ...], [hi1, hi2, ...]]   (2-element array of lower/upper)
    - [[lo1, hi1], [lo2, hi2], ...]         (n-element array of per-var pairs)
"""

using Pkg
Pkg.activate(joinpath(@__DIR__))
Pkg.instantiate()

using GIVPOptimizer
using JSON

# ── Helpers ───────────────────────────────────────────────────────────────────

function _die(msg::String, code::Int = 1)
    println(stderr, "givp: error: $msg")
    exit(code)
end

function _parse_bounds(raw::Any)::Vector{Tuple{Float64, Float64}}
    raw isa Vector || _die("bounds must be a JSON array")
    length(raw) >= 2 || _die("bounds must have at least 2 elements")
    first = raw[1]
    if first isa Vector
        if length(raw) == 2 && length(raw[1]) > 1 && length(raw[2]) > 1
            # [[lo1,lo2,...], [hi1,hi2,...]]  — 2-element lower/upper arrays
            lows = Float64.(raw[1])
            highs = Float64.(raw[2])
            length(lows) == length(highs) ||
                _die("lower and upper arrays must have the same length")
            return [(lows[i], highs[i]) for i in eachindex(lows)]
        else
            # [[lo1,hi1], [lo2,hi2], ...] — per-variable pairs
            return [Tuple{Float64, Float64}((Float64(p[1]), Float64(p[2]))) for p in raw]
        end
    else
        # flat [lo, hi] — single-variable
        length(raw) == 2 || _die("flat bounds must be [lo, hi] (single variable)")
        return [(Float64(raw[1]), Float64(raw[2]))]
    end
end

function _parse_config(raw::Union{Dict, Nothing})::GIVPConfig
    raw === nothing && return GIVPConfig()
    cfg = GIVPConfig()
    # Map JSON keys to GIVPConfig fields
    field_map = Dict{String, Symbol}(
        "max_iterations" => :max_iterations,
        "alpha" => :alpha,
        "vnd_iterations" => :vnd_iterations,
        "ils_iterations" => :ils_iterations,
        "perturbation_strength" => :perturbation_strength,
        "use_elite_pool" => :use_elite_pool,
        "elite_size" => :elite_size,
        "path_relink_frequency" => :path_relink_frequency,
        "adaptive_alpha" => :adaptive_alpha,
        "alpha_min" => :alpha_min,
        "alpha_max" => :alpha_max,
        "num_candidates_per_step" => :num_candidates_per_step,
        "use_cache" => :use_cache,
        "cache_size" => :cache_size,
        "early_stop_threshold" => :early_stop_threshold,
        "use_convergence_monitor" => :use_convergence_monitor,
        "n_workers" => :n_workers,
        "time_limit" => :time_limit,
        "integer_split" => :integer_split,
        "group_size" => :group_size,
    )
    for (key, field) in field_map
        haskey(raw, key) && setproperty!(cfg, field, raw[key])
    end
    return cfg
end

function _load_func_from_file(path::String, name::String)::Function
    isfile(path) || _die("func-file not found: $path")
    m = Module()
    Base.include(m, abspath(path))
    isdefined(m, Symbol(name)) || _die("function '$name' not defined in '$path'")
    getfield(m, Symbol(name))
end

function _load_inline_func(expr::String)::Function
    # Security note: eval(Meta.parse(expr)) executes arbitrary Julia code
    # supplied by the caller.  This is equivalent to running the expression
    # directly in the REPL.  Only pass expressions from trusted sources —
    # never from untrusted network input or user-submitted data.
    try
        f = eval(Meta.parse(expr))
        f isa Function || _die("--func expression did not evaluate to a Function")
        return f
    catch e
        _die("could not parse --func expression: $e")
    end
end

# ── Command: run ─────────────────────────────────────────────────────────────

function cmd_run(argv::Vector{String})
    params = Dict{String, Any}(
        "func-file" => "",
        "func-name" => "objective",
        "func" => "",
        "bounds" => nothing,
        "direction" => "minimize",
        "seed" => nothing,
        "config" => nothing,
        "json" => "",
        "verbose" => false,
    )

    i = 1
    while i <= length(argv)
        arg = argv[i]
        if arg in ("--help", "-h")
            println("""
Usage: julia julia/cli.jl run [OPTIONS]

Function source (one required):
  --func-file PATH    Julia source file containing the objective function
  --func-name NAME    Function name inside --func-file (default: "objective")
  --func EXPR         Inline lambda, e.g. '(x) -> sum(x .^ 2)'

Problem definition:
  --bounds JSON       Bounds as JSON string (see formats below)
  --direction STR     minimize | maximize (default: minimize)
  --seed INT          RNG seed for reproducibility

Algorithm:
  --config JSON       GIVPConfig overrides as JSON object, e.g. '{"max_iterations":50}'

Convenience:
  --json JSON         All parameters as a single JSON object
  --verbose           Print progress to stderr

Bounds formats:
  [[lo1, lo2, ...], [hi1, hi2, ...]]  — lower and upper arrays
  [[lo1, hi1], [lo2, hi2], ...]       — per-variable pairs

Output (stdout): JSON with x, fun, nit, nfev, success, message, time_s, givp_version
""")
            exit(0)
        elseif arg == "--func-file" && i < length(argv)
            params["func-file"] = argv[i + 1]
            i += 2
        elseif arg == "--func-name" && i < length(argv)
            params["func-name"] = argv[i + 1]
            i += 2
        elseif arg == "--func" && i < length(argv)
            params["func"] = argv[i + 1]
            i += 2
        elseif arg == "--bounds" && i < length(argv)
            params["bounds"] = JSON.parse(argv[i + 1])
            i += 2
        elseif arg == "--direction" && i < length(argv)
            params["direction"] = argv[i + 1]
            i += 2
        elseif arg == "--seed" && i < length(argv)
            params["seed"] = parse(Int, argv[i + 1])
            i += 2
        elseif arg == "--config" && i < length(argv)
            params["config"] = JSON.parse(argv[i + 1])
            i += 2
        elseif arg == "--json" && i < length(argv)
            params["json"] = argv[i + 1]
            i += 2
        elseif arg == "--verbose"
            params["verbose"] = true
            i += 1
        else
            println(stderr, "givp run: unknown option: $arg (ignored)")
            i += 1
        end
    end

    # JSON mode: merge JSON fields into params (explicit flags override)
    if !isempty(params["json"])
        raw_json = JSON.parse(params["json"])
        for (k, v) in raw_json
            k_dash = replace(k, "_" => "-")
            (isempty(get(params, k_dash, "")) || params[k_dash] === nothing) &&
                (params[k_dash] = v)
        end
    end

    # Resolve objective function
    func = if !isempty(params["func"])
        _load_inline_func(params["func"])
    elseif !isempty(params["func-file"])
        _load_func_from_file(params["func-file"], params["func-name"])
    else
        _die("one of --func, --func-file, or --json with func_file is required")
    end

    # Resolve bounds
    params["bounds"] === nothing && _die("--bounds is required (or provide in --json)")
    bounds = _parse_bounds(params["bounds"])

    # Direction
    dir_str = lowercase(params["direction"])
    direction =
        dir_str == "maximize" ? maximize :
        dir_str == "minimize" ? minimize :
        _die("direction must be 'minimize' or 'maximize', got: $(params["direction"])")

    # Config
    cfg = _parse_config(params["config"])
    cfg.direction = direction

    # Seed
    seed = params["seed"]

    # Run
    t0 = time()
    result = try
        givp(
            func,
            bounds;
            config = cfg,
            seed = seed,
            direction = direction,
            verbose = params["verbose"],
        )
    catch e
        if e isa GivpError
            _die(sprint(showerror, e))
        else
            rethrow(e)
        end
    end
    elapsed = time() - t0

    # Output JSON to stdout
    output = Dict{String, Any}(
        "x" => result.x,
        "fun" => result.fun,
        "nit" => result.nit,
        "nfev" => result.nfev,
        "success" => result.success,
        "message" => result.message,
        "time_s" => elapsed,
        "givp_version" => GIVPOptimizer.__version__,
    )
    println(JSON.json(output, 2))
end

# ── Entry point ───────────────────────────────────────────────────────────────

function main()
    argv = collect(ARGS)

    if isempty(argv) || argv[1] in ("--help", "-h")
        println("""
GIVP — GRASP-ILS-VND with Path Relinking (Julia v$(GIVPOptimizer.__version__))

Usage: julia julia/cli.jl <command> [OPTIONS]

Commands:
  run      Run the optimizer on a function
  version  Print version information

Run 'julia julia/cli.jl run --help' for detailed usage.
""")
        exit(0)
    end

    cmd = argv[1]
    rest = argv[2:end]

    if cmd in ("--version", "version", "-V")
        println("givp $(GIVPOptimizer.__version__) (Julia $(VERSION))")
        exit(0)
    elseif cmd == "run"
        cmd_run(rest)
    else
        _die("unknown command: '$cmd'. Use 'run' or '--help'.", 2)
    end
end

main()
