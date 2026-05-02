# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT

"""GRASP-ILS-VND-PR — Main orchestrator."""

function evaluate_with_cache_impl(
    sol::Vector{Float64},
    cost_fn::Function,
    cache::Union{EvaluationCache, Nothing},
)::Float64
    if cache !== nothing
        cached = cache_get(cache, sol)
        cached !== nothing && return cached
        cost = cost_fn(sol)
        cache_put!(cache, sol, cost)
        return cost
    end
    return cost_fn(sol)
end

function do_path_relinking!(
    iteration::Int,
    best_cost::Float64,
    best_solution::Vector{Float64},
    stagnation::Int,
    config::GIVPConfig,
    elite_pool::Union{ElitePool, Nothing},
    cost_fn::Function,
    num_vars::Int;
    lower::Union{Vector{Float64}, Nothing} = nothing,
    upper::Union{Vector{Float64}, Nothing} = nothing,
    cache::Union{EvaluationCache, Nothing} = nothing,
    deadline::Float64 = 0.0,
)
    if !(
        config.use_elite_pool &&
        elite_pool !== nothing &&
        iteration > 0 &&
        iteration % config.path_relink_frequency == 0 &&
        pool_size(elite_pool) >= 2
    )
        return best_cost, best_solution, stagnation
    end

    elite_solutions = get_all(elite_pool)
    cached_fn = create_cached_cost_fn(cost_fn, cache)

    for i in 1:min(3, length(elite_solutions))
        for j in (i + 1):min(4, length(elite_solutions))
            expired(deadline) && break
            source = elite_solutions[i][1]
            target = elite_solutions[j][1]

            pr_solution, _ =
                bidirectional_path_relinking(cached_fn, source, target; deadline)
            pr_solution = local_search_vnd(
                cached_fn,
                pr_solution,
                num_vars;
                max_iter = config.vnd_iterations ÷ 2,
                lower,
                upper,
                cache,
                deadline,
            )
            pr_cost = cached_fn(pr_solution)

            if pr_cost < best_cost
                best_cost = pr_cost
                best_solution = copy(pr_solution)
                stagnation = 0
            end
            add!(elite_pool, pr_solution, pr_cost)
        end
    end

    return best_cost, best_solution, stagnation
end

function grasp_ils_vnd(
    cost_fn::Function,
    num_vars::Int,
    config::GIVPConfig;
    verbose::Bool = false,
    iteration_callback::Union{Function, Nothing} = nothing,
    lower::Union{Vector{Float64}, Nothing} = nothing,
    upper::Union{Vector{Float64}, Nothing} = nothing,
    initial_guess::Union{Vector{Float64}, Nothing} = nothing,
)
    if lower === nothing || upper === nothing
        throw(InvalidBoundsError("lower and upper bounds must be provided"))
    end
    if length(lower) != num_vars || length(upper) != num_vars
        throw(InvalidBoundsError("bounds must have length $num_vars"))
    end
    if any(upper .<= lower)
        throw(InvalidBoundsError("each element of upper must be > lower"))
    end

    # Set integer split
    set_integer_split!(config.integer_split)
    set_group_size!(config.group_size)

    # Initialize components
    elite_pool =
        config.use_elite_pool ? ElitePool(; max_size = config.elite_size, lower, upper) :
        nothing
    cache = config.use_cache ? EvaluationCache(; maxsize = config.cache_size) : nothing
    conv_monitor =
        config.use_convergence_monitor ?
        ConvergenceMonitor(; restart_threshold = config.early_stop_threshold) : nothing

    # Initial solution
    rng = new_rng()
    initial_arr = if initial_guess !== nothing
        copy(initial_guess)
    else
        lower .+ (upper .- lower) .* rand(rng, num_vars)
    end

    best_cost = Inf
    best_solution = copy(initial_arr)
    stagnation = 0
    start_time = time()
    deadline = config.time_limit > 0 ? start_time + config.time_limit : 0.0

    # Warm start
    if initial_guess !== nothing && elite_pool !== nothing
        init_cost = evaluate_with_cache_impl(initial_arr, cost_fn, cache)
        add!(elite_pool, copy(initial_arr), init_cost)
        if init_cost < best_cost
            best_cost = init_cost
            best_solution = copy(initial_arr)
        end
    end

    verbose && @info @sprintf(
        "GRASP-ILS-VND-PR start | n=%d | iters=%d | alpha=[%.3f, %.3f] | elite=%d",
        num_vars,
        config.max_iterations,
        config.adaptive_alpha ? config.alpha_min : config.alpha,
        config.adaptive_alpha ? config.alpha_max : config.alpha,
        config.use_elite_pool ? config.elite_size : 0
    )

    actual_nit = config.max_iterations
    termination_msg = "max iterations reached"
    for iteration in 0:(config.max_iterations - 1)
        if expired(deadline)
            verbose && @info @sprintf(
                "TIME LIMIT: %.0fs reached at iteration %d",
                config.time_limit,
                iteration + 1
            )
            termination_msg = "time limit reached"
            actual_nit = iteration
            break
        end

        current_alpha = get_current_alpha(iteration, config)

        # GRASP construction
        sol = construct_grasp(
            num_vars,
            lower,
            upper,
            cost_fn,
            initial_guess,
            current_alpha;
            num_candidates_per_step = config.num_candidates_per_step,
            cache,
            n_workers = config.n_workers,
        )

        # VND local search
        sol = local_search_vnd(
            cost_fn,
            sol,
            num_vars;
            max_iter = config.vnd_iterations,
            lower,
            upper,
            cache,
            deadline,
        )

        # ILS
        vnd_cost = evaluate_with_cache_impl(sol, cost_fn, cache)
        sol, vnd_cost = ils_search(
            sol,
            vnd_cost,
            num_vars,
            cost_fn,
            config;
            lower,
            upper,
            cache,
            deadline,
        )

        cost = evaluate_with_cache_impl(sol, cost_fn, cache)

        # Callback
        if iteration_callback !== nothing
            try
                iteration_callback(iteration, cost, sol)
            catch e
                @warn "iteration_callback raised" exception = (e, catch_backtrace())
            end
        end

        original_best = best_cost
        if cost < best_cost
            best_cost = cost
            best_solution = copy(sol)
            stagnation = 0
        else
            stagnation += 1
        end

        # Elite pool
        if config.use_elite_pool && elite_pool !== nothing
            add!(elite_pool, sol, cost)
        end

        # Convergence monitor
        if conv_monitor !== nothing
            status = update!(conv_monitor, best_cost, elite_pool)
            if status["should_restart"]
                if elite_pool !== nothing && pool_size(elite_pool) > 2
                    best_two = get_all(elite_pool)[1:2]
                    pool_clear!(elite_pool)
                    for (s, c) in best_two
                        add!(elite_pool, s, c)
                    end
                end
                stagnation = 0
            end
        end

        # Path relinking
        best_cost, best_solution, stagnation = do_path_relinking!(
            iteration,
            best_cost,
            best_solution,
            stagnation,
            config,
            elite_pool,
            cost_fn,
            num_vars;
            lower,
            upper,
            cache,
            deadline,
        )

        # Verbose logging
        if verbose
            elapsed = time() - start_time
            marker = cost < original_best ? "*" : " "
            @info @sprintf(
                "%s iter %3d/%d | cur=%12.4f | best=%12.4f | alpha=%.3f | stag=%3d | elite=%2d | t=%6.2fs",
                marker,
                iteration + 1,
                config.max_iterations,
                cost,
                best_cost,
                current_alpha,
                stagnation,
                elite_pool !== nothing ? pool_size(elite_pool) : 0,
                elapsed
            )
        end

        # Stagnation restart
        if stagnation > config.max_iterations ÷ 4
            verbose && @info @sprintf(
                "Stagnation detected (%d iter without improvement) — partial restart",
                stagnation
            )
            restart_rng = new_rng()
            restart_arr = lower .+ (upper .- lower) .* rand(restart_rng, num_vars)
            half = get_half(num_vars)
            for i in (half + 1):num_vars
                lo = ceil(Int, lower[i])
                hi = floor(Int, upper[i])
                if lo <= hi
                    restart_arr[i] = Float64(clamp(round(Int, restart_arr[i]), lo, hi))
                else
                    restart_arr[i] = clamp(restart_arr[i], lower[i], upper[i])
                end
            end
            restart_arr = local_search_vnd(
                cost_fn,
                restart_arr,
                num_vars;
                max_iter = config.vnd_iterations,
                lower,
                upper,
                cache,
                deadline,
            )
            restart_cost = cost_fn(restart_arr)
            restart_arr, restart_cost = ils_search(
                restart_arr,
                restart_cost,
                num_vars,
                cost_fn,
                config;
                lower,
                upper,
                cache,
                deadline,
            )
            if restart_cost < best_cost
                best_cost = restart_cost
                best_solution = copy(restart_arr)
            end
            if config.use_elite_pool && elite_pool !== nothing
                add!(elite_pool, restart_arr, restart_cost)
            end
            stagnation = 0
        end

        # Early stopping
        if conv_monitor !== nothing &&
           conv_monitor.no_improve_count >= config.early_stop_threshold
            verbose && @info @sprintf(
                "EARLY STOP: %d iterations without improvement",
                conv_monitor.no_improve_count
            )
            termination_msg = "early stop triggered"
            actual_nit = iteration + 1
            break
        end
    end

    if verbose
        elapsed = time() - start_time
        @info @sprintf(
            "GRASP-ILS-VND-PR end | best=%.4f | stagnation=%d | t=%.2fs",
            best_cost,
            stagnation,
            elapsed
        )
        if cache !== nothing
            stats = cache_stats(cache)
            @info @sprintf(
                "Cache Stats: %d hits, %d misses, rate=%.1f%%, size=%d",
                stats["hits"],
                stats["misses"],
                stats["hit_rate"],
                stats["size"]
            )
        end
    end

    meta = cache !== nothing ? cache_stats(cache) : Dict{String, Any}()
    return best_solution, best_cost, actual_nit, termination_msg, meta
end
