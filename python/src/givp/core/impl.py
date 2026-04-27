# SPDX-FileCopyrightText: 2026 Arnaldo Mendes Pires Junior
# SPDX-License-Identifier: MIT
"""GRASP-ILS-VND-PR — Main orchestrator.

This module wires together the four metaheuristic components:
  - grasp.py      — GRASP construction phase
  - vnd.py        — Variable Neighborhood Descent (local search)
  - ils.py        — Iterated Local Search (perturbation + VND)
  - path_relinking.py — Path Relinking (intensification)

All public symbols from those modules are re-exported here so that
existing callers importing from ``givp.core.impl`` continue to work
without changes.
"""

# pylint: disable=too-many-lines

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from givp.core.cache import EvaluationCache
from givp.core.convergence import ConvergenceMonitor
from givp.core.elite import ElitePool
from givp.core.grasp import construct_grasp, get_current_alpha
from givp.core.helpers import (
    _ensure_verbose_handler,
    _expired,
    _get_half,
    _new_rng,
    _set_group_size,
    _set_integer_split,
    _time_mod,
    logger,
)
from givp.core.ils import ils_search
from givp.core.pr import bidirectional_path_relinking
from givp.core.vnd import _create_cached_cost_fn, local_search_vnd
from givp.exceptions import InvalidBoundsError


@dataclass
class GIVPConfig:
    """
    Configuração para o algoritmo GRASP-ILS com VND e Path Relinking.

    Define parâmetros de controle para iterações, randomização, busca local, perturbação,
    uso de pool elite e adaptação do parâmetro alpha.

    Atributos:
        max_iterations (int): Número máximo de iterações do GRASP.
        alpha (float): Parâmetro de randomização inicial para construção gulosa.
        vnd_iterations (int): Máximo de iterações da busca local VND.
        ils_iterations (int): Máximo de iterações do ILS.
        perturbation_strength (int): Intensidade da perturbação no ILS.
        use_elite_pool (bool): Se True, utiliza pool de soluções elite.
        elite_size (int): Tamanho do pool de soluções elite.
        path_relink_frequency (int): Frequência de execução do Path Relinking.
        adaptive_alpha (bool): Se True, adapta alpha ao longo das iterações.
        alpha_min (float): Valor mínimo de alpha.
        alpha_max (float): Valor máximo de alpha.
        num_candidates_per_step (int): Número de candidatos a avaliar por passo na construção.
        use_cache (bool): Se True, usa cache de avaliações.
        cache_size (int): Tamanho máximo do cache.
        early_stop_threshold (int): Iterações sem melhoria para early stopping.
        use_convergence_monitor (bool): Se True, monitora convergência e faz restart.
        time_limit (float): Limite de tempo em segundos (0 = sem limite).
    """

    max_iterations: int = 100
    alpha: float = 0.12
    vnd_iterations: int = 200
    ils_iterations: int = 10
    perturbation_strength: int = 4
    use_elite_pool: bool = True
    elite_size: int = 7
    path_relink_frequency: int = 8
    adaptive_alpha: bool = True
    alpha_min: float = 0.08
    alpha_max: float = 0.18
    num_candidates_per_step: int = 20
    use_cache: bool = True
    cache_size: int = 10000
    early_stop_threshold: int = 80
    use_convergence_monitor: bool = True
    n_workers: int = 1  # P11: nº de threads para avaliação paralela na construção
    time_limit: float = 0.0  # Limite de tempo em segundos (0 = sem limite)
    # Index where integer variables begin. ``None`` preserves the legacy
    # SOG2 behavior of splitting the vector in half; set it to ``num_vars``
    # for fully continuous problems or to ``0`` for fully integer problems.
    integer_split: int | None = None
    # Number of steps per group for the group/block neighbourhoods.
    # Set this when your problem has structured groups of variables, e.g.
    # group_size=24 for 3 groups of 24 time-steps each (72 continuous vars).
    # None disables the group and block neighbourhoods.
    group_size: int | None = None


def _safe_iteration_callback(
    iteration_callback: Callable | None,
    iter_idx: int,
    benefit: float,
    sol: np.ndarray,
    verbose: bool,
) -> None:
    """Call iteration callback safely and log when verbose."""
    if iteration_callback is None:
        return
    try:
        iteration_callback(iter_idx, benefit, sol)
    except Exception:  # pylint: disable=broad-except
        logger.warning(
            "iteration_callback raised at iter %d; continuing", iter_idx, exc_info=True
        )
        if verbose:
            logger.info(
                "iteration_callback error at iter %d (see warning above)", iter_idx
            )


def _handle_convergence_monitor(
    conv_monitor: "ConvergenceMonitor | None",
    best_cost: float,
    elite_pool: "ElitePool | None",
    verbose: bool,
) -> int:
    """Processa monitor de convergência e retorna nova estagnação."""
    if conv_monitor is None:
        return 0

    status = conv_monitor.update(best_cost, elite_pool)

    if status["should_restart"] and verbose:
        if elite_pool is not None and elite_pool.size() > 2:
            best_two = elite_pool.get_all()[:2]
            elite_pool.clear()
            for sol_elite, cost_elite in best_two:
                elite_pool.add(sol_elite, cost_elite)
        return 0

    return -1  # Não resetar estagnação


def _evaluate_solution_with_cache(
    sol: np.ndarray, cost_fn: Callable, cache: "EvaluationCache | None"
) -> float:
    """Avalia solução usando cache se disponível."""
    if cache is not None:
        cached = cache.get(sol)
        if cached is not None:
            return float(cached)
        cost = float(cost_fn(sol))
        cache.put(sol, cost)
        return cost
    return float(cost_fn(sol))


def _print_iteration_status(
    verbose: bool,
    iter_idx: int,
    max_iterations: int,
    construction_cost: float,
    best_cost: float,
    original_best: float,
    *,
    alpha: float | None = None,
    stagnation: int = 0,
    elite_size: int = 0,
    start_time: float | None = None,
) -> None:
    """Loga status da iteração (nível INFO quando verbose).

    Mostra: iter, marcador de melhora, custo da construção, melhor custo
    global, alpha corrente, estagnação, tamanho do pool elite e tempo
    decorrido. Linhas com novo melhor são prefixadas com ``*``.
    """
    if not verbose:
        return

    is_new_best = construction_cost < original_best
    marker = "*" if is_new_best else " "
    elapsed = _time_mod.monotonic() - start_time if start_time is not None else 0.0
    alpha_str = f"{alpha:.3f}" if alpha is not None else "  -  "
    logger.info(
        "%s iter %3d/%d | cur=%12.4f | best=%12.4f | alpha=%s "
        "| stag=%3d | elite=%2d | t=%6.2fs",
        marker,
        iter_idx + 1,
        max_iterations,
        construction_cost,
        best_cost,
        alpha_str,
        stagnation,
        elite_size,
        elapsed,
    )


def _run_iteration_step(
    iter_idx: int,
    cost_fn: Callable,
    num_vars: int,
    lower_arr: np.ndarray,
    upper_arr: np.ndarray,
    initial_guess: np.ndarray | None,
    config: GIVPConfig,
    callbacks: tuple[
        Callable | None,
        ElitePool | None,
        "EvaluationCache | None",
        "ConvergenceMonitor | None",
    ],
    verbose: bool,
    state: tuple[float, np.ndarray, int],
    deadline: float = 0.0,
    start_time: float | None = None,
) -> tuple[float, np.ndarray, int]:
    """Executa uma iteração do GRASP (construção, busca local, path relinking).

    Retorna os valores atualizados (best_benefit, best_solution, stagnation).
    """
    iteration_callback, elite_pool, cache, conv_monitor = callbacks
    best_cost, best_solution, stagnation = state

    current_alpha = get_current_alpha(iter_idx, config)
    sol = construct_grasp(
        num_vars,
        lower_arr,
        upper_arr,
        cost_fn,
        initial_guess,
        alpha=current_alpha,
        num_candidates_per_step=config.num_candidates_per_step,
        cache=cache,
        n_workers=config.n_workers,
    )
    sol = local_search_vnd(
        cost_fn,
        sol,
        num_vars,
        config.vnd_iterations,
        lower_arr=lower_arr,
        upper_arr=upper_arr,
        cache=cache,
        deadline=deadline,
    )

    # ILS: perturbação + busca local para escapar de ótimos locais
    vnd_cost = _evaluate_solution_with_cache(sol, cost_fn, cache)
    sol, vnd_cost = ils_search(
        sol,
        vnd_cost,
        num_vars,
        cost_fn,
        config,
        lower_arr=lower_arr,
        upper_arr=upper_arr,
        cache=cache,
        deadline=deadline,
    )

    cost = _evaluate_solution_with_cache(sol, cost_fn, cache)
    _safe_iteration_callback(iteration_callback, iter_idx, cost, sol, verbose)

    construction_cost = cost
    if cost < best_cost:
        best_cost = cost
        best_solution = sol.copy()
        stagnation = 0
    else:
        stagnation += 1

    if config.use_elite_pool and elite_pool is not None:
        elite_pool.add(sol, cost)

    new_stagnation = _handle_convergence_monitor(
        conv_monitor, best_cost, elite_pool, verbose
    )
    if new_stagnation >= 0:
        stagnation = new_stagnation

    best_cost, best_solution, stagnation = do_path_relinking(
        iter_idx,
        best_cost,
        best_solution,
        stagnation,
        config,
        elite_pool,
        cost_fn,
        num_vars,
        cache,
        deadline=deadline,
    )

    _print_iteration_status(
        verbose,
        iter_idx,
        config.max_iterations,
        construction_cost,
        best_cost,
        state[0],
        alpha=current_alpha,
        stagnation=stagnation,
        elite_size=elite_pool.size() if elite_pool is not None else 0,
        start_time=start_time,
    )

    # P15: escalonamento reativo à estagnação — restart parcial
    if stagnation > config.max_iterations // 4:
        if verbose:
            logger.info(
                "Estagnação detectada (%d iter sem melhoria) — restart parcial",
                stagnation,
            )
        # Reiniciar de solução aleatória para diversificação real
        rng = _new_rng()
        initial_arr = lower_arr + (upper_arr - lower_arr) * rng.random(size=num_vars)
        # Arredondar variáveis inteiras (segunda metade)
        half = _get_half(num_vars)
        initial_arr[half:] = np.rint(initial_arr[half:])
        # VND completo + ILS na solução aleatória para qualidade competitiva
        initial_arr = local_search_vnd(
            cost_fn,
            initial_arr,
            num_vars,
            max_iter=config.vnd_iterations,
            lower_arr=lower_arr,
            upper_arr=upper_arr,
            cache=cache,
            deadline=deadline,
        )
        restart_cost = cost_fn(initial_arr)
        # ILS para escapar do ótimo local da solução restart
        initial_arr, restart_cost = ils_search(
            initial_arr,
            restart_cost,
            num_vars,
            cost_fn,
            config,
            lower_arr=lower_arr,
            upper_arr=upper_arr,
            cache=cache,
            deadline=deadline,
        )
        if restart_cost < best_cost:
            best_cost = restart_cost
            best_solution = initial_arr.copy()
        # Adicionar ao elite pool para diversificar o PR
        if config.use_elite_pool and elite_pool is not None:
            elite_pool.add(initial_arr, restart_cost)
        stagnation = 0

    return best_cost, best_solution, stagnation


def _apply_path_relinking_to_pair(
    source: np.ndarray,
    target: np.ndarray,
    cached_fn: Callable,
    num_vars: int,
    config: "GIVPConfig",
    cache: "EvaluationCache | None",
    deadline: float = 0.0,
) -> tuple[np.ndarray, float]:
    """Aplica path relinking e busca local a um par de soluções."""
    pr_solution, _ = bidirectional_path_relinking(
        cached_fn, source, target, deadline=deadline
    )
    pr_solution = local_search_vnd(
        cached_fn,
        pr_solution,
        num_vars,
        max_iter=config.vnd_iterations // 2,
        cache=cache,
        deadline=deadline,
    )
    pr_cost = cached_fn(pr_solution)
    return pr_solution, pr_cost


def _process_path_relinking_pairs(
    elite_solutions,
    cost_fn,
    num_vars,
    config,
    best_cost,
    best_solution,
    stagnation,
    elite_pool,
    cache,
    deadline=0.0,
):
    """Processa pares de soluções elite com path relinking."""
    cached_fn = _create_cached_cost_fn(cost_fn, cache)

    for i in range(min(3, len(elite_solutions))):
        for j in range(i + 1, min(4, len(elite_solutions))):
            if _expired(deadline):
                break
            source = elite_solutions[i][0]
            target = elite_solutions[j][0]

            pr_solution, pr_cost = _apply_path_relinking_to_pair(
                source,
                target,
                cached_fn,
                num_vars,
                config,
                cache,
                deadline=deadline,
            )

            if pr_cost < best_cost:
                best_cost = pr_cost
                best_solution = pr_solution.copy()
                stagnation = 0

            elite_pool.add(pr_solution, pr_cost)

    return best_cost, best_solution, stagnation


def do_path_relinking(
    iteration: int,
    best_cost: float,
    best_solution: np.ndarray,
    stagnation: int,
    config: "GIVPConfig",
    elite_pool: "ElitePool | None",
    cost_fn: Callable,
    num_vars: int,
    cache: "EvaluationCache | None" = None,
    deadline: float = 0.0,
) -> tuple[float, np.ndarray, int]:
    """
    Executa Path Relinking entre pares de soluções elite, se condições forem atendidas.

    Args:
        iteration (int): Iteração atual do GRASP.
        best_benefit (float): Melhor benefício atual.
        best_solution (np.ndarray): Melhor solução atual.
        stagnation (int): Contador de estagnação.
        config (GIVPConfig): Configuração do algoritmo.
        elite_pool (ElitePool | None): Pool de soluções elite.
        cost_fn (Callable): Função de custo.
        num_vars (int): Número de variáveis.

    Returns:
        tuple: (melhor benefício, melhor solução, estagnação)
    """
    if (
        config.use_elite_pool
        and elite_pool is not None
        and iteration > 0
        and iteration % config.path_relink_frequency == 0
        and elite_pool.size() >= 2
    ):
        elite_solutions = elite_pool.get_all()
        best_cost, best_solution, stagnation = _process_path_relinking_pairs(
            elite_solutions,
            cost_fn,
            num_vars,
            config,
            best_cost,
            best_solution,
            stagnation,
            elite_pool,
            cache,
            deadline=deadline,
        )
    return best_cost, best_solution, stagnation


def _initialize_optimization_components(
    config: "GIVPConfig",
    lower_arr: np.ndarray | None = None,
    upper_arr: np.ndarray | None = None,
) -> tuple["ElitePool | None", "EvaluationCache | None", "ConvergenceMonitor | None"]:
    """Inicializa componentes de otimização (pool elite, cache, monitor)."""
    elite_pool = (
        ElitePool(max_size=config.elite_size, lower=lower_arr, upper=upper_arr)
        if config.use_elite_pool
        else None
    )
    cache = EvaluationCache(maxsize=config.cache_size) if config.use_cache else None
    conv_monitor = (
        ConvergenceMonitor(restart_threshold=config.early_stop_threshold)
        if config.use_convergence_monitor
        else None
    )
    return elite_pool, cache, conv_monitor


def _check_early_stopping(
    conv_monitor: "ConvergenceMonitor | None",
    config: "GIVPConfig",
    verbose: bool,
) -> bool:
    """Verifica se deve fazer early stopping."""
    if conv_monitor is None:
        return False

    if conv_monitor.no_improve_count >= config.early_stop_threshold:
        if verbose:
            logger.info(
                "EARLY STOP: %d iterações sem melhoria", conv_monitor.no_improve_count
            )
        return True
    return False


def _print_cache_stats(cache: "EvaluationCache | None", verbose: bool) -> None:
    """Loga estatísticas do cache."""
    if verbose and cache is not None:
        stats = cache.stats()
        logger.info(
            "Cache Stats: %d hits, %d misses, taxa=%.1f%%, tamanho=%d",
            stats["hits"],
            stats["misses"],
            stats["hit_rate"],
            stats["size"],
        )


def _prepare_bounds(
    lower: list[float] | None,
    upper: list[float] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and convert bounds to numpy arrays."""
    if lower is None or upper is None:
        raise InvalidBoundsError("lower and upper bounds must be provided")
    lower_arr = np.array(lower, dtype=float)
    upper_arr = np.array(upper, dtype=float)
    if lower_arr.shape != upper_arr.shape:
        raise InvalidBoundsError(
            f"lower (shape={lower_arr.shape}) and upper "
            f"(shape={upper_arr.shape}) must have the same shape"
        )
    if np.any(upper_arr <= lower_arr):
        bad = np.nonzero(upper_arr <= lower_arr)[0]
        raise InvalidBoundsError(
            f"each element of upper must be strictly greater than lower; "
            f"violating indices: {bad.tolist()[:10]}"
        )
    return lower_arr, upper_arr


def _prepare_initial_array(
    initial_guess: list[float] | None,
    lower_arr: np.ndarray,
    upper_arr: np.ndarray,
    num_vars: int,
) -> np.ndarray:
    """Build initial candidate either from initial_guess or random sample."""
    if initial_guess is not None:
        return np.array(initial_guess, dtype=float)
    rng = _new_rng()
    result: np.ndarray = np.asarray(
        lower_arr + (upper_arr - lower_arr) * rng.random(size=num_vars)
    )
    return result


def _maybe_apply_warm_start(
    initial_guess: list[float] | None,
    elite_pool: "ElitePool | None",
    cost_fn: Callable,
    initial_arr: np.ndarray,
    best_cost: float,
    best_solution: np.ndarray,
    verbose: bool,
) -> tuple[float, np.ndarray]:
    """Insert warm start in elite pool and update incumbent if needed."""
    if initial_guess is None or elite_pool is None:
        return best_cost, best_solution

    init_cost = cost_fn(initial_arr)
    elite_pool.add(initial_arr.copy(), init_cost)
    if init_cost < best_cost:
        best_cost = init_cost
        best_solution = initial_arr.copy()
    if verbose:
        logger.info("[P14 warm-start] initial_guess cost = %.2f", init_cost)
    return best_cost, best_solution


def _print_run_header(verbose: bool, num_vars: int, config: "GIVPConfig") -> None:
    """Log a single header line summarising the run configuration."""
    if not verbose:
        return
    _ensure_verbose_handler()
    logger.info(
        "GRASP-ILS-VND-PR start | n=%d | iters=%d | alpha=[%.3f, %.3f] "
        "| elite=%d | time_limit=%s",
        num_vars,
        config.max_iterations,
        config.alpha_min if config.adaptive_alpha else config.alpha,
        config.alpha_max if config.adaptive_alpha else config.alpha,
        config.elite_size if config.use_elite_pool else 0,
        f"{config.time_limit:.1f}s" if config.time_limit > 0 else "unlimited",
    )


def _print_run_footer(
    verbose: bool, best_cost: float, stagnation: int, start_time: float
) -> None:
    """Log a single footer line summarising the final state."""
    if not verbose:
        return
    elapsed = _time_mod.monotonic() - start_time
    logger.info(
        "GRASP-ILS-VND-PR end   | best=%.4f | stagnation=%d | t=%.2fs",
        best_cost,
        stagnation,
        elapsed,
    )


def _run_grasp_loop(
    cost_fn: Callable,
    num_vars: int,
    config: "GIVPConfig",
    lower_arr: np.ndarray,
    upper_arr: np.ndarray,
    initial_arr: np.ndarray,
    callbacks: tuple[
        Callable | None,
        "ElitePool | None",
        "EvaluationCache | None",
        "ConvergenceMonitor | None",
    ],
    verbose: bool,
    state: tuple[float, np.ndarray, int],
) -> tuple[float, np.ndarray, int]:
    """Execute main GRASP iterations until stop conditions are met."""
    iteration_callback, elite_pool, cache, conv_monitor = callbacks
    best_cost, best_solution, stagnation = state
    start_time = _time_mod.monotonic()
    deadline = (start_time + config.time_limit) if config.time_limit > 0 else 0.0

    _print_run_header(verbose, num_vars, config)

    for iteration in range(config.max_iterations):
        if _expired(deadline):
            if verbose:
                logger.info(
                    "TIME LIMIT: %.0fs atingido na iteração %d",
                    config.time_limit,
                    iteration + 1,
                )
            break

        best_cost, best_solution, stagnation = _run_iteration_step(
            iteration,
            cost_fn,
            num_vars,
            lower_arr,
            upper_arr,
            initial_arr,
            config,
            (iteration_callback, elite_pool, cache, conv_monitor),
            verbose,
            (best_cost, best_solution, stagnation),
            deadline=deadline,
            start_time=start_time,
        )

        if _check_early_stopping(conv_monitor, config, verbose):
            break

    _print_run_footer(verbose, best_cost, stagnation, start_time)

    return best_cost, best_solution, stagnation


def grasp_ils_vnd(
    cost_fn: Callable,
    num_vars: int,
    config: GIVPConfig | None = None,
    verbose: bool = False,
    iteration_callback: Callable | None = None,
    lower: list[float] | None = None,
    upper: list[float] | None = None,
    initial_guess: list[float] | None = None,
) -> tuple[list[int], float]:
    """
    Executa o algoritmo GRASP-ILS com VND e Path Relinking, retornando a melhor solução encontrada.

    Args:
        cost_fn (Callable): Função de custo a maximizar.
        num_vars (int): Número de variáveis/pacotes.
        c (list[int]): Benefícios dos pacotes.
        a (list[int]): Custos das dependências.
        deps (list[list[int]]): Lista de dependências por pacote.
        b (int): Limite de orçamento.
        config (GIVPConfig, optional): Configuração do algoritmo.

    Returns:
        tuple: (melhor solução como lista binária, benefício da solução)
    """
    if config is None:
        config = GIVPConfig()

    lower_arr, upper_arr = _prepare_bounds(lower, upper)

    # Configure the continuous/integer split for the duration of this run.
    _set_integer_split(config.integer_split)
    _set_group_size(config.group_size)

    initial_arr = _prepare_initial_array(initial_guess, lower_arr, upper_arr, num_vars)

    elite_pool, cache, conv_monitor = _initialize_optimization_components(
        config, lower_arr, upper_arr
    )

    # P14: warm start — avaliar initial_guess e inserir no elite pool
    best_solution = np.zeros(num_vars, dtype=float)
    best_cost = float("inf")
    stagnation = 0

    best_cost, best_solution = _maybe_apply_warm_start(
        initial_guess,
        elite_pool,
        cost_fn,
        initial_arr,
        best_cost,
        best_solution,
        verbose,
    )

    best_cost, best_solution, stagnation = _run_grasp_loop(
        cost_fn,
        num_vars,
        config,
        lower_arr,
        upper_arr,
        initial_arr,
        (iteration_callback, elite_pool, cache, conv_monitor),
        verbose,
        (best_cost, best_solution, stagnation),
    )

    _print_cache_stats(cache, verbose)

    return best_solution.tolist(), best_cost
