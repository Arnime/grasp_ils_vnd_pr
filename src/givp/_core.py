"""
GRASP-ILS com VND e Path Relinking - Versão Otimizada

Este módulo implementa melhorias avançadas para o GRASP:
- VND (Variable Neighborhood Descent) para busca local mais eficaz
- Path Relinking para explorar caminhos entre soluções elite
- Pool de soluções elite para diversificação
- Cache de avaliações com LRU para evitar reavaliações
- Convergência adaptativa com restart inteligente
- Amostragem estratificada na construção
"""

# pylint: disable=too-many-lines

import time as _time_mod
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable

import numpy as np


# Module-level override for the boundary between continuous and integer vars.
# Set by ``grasp_ils_vnd`` at entry based on ``config.integer_split``; when
# ``None`` (default) every helper falls back to ``n // 2``, preserving the
# legacy behavior used by the SOG2 hydropower model where the first half of the
# vector holds continuous variables (e.g. turbined flows) and the second half
# holds integer counts (e.g. number of generators).
_INTEGER_SPLIT: int | None = None


def _get_half(n: int) -> int:
    """Return the index where integer variables begin, given vector length ``n``."""
    if _INTEGER_SPLIT is not None and 0 <= _INTEGER_SPLIT <= n:
        return _INTEGER_SPLIT
    return n // 2


def _set_integer_split(split: int | None) -> None:
    """Set the module-level integer split used by the helpers below."""
    global _INTEGER_SPLIT
    _INTEGER_SPLIT = split


def _expired(deadline: float) -> bool:
    """Retorna True se o deadline foi atingido (0 = sem limite)."""
    return deadline > 0 and _time_mod.perf_counter() >= deadline


class EvaluationCache:
    """
    Cache LRU para armazenar avaliações de soluções.

    Reduz drasticamente o número de avaliações da função objetivo,
    especialmente em buscas locais que revisitam soluções similares.

    Args:
        maxsize (int): Tamanho máximo do cache.

    Attributes:
        cache (dict): Dicionário de cache {hash: custo}.
        hits (int): Contador de acertos no cache.
        misses (int): Contador de falhas no cache.
    """

    def __init__(self, maxsize: int = 10000):
        self.maxsize = maxsize
        self.cache = {}
        self.hits = 0
        self.misses = 0
        self.insertion_order: deque = deque()

    def _hash_solution(self, solution: np.ndarray) -> int:
        """Gera hash rápido da solução arredondada (P15: rounding mais grosseiro para contínuas)."""
        half = _get_half(solution.size)
        rounded = np.empty_like(solution)
        rounded[:half] = np.round(solution[:half], decimals=3)
        rounded[half:] = np.round(solution[half:], decimals=0)
        return hash(rounded.tobytes())

    def get(self, solution: np.ndarray) -> float | None:
        """Retorna custo cached ou None se não encontrado."""
        key = self._hash_solution(solution)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None

    def put(self, solution: np.ndarray, cost: float) -> None:
        """Armazena solução e custo no cache."""
        key = self._hash_solution(solution)

        if key not in self.cache and len(self.cache) >= self.maxsize:
            oldest = self.insertion_order.popleft()
            self.cache.pop(oldest, None)

        if key not in self.cache:
            self.insertion_order.append(key)

        self.cache[key] = cost

    def clear(self) -> None:
        """Limpa o cache."""
        self.cache.clear()
        self.insertion_order = deque()
        self.hits = 0
        self.misses = 0

    def stats(self) -> dict:
        """Retorna estatísticas do cache."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "size": len(self.cache),
        }


class ConvergenceMonitor:
    """
    Monitora convergência e decide quando fazer restart.

    Rastreia histórico de melhorias e detecta estagnação prolongada,
    sugerindo restart com perturbação forte quando necessário.

    Args:
        window_size (int): Tamanho da janela para análise de tendência.
        restart_threshold (int): Iterações sem melhoria para sugerir restart.

    Attributes:
        history (list): Histórico dos melhores custos.
        no_improve_count (int): Contador de iterações sem melhoria.
    """

    def __init__(self, window_size: int = 20, restart_threshold: int = 50):
        self.window_size = window_size
        self.restart_threshold = restart_threshold
        self.history = []
        self.no_improve_count = 0
        self.best_ever = float("inf")
        self.diversity_scores = []

    def update(self, current_cost: float, elite_pool=None) -> dict:
        """
        Atualiza monitor com novo custo e retorna recomendações.

        Returns:
            dict com chaves: should_restart, should_intensify, diversity
        """
        self.history.append(current_cost)

        if current_cost < self.best_ever:
            self.best_ever = current_cost
            self.no_improve_count = 0
        else:
            self.no_improve_count += 1

        diversity = 0.0
        if elite_pool is not None and elite_pool.size() >= 2:
            solutions = [sol for sol, _ in elite_pool.get_all()]
            distances = []
            for i, sol_i in enumerate(solutions):
                for sol_j in solutions[i + 1 :]:
                    dist = np.linalg.norm(sol_i - sol_j)
                    distances.append(dist)
            diversity = float(np.mean(distances)) if distances else 0.0

        self.diversity_scores.append(diversity)

        should_restart = self.no_improve_count >= self.restart_threshold
        should_intensify = (
            self.no_improve_count >= self.restart_threshold // 2 and diversity < 0.5
        )

        return {
            "should_restart": should_restart,
            "should_intensify": should_intensify,
            "diversity": diversity,
            "no_improve_count": self.no_improve_count,
        }


class ElitePool:
    """
    Pool de soluções elite com manutenção de diversidade.

    Mantém um conjunto de soluções de alta qualidade, garantindo diversidade mínima entre elas
    (distância relativa normalizada). Permite adicionar, recuperar e limpar soluções elite.

    Args:
        max_size (int): Tamanho máximo do pool.
        min_distance (float): Distância relativa mínima normalizada entre soluções (0-1).
        lower (np.ndarray | None): Limites inferiores das variáveis (para normalização).
        upper (np.ndarray | None): Limites superiores das variáveis (para normalização).

    Attributes:
        pool (list[tuple[np.ndarray, float]]): Lista de tuplas (solução, benefício).
    """

    def __init__(
        self,
        max_size: int = 5,
        min_distance: float = 0.05,
        lower: np.ndarray | None = None,
        upper: np.ndarray | None = None,
    ):
        self.max_size = max_size
        self.min_distance = min_distance
        self.pool: list[tuple[np.ndarray, float]] = []
        self._range = None
        if lower is not None and upper is not None:
            self._range = np.maximum(upper - lower, 1e-12)

    def _relative_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calcula distância relativa normalizada média entre duas soluções."""
        if self._range is not None:
            return float(np.mean(np.abs(a - b) / self._range))
        return float(np.linalg.norm(a - b))

    def add(self, solution: np.ndarray, benefit: float) -> bool:
        """
        Adiciona solução ao pool se for boa o suficiente e diversa.

        Returns:
            True se a solução foi adicionada
        """
        solution = np.array(solution, dtype=float)

        for elite_sol, _ in self.pool:
            distance = self._relative_distance(solution, elite_sol)
            if distance < self.min_distance:
                return False

        if len(self.pool) < self.max_size:
            self.pool.append((solution.copy(), benefit))
            self.pool.sort(key=lambda x: x[1])
            return True

        if benefit < self.pool[-1][1]:
            self.pool[-1] = (solution.copy(), benefit)
            self.pool.sort(key=lambda x: x[1])
            return True

        return False

    def get_best(self) -> tuple[np.ndarray, float]:
        """Retorna a melhor solução do pool."""
        if not self.pool:
            raise ValueError("Pool vazio")
        return self.pool[0]

    def get_all(self) -> list[tuple[np.ndarray, float]]:
        """Retorna todas as soluções do pool."""
        return self.pool.copy()

    def size(self) -> int:
        """Retorna o tamanho atual do pool."""
        return len(self.pool)

    def clear(self):
        """Limpa o pool."""
        self.pool.clear()


@dataclass
class GraspIlsVndConfig:
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


def evaluate_candidates(
    available, deps_active, current_cost, deps_matrix, deps_len, c_arr, a_arr, b
):
    """
    Avalia candidatos com base em informações de dependência e custos incrementais.

    NOTE: This helper is retained for compatibility with legacy discrete packing
    use-cases; the main GRASP flow in this module works in continuous space via
    `construct_solution_numpy` and an external `evaluator`.

    Args:
        available (np.ndarray): Índices dos pacotes disponíveis.
        deps_active (np.ndarray): Array booleano indicando dependências já ativas.
        current_cost (int): Custo total atual (legacy semantics).
        deps_matrix (np.ndarray): Matriz de dependências dos pacotes.
        deps_len (np.ndarray): Array de quantidade de dependências por pacote.
        c_arr (np.ndarray): Array de benefícios dos pacotes.
        a_arr (np.ndarray): Array de custos das dependências.
        b (int): Limite de orçamento.

    Returns:
        tuple: (ratios, incremental_costs, valid_mask)
            ratios (np.ndarray): Razão benefício/custo incremental (legacy).
            incremental_costs (np.ndarray): Custos incrementais dos candidatos.
            valid_mask (np.ndarray): Máscara booleana de candidatos válidos.
    """
    n_available = len(available)
    ratios = np.full(n_available, -np.inf, dtype=np.float32)
    inc_costs = np.zeros(n_available, dtype=np.int32)
    valid = np.zeros(n_available, dtype=np.bool_)
    for idx, i in enumerate(available):
        n_deps = deps_len[i]
        incremental_cost = 0
        if n_deps > 0:
            pkg_deps = deps_matrix[i, :n_deps]
            new_deps_mask = ~deps_active[pkg_deps]
            new_deps = pkg_deps[new_deps_mask]
            if len(new_deps) > 0:
                incremental_cost = a_arr[new_deps].sum()
        if current_cost + incremental_cost <= b:
            valid[idx] = True
            inc_costs[idx] = incremental_cost
            ratios[idx] = c_arr[i] / incremental_cost if incremental_cost > 0 else 1e9
    return ratios, inc_costs, valid


def select_rcl(valid_indices, valid_ratios, alpha):
    """
    Seleciona a Restricted Candidate List (RCL) com base nas razões benefício/custo e
    parâmetro alpha.

    Args:
        valid_indices (np.ndarray): Índices dos candidatos válidos.
        valid_ratios (np.ndarray): Razões benefício/custo dos candidatos válidos.
        alpha (float): Parâmetro de randomização (0 = guloso, 1 = aleatório).

    Returns:
        np.ndarray: Índices dos candidatos na RCL.
    """
    max_ratio = valid_ratios.max()
    min_ratio = valid_ratios.min()
    threshold = max_ratio - alpha * (max_ratio - min_ratio)
    rcl_mask = valid_ratios >= threshold
    rcl_indices = valid_indices[rcl_mask]
    if len(rcl_indices) == 0:
        n_top = max(1, int(len(valid_indices) * 0.3))
        top_idx = np.argpartition(valid_ratios, -n_top)[-n_top:]
        rcl_indices = valid_indices[top_idx]
    return rcl_indices


def _validate_bounds_and_initial(
    lower_arr: np.ndarray,
    upper_arr: np.ndarray,
    initial_guess: np.ndarray | None,
    num_vars: int,
):
    if lower_arr.shape[0] != num_vars or upper_arr.shape[0] != num_vars:
        raise ValueError("lower and upper must have length num_vars")
    if initial_guess is not None:
        if initial_guess.shape[0] != num_vars:
            raise ValueError("initial_guess must have length num_vars")
        if np.any(initial_guess <= lower_arr) or np.any(initial_guess >= upper_arr):
            raise ValueError(
                "initial_guess values must be strictly between lower and upper"
            )


def _seed_from_initial(
    chute: np.ndarray,
    num_vars: int,
    evaluator: Callable,
    lower_arr: np.ndarray,
    upper_arr: np.ndarray,
) -> np.ndarray:
    try:
        cost = evaluator(chute)
    except ValueError:
        cost = np.inf
    if np.isfinite(cost):
        return chute.copy()
    rng = np.random.default_rng()
    return lower_arr + (upper_arr - lower_arr) * rng.random(size=num_vars)


def _sample_candidate_indices(
    current_sel: np.ndarray, max_candidates: int | None, rng: np.random.Generator
) -> np.ndarray:
    """Amostra índices de candidatos de forma estratificada."""
    half = _get_half(current_sel.size)
    if max_candidates is None or max_candidates >= current_sel.size:
        return np.arange(current_sel.size)

    n_int = min(max_candidates // 2, current_sel.size - half)
    n_cont = max_candidates - n_int

    cont_vars = rng.choice(half, size=min(n_cont, half), replace=False)
    int_vars = rng.choice(np.arange(half, current_sel.size), size=n_int, replace=False)
    available = np.concatenate([cont_vars, int_vars])
    rng.shuffle(available)
    return available


def _generate_candidate_value(
    var: int,
    current_sel: np.ndarray,
    lower_arr: np.ndarray,
    upper_arr: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Gera valor candidato para uma variável."""
    cand = current_sel.copy()
    half = _get_half(current_sel.size)
    if var >= half:
        lo = int(np.ceil(lower_arr[var]))
        hi = int(np.floor(upper_arr[var]))
        if hi < lo:
            cand[var] = int(np.rint((lower_arr[var] + upper_arr[var]) / 2.0))
        else:
            cand[var] = float(rng.integers(lo, hi + 1))
    else:
        cand[var] = lower_arr[var] + (upper_arr[var] - lower_arr[var]) * rng.random()

    return cand


def _evaluate_with_cache(
    cand: np.ndarray, evaluator: Callable, cache: "EvaluationCache | None"
) -> float:
    """Avalia candidato usando cache se disponível."""
    if cache is not None:
        cached_cost = cache.get(cand)
        if cached_cost is not None:
            return cached_cost

    try:
        cost = evaluator(cand)
        if cache is not None:
            cache.put(cand, cost)
        return cost
    except ValueError:
        return np.inf


def _evaluate_candidate_costs(
    current_sel: np.ndarray,
    evaluator: Callable,
    lower_arr: np.ndarray,
    upper_arr: np.ndarray,
    max_candidates: int | None = None,
    cache: "EvaluationCache | None" = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Avalia custos de candidatos para construção gulosa com cache e amostragem estratificada.

    Args:
        current_sel: Solução parcial atual
        evaluator: Função de avaliação
        lower_arr: Limites inferiores
        upper_arr: Limites superiores
        max_candidates: Número máximo de candidatos a avaliar (None = todos)
        cache: Cache de avaliações (opcional)

    Returns:
        Tupla com índices disponíveis e custos correspondentes
    """
    rng = np.random.default_rng()
    available = _sample_candidate_indices(current_sel, max_candidates, rng)
    costs = np.full(len(available), np.inf, dtype=float)

    for i_idx, var in enumerate(available):
        cand = _generate_candidate_value(var, current_sel, lower_arr, upper_arr, rng)
        costs[i_idx] = _evaluate_with_cache(cand, evaluator, cache)

    return available, costs


def _select_from_rcl(
    costs: np.ndarray, alpha: float, rng: np.random.Generator
) -> int | None:
    valid_mask = np.isfinite(costs)
    if not np.any(valid_mask):
        return None
    valid_idx = np.nonzero(valid_mask)[0]
    valid_costs = costs[valid_idx]
    min_cost = valid_costs.min()
    max_cost = valid_costs.max()
    threshold = min_cost + alpha * (max_cost - min_cost)
    rcl_local = valid_idx[valid_costs <= threshold]
    if rcl_local.size == 0:
        rcl_local = valid_idx
    return int(rng.choice(rcl_local))


def construct_solution_numpy(
    num_vars,
    lower_arr: np.ndarray,
    upper_arr: np.ndarray,
    evaluator: Callable,
    initial_guess: np.ndarray | None,
    alpha: float,
    seed: int | None = None,
    num_candidates_per_step: int | None = None,
    cache: "EvaluationCache | None" = None,
    n_workers: int = 1,
):
    """
    Constrói uma solução inicial por amostragem aleatória (Latin-style) + avaliação.

    Gera N soluções aleatórias respeitando limites, avalia todas, e seleciona
    a melhor via RCL (Restricted Candidate List). Muito mais eficiente que a
    construção greedy coordenada-a-coordenada para espaço contínuo.

    Args:
        num_vars (int): Número de variáveis.
        lower_arr (np.ndarray): Limites inferiores.
        upper_arr (np.ndarray): Limites superiores.
        evaluator (Callable): Função de avaliação.
        initial_guess (np.ndarray | None): Solução inicial sugerida.
        alpha (float): Parâmetro de randomização para RCL.
        seed (int, optional): Semente para aleatoriedade.
        num_candidates_per_step (int, optional): Número de soluções candidatas a gerar.
        cache (EvaluationCache | None): Cache de avaliações.

    Returns:
        np.ndarray: Melhor solução construída.
    """
    rng = np.random.default_rng(seed)
    _validate_bounds_and_initial(lower_arr, upper_arr, initial_guess, num_vars)

    half = _get_half(num_vars)
    n_candidates = max(num_candidates_per_step or 10, 5)

    candidates = []
    costs = []

    # Se temos initial_guess, inclui como candidato
    if initial_guess is not None:
        seed_candidate = _seed_from_initial(
            initial_guess, num_vars, evaluator, lower_arr, upper_arr
        )
        for i in range(half, num_vars):
            seed_candidate[i] = float(int(np.rint(seed_candidate[i])))
        candidates.append(seed_candidate)
        costs.append(_evaluate_with_cache(seed_candidate, evaluator, cache))
        n_candidates -= 1

    # P10: gerar candidatas com heurística de despacho proporcional (metade)
    # e aleatórias (outra metade)
    n_heuristic = max(1, n_candidates // 2)
    n_random = n_candidates - n_heuristic

    # P10: gerar candidatas heurísticas (despacho proporcional)
    for _ in range(n_heuristic):
        sol = np.empty(num_vars, dtype=float)
        mid = (lower_arr[:half] + upper_arr[:half]) / 2.0
        span = upper_arr[:half] - lower_arr[:half]
        noise = rng.uniform(-0.15, 0.15, size=half)
        sol[:half] = np.clip(mid + noise * span, lower_arr[:half], upper_arr[:half])
        for i in range(half, num_vars):
            lo = int(np.ceil(lower_arr[i]))
            hi = int(np.floor(upper_arr[i]))
            cont_idx = i - half
            if hi > lo and span[cont_idx] > 0:
                frac = (sol[cont_idx] - lower_arr[cont_idx]) / span[cont_idx]
                target = lo + frac * (hi - lo)
                sol[i] = float(int(np.clip(np.rint(target), lo, hi)))
            else:
                sol[i] = float(
                    hi if hi >= lo else int(np.rint((lower_arr[i] + upper_arr[i]) / 2.0))
                )
        candidates.append(sol)

    # Gerar candidatas aleatórias
    for _ in range(n_random):
        sol = np.empty(num_vars, dtype=float)
        sol[:half] = lower_arr[:half] + (
            upper_arr[:half] - lower_arr[:half]
        ) * rng.random(half)
        for i in range(half, num_vars):
            lo = int(np.ceil(lower_arr[i]))
            hi = int(np.floor(upper_arr[i]))
            sol[i] = (
                float(rng.integers(lo, hi + 1))
                if hi >= lo
                else float(int(np.rint((lower_arr[i] + upper_arr[i]) / 2.0)))
            )
        candidates.append(sol)

    # P11: avaliar candidatas (paralelo se n_workers > 1, senão sequencial)
    unevaluated = candidates[len(costs) :]  # pular initial_guess já avaliado
    if n_workers > 1 and len(unevaluated) > 1:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            new_costs = list(
                executor.map(
                    lambda s: _evaluate_with_cache(s, evaluator, cache),
                    unevaluated,
                )
            )
        costs.extend(new_costs)
    else:
        for sol in unevaluated:
            costs.append(_evaluate_with_cache(sol, evaluator, cache))

    # Selecionar via RCL
    costs_arr = np.array(costs)
    chosen = _select_from_rcl(costs_arr, alpha, rng)
    if chosen is None:
        return candidates[0]
    return candidates[chosen]


def _try_integer_moves_module(
    idx: int,
    sol: np.ndarray,
    best_benefit: float,
    cost_fn: Callable,
    lower_arr: np.ndarray | None,
    upper_arr: np.ndarray | None,
):
    old = sol[idx]
    base = int(np.rint(old))
    cand_vals = (base - 1, base, base + 1)
    lo = int(np.ceil(lower_arr[idx])) if (lower_arr is not None) else -np.inf
    hi = int(np.floor(upper_arr[idx])) if (upper_arr is not None) else np.inf
    for v in cand_vals:
        if v < lo or v > hi:
            continue
        sol[idx] = float(v)
        c = cost_fn(sol)
        if c < best_benefit:
            return sol.copy(), c, True
    sol[idx] = old
    return sol, best_benefit, False


def _try_continuous_move_module(
    idx: int,
    sol: np.ndarray,
    best_benefit: float,
    cost_fn: Callable,
    rng: np.random.Generator,
    lower_arr: np.ndarray | None,
    upper_arr: np.ndarray | None,
):
    old = sol[idx]
    if lower_arr is not None and upper_arr is not None:
        span = upper_arr[idx] - lower_arr[idx]
        perturb = rng.uniform(-0.05 * span, 0.05 * span)
    else:
        perturb = rng.uniform(-0.1, 0.1)
    new_val = old + perturb
    if lower_arr is not None and upper_arr is not None:
        new_val = float(np.clip(new_val, lower_arr[idx], upper_arr[idx]))
    sol[idx] = new_val
    cost = cost_fn(sol)
    if cost < best_benefit:
        return True, cost
    sol[idx] = old
    return False, best_benefit


def _search_integer_flip_module(
    sol: np.ndarray,
    best_benefit: float,
    indices: np.ndarray,
    cost_fn: Callable,
    lower_arr: np.ndarray | None,
    upper_arr: np.ndarray | None,
    first_improvement: bool,
    deadline: float = 0.0,
) -> tuple[np.ndarray, float]:
    best_sol = sol.copy()
    best_ben = best_benefit
    for count, i in enumerate(indices):
        if count % 8 == 0 and _expired(deadline):
            break
        new_sol, new_ben, improved = _try_integer_moves_module(
            i, sol, best_ben, cost_fn, lower_arr, upper_arr
        )
        if improved:
            best_ben = new_ben
            best_sol = new_sol.copy()
            if first_improvement:
                return best_sol, best_ben
    return best_sol, best_ben


def _search_continuous_flip_module(
    sol: np.ndarray,
    best_benefit: float,
    indices: np.ndarray,
    cost_fn: Callable,
    rng: np.random.Generator,
    lower_arr: np.ndarray | None,
    upper_arr: np.ndarray | None,
    first_improvement: bool,
    deadline: float = 0.0,
) -> tuple[np.ndarray, float]:
    best_sol = sol.copy()
    best_ben = best_benefit
    for count, i in enumerate(indices):
        if count % 8 == 0 and _expired(deadline):
            break
        old_val = sol[i]
        changed, new_ben = _try_continuous_move_module(
            i, sol, best_ben, cost_fn, rng, lower_arr, upper_arr
        )
        if changed:
            best_ben = new_ben
            best_sol = sol.copy()
            if first_improvement:
                return best_sol, best_ben
        sol[i] = old_val
    return best_sol, best_ben


def _modify_indices_for_multiflip(
    solution: np.ndarray,
    indices: np.ndarray,
    rng: np.random.Generator,
    lower_arr: np.ndarray | None,
    upper_arr: np.ndarray | None,
):
    half = _get_half(solution.size)
    old_vals = solution[indices].copy()

    int_mask = indices >= half
    int_indices = indices[int_mask]
    if int_indices.size > 0:
        base = np.rint(solution[int_indices]).astype(int)
        deltas = rng.integers(-1, 2, size=int_indices.size)
        new_vals = base + deltas
        if lower_arr is not None and upper_arr is not None:
            lo = np.ceil(lower_arr[int_indices]).astype(int)
            hi = np.floor(upper_arr[int_indices]).astype(int)
            new_vals = np.clip(new_vals, lo, hi)
        solution[int_indices] = new_vals.astype(float)

    cont_mask = indices < half
    cont_indices = indices[cont_mask]
    if cont_indices.size > 0:
        if lower_arr is not None and upper_arr is not None:
            spans = upper_arr[cont_indices] - lower_arr[cont_indices]
            perturb = rng.uniform(-0.05, 0.05, size=cont_indices.size) * spans
        else:
            perturb = rng.uniform(-0.1, 0.1, size=cont_indices.size)
        new_vals = solution[cont_indices] + perturb
        if lower_arr is not None and upper_arr is not None:
            new_vals = np.clip(
                new_vals, lower_arr[cont_indices], upper_arr[cont_indices]
            )
        solution[cont_indices] = new_vals

    return old_vals


def _perturb_index(
    perturbed: np.ndarray,
    idx: int,
    strength: int,
    rng: np.random.Generator,
    lower_arr: np.ndarray | None,
    upper_arr: np.ndarray | None,
):
    half = _get_half(perturbed.size)
    old = perturbed[idx]
    if idx >= half:
        lo = int(np.ceil(lower_arr[idx])) if lower_arr is not None else None
        hi = int(np.floor(upper_arr[idx])) if upper_arr is not None else None
        step = max(1, int(round(strength / 2)))
        delta = int(rng.integers(-step, step + 1))
        new_val = int(np.rint(old)) + delta
        if lo is not None and hi is not None:
            new_val = int(np.clip(new_val, lo, hi))
        perturbed[idx] = float(new_val)
    else:
        if lower_arr is not None and upper_arr is not None:
            span = upper_arr[idx] - lower_arr[idx]
            # P15: amplitude aumentada para ±15% para diversificação efetiva
            delta = rng.uniform(-0.15 * span, 0.15 * span)
            perturbed[idx] = float(np.clip(old + delta, lower_arr[idx], upper_arr[idx]))
        else:
            delta = rng.normal(scale=0.12 * (abs(old) + 1e-6))
            perturbed[idx] = old + delta


def _create_cached_cost_fn(
    cost_fn: Callable, cache: "EvaluationCache | None"
) -> Callable:
    """Cria wrapper de função de custo com cache."""

    def cached_cost_fn(sol):
        if cache is not None:
            cached = cache.get(sol)
            if cached is not None:
                return cached
        cost = cost_fn(sol)
        if cache is not None:
            cache.put(sol, cost)
        return cost

    return cached_cost_fn


def _try_neighborhoods(
    cached_cost_fn: Callable,
    solution: np.ndarray,
    current_benefit: float,
    num_vars: int,
    use_first_improvement: bool,
    iteration: int,
    no_improve_flip_limit: int,
    lower_arr: np.ndarray | None,
    upper_arr: np.ndarray | None,
    sensitivity: np.ndarray | None = None,
    deadline: float = 0.0,
) -> tuple[np.ndarray, float, bool]:
    """Tenta vizinhanças 1-opt, 2-opt, cascade, temporal e multiflip.

    Retorna (solution, benefit, improved).
    """
    if _expired(deadline):
        return solution, current_benefit, False

    new_solution, new_benefit = _neighborhood_flip(
        cached_cost_fn,
        solution,
        current_benefit,
        num_vars,
        use_first_improvement,
        seed=None,
        lower_arr=lower_arr,
        upper_arr=upper_arr,
        sensitivity=sensitivity,
        deadline=deadline,
    )
    if new_benefit < current_benefit:
        return new_solution, new_benefit, True

    if _expired(deadline):
        return solution, current_benefit, False

    new_solution, new_benefit = _neighborhood_swap(
        cached_cost_fn,
        solution,
        current_benefit,
        num_vars,
        use_first_improvement,
        lower_arr=lower_arr,
        upper_arr=upper_arr,
        deadline=deadline,
    )
    if new_benefit < current_benefit:
        return new_solution, new_benefit, True

    if _expired(deadline):
        return solution, current_benefit, False

    # P13: vizinhanças domain-aware — cascata e bloco temporal
    new_solution, new_benefit = _neighborhood_cascade(
        cached_cost_fn,
        solution,
        current_benefit,
        num_vars,
        use_first_improvement,
        lower_arr=lower_arr,
        upper_arr=upper_arr,
        deadline=deadline,
    )
    if new_benefit < current_benefit:
        return new_solution, new_benefit, True

    if _expired(deadline):
        return solution, current_benefit, False

    new_solution, new_benefit = _neighborhood_temporal_block(
        cached_cost_fn,
        solution,
        current_benefit,
        num_vars,
        use_first_improvement,
        lower_arr=lower_arr,
        upper_arr=upper_arr,
        deadline=deadline,
    )
    if new_benefit < current_benefit:
        return new_solution, new_benefit, True

    if iteration % no_improve_flip_limit == 0:
        if _expired(deadline):
            return solution, current_benefit, False
        new_solution, new_benefit = _neighborhood_multiflip(
            cached_cost_fn,
            solution,
            current_benefit,
            num_vars,
            k=no_improve_flip_limit,
            seed=None,
            lower_arr=lower_arr,
            upper_arr=upper_arr,
            deadline=deadline,
        )
        if new_benefit < current_benefit:
            return new_solution, new_benefit, True

    return solution, current_benefit, False


def local_search_vnd(
    cost_fn: Callable,
    solution: np.ndarray,
    num_vars: int,
    max_iter: int = 300,
    use_first_improvement: bool = True,
    no_improve_limit: int = 5,
    no_improve_flip_limit: int = 3,
    lower_arr: np.ndarray | None = None,
    upper_arr: np.ndarray | None = None,
    cache: "EvaluationCache | None" = None,
    deadline: float = 0.0,
) -> np.ndarray:
    """
    Executa busca local usando Variable Neighborhood Descent (VND),
    alternando entre vizinhanças 1-opt, 2-opt e multi-flip com cache.

    Args:
        cost_fn (Callable): Função de custo a maximizar.
        solution (np.ndarray): Solução inicial.
        num_vars (int): Número de variáveis.
        max_iter (int): Máximo de iterações.
        use_first_improvement (bool): Se True, aceita primeira melhoria encontrada.
        no_improve_limit (int): Limite de iterações sem melhoria.
        no_improve_flip_limit (int): Frequência para multi-flip.
        cache: Cache de avaliações (opcional).

    Returns:
        np.ndarray: Solução melhorada.
    """
    solution = np.array(solution, dtype=float)
    cached_cost_fn = _create_cached_cost_fn(cost_fn, cache)
    current_benefit = cached_cost_fn(solution)

    # P9: rastrear sensibilidade das variáveis para priorização
    sensitivity = np.zeros(num_vars, dtype=float)

    iteration = 0
    no_improve_count = 0

    while iteration < max_iter and no_improve_count < no_improve_limit:
        if _expired(deadline):
            break
        iteration += 1
        old_benefit = current_benefit
        old_solution = solution.copy()
        solution, current_benefit, improved = _try_neighborhoods(
            cached_cost_fn,
            solution,
            current_benefit,
            num_vars,
            use_first_improvement,
            iteration,
            no_improve_flip_limit,
            lower_arr,
            upper_arr,
            sensitivity=sensitivity,
            deadline=deadline,
        )

        if improved:
            no_improve_count = 0
            # P9: atualizar sensibilidade — variáveis que mudaram recebem crédito
            changed_mask = np.abs(solution - old_solution) > 1e-12
            improvement = old_benefit - current_benefit
            sensitivity[changed_mask] += improvement
            # Decaimento exponencial para não ficar preso em variáveis antigas
            sensitivity *= 0.9
        else:
            no_improve_count += 1

    return solution


def _execute_neighborhood(
    idx: int,
    cost_fn: Callable,
    solution: np.ndarray,
    current_benefit: float,
    num_vars: int,
    first_improvement: bool,
    lower_arr: np.ndarray | None,
    upper_arr: np.ndarray | None,
    sensitivity: np.ndarray | None,
    deadline: float = 0.0,
) -> tuple[np.ndarray, float]:
    """Executa uma vizinhança específica pelo índice.

    Args:
        idx: Índice da vizinhança (0=flip, 1=swap, 2=cascade, 3=temporal, 4=multiflip).
        cost_fn: Função de custo.
        solution: Solução atual.
        current_benefit: Custo atual.
        num_vars: Número de variáveis.
        first_improvement: Se True, aceita primeira melhoria.
        lower_arr: Limites inferiores.
        upper_arr: Limites superiores.
        sensitivity: Scores de sensibilidade por variável (P9).

    Returns:
        tuple: (melhor_solução, melhor_benefício).
    """
    if idx == 0:
        return _neighborhood_flip(
            cost_fn,
            solution,
            current_benefit,
            num_vars,
            first_improvement,
            seed=None,
            lower_arr=lower_arr,
            upper_arr=upper_arr,
            sensitivity=sensitivity,
            deadline=deadline,
        )
    elif idx == 1:
        return _neighborhood_swap(
            cost_fn,
            solution,
            current_benefit,
            num_vars,
            first_improvement,
            lower_arr=lower_arr,
            upper_arr=upper_arr,
            deadline=deadline,
        )
    elif idx == 2:
        return _neighborhood_cascade(
            cost_fn,
            solution,
            current_benefit,
            num_vars,
            first_improvement,
            lower_arr=lower_arr,
            upper_arr=upper_arr,
            deadline=deadline,
        )
    elif idx == 3:
        return _neighborhood_temporal_block(
            cost_fn,
            solution,
            current_benefit,
            num_vars,
            first_improvement,
            lower_arr=lower_arr,
            upper_arr=upper_arr,
            deadline=deadline,
        )
    else:
        return _neighborhood_multiflip(
            cost_fn,
            solution,
            current_benefit,
            num_vars,
            k=3,
            seed=None,
            lower_arr=lower_arr,
            upper_arr=upper_arr,
            deadline=deadline,
        )


def local_search_vnd_adaptive(
    cost_fn: Callable,
    solution: np.ndarray,
    num_vars: int,
    max_iter: int = 300,
    use_first_improvement: bool = True,
    no_improve_limit: int = 5,
    lower_arr: np.ndarray | None = None,
    upper_arr: np.ndarray | None = None,
    cache: "EvaluationCache | None" = None,
    reward_factor: float = 1.0,
    decay_factor: float = 0.95,
    min_probability: float = 0.05,
) -> np.ndarray:
    """
    VND com Seleção Adaptativa de Vizinhança (Adaptive Neighborhood Selection — ANS).

    Em vez de aplicar vizinhanças em ordem fixa (flip→swap→cascade→temporal→multiflip),
    usa roulette-wheel proporcional ao sucesso histórico de cada vizinhança.
    Vizinhanças que geram melhorias recebem recompensa; todas sofrem decaimento
    exponencial para permitir readaptação ao longo da busca.

    Args:
        cost_fn: Função de custo (minimização).
        solution: Solução inicial.
        num_vars: Número de variáveis.
        max_iter: Máximo de iterações.
        use_first_improvement: Se True, aceita primeira melhoria encontrada.
        no_improve_limit: Limite de iterações sem melhoria para parar.
        lower_arr: Limites inferiores das variáveis.
        upper_arr: Limites superiores das variáveis.
        cache: Cache de avaliações (opcional).
        reward_factor: Fator de recompensa para vizinhança bem-sucedida.
        decay_factor: Fator de decaimento dos scores por iteração (0 < d < 1).
        min_probability: Probabilidade mínima de seleção por vizinhança.

    Returns:
        np.ndarray: Solução melhorada.
    """
    solution = np.array(solution, dtype=float)
    cached_cost_fn = _create_cached_cost_fn(cost_fn, cache)
    current_benefit = cached_cost_fn(solution)

    # P9: rastrear sensibilidade das variáveis para priorização
    sensitivity = np.zeros(num_vars, dtype=float)

    # ANS: scores de sucesso por vizinhança
    n_neighborhoods = 5  # flip, swap, cascade, temporal, multiflip
    scores = np.ones(n_neighborhoods, dtype=float)

    rng = np.random.default_rng()
    iteration = 0
    no_improve_count = 0

    while iteration < max_iter and no_improve_count < no_improve_limit:
        iteration += 1
        old_benefit = current_benefit
        old_solution = solution.copy()

        # ANS: calcular probabilidades via roulette-wheel
        probs = scores / scores.sum()
        probs = np.maximum(probs, min_probability)
        probs /= probs.sum()

        # Selecionar vizinhança por roulette-wheel
        neighborhood_idx = int(rng.choice(n_neighborhoods, p=probs))

        # Executar vizinhança selecionada
        new_solution, new_benefit = _execute_neighborhood(
            neighborhood_idx,
            cached_cost_fn,
            solution,
            current_benefit,
            num_vars,
            use_first_improvement,
            lower_arr,
            upper_arr,
            sensitivity,
        )

        if new_benefit < current_benefit:
            improvement = current_benefit - new_benefit
            solution = new_solution
            current_benefit = new_benefit

            # ANS: recompensar vizinhança bem-sucedida (normalizado pelo custo)
            scores[neighborhood_idx] += (
                reward_factor * improvement / max(abs(old_benefit), 1e-10)
            )

            # P9: atualizar sensibilidade
            changed_mask = np.abs(solution - old_solution) > 1e-12
            sensitivity[changed_mask] += improvement
            sensitivity *= 0.9

            no_improve_count = 0
        else:
            no_improve_count += 1

        # ANS: decaimento dos scores para permitir readaptação
        scores *= decay_factor
        scores = np.maximum(scores, 0.01)

    return solution


def _neighborhood_flip(
    cost_fn: Callable,
    solution: np.ndarray,
    current_benefit: float,
    num_vars: int,
    first_improvement: bool = True,
    seed: int | None = None,
    lower_arr: np.ndarray | None = None,
    upper_arr: np.ndarray | None = None,
    sensitivity: np.ndarray | None = None,
    deadline: float = 0.0,
) -> tuple[np.ndarray, float]:
    """
    Vizinhança 1-opt: Tenta inverter cada bit individualmente na solução.

    P9: Se sensitivity array fornecido, prioriza variáveis com maior sensibilidade
    (maior impacto histórico no custo), reduzindo avaliações desperdiçadas.

    Args:
        cost_fn (Callable): Função de custo.
        solution (np.ndarray): Solução atual.
        current_benefit (float): Benefício atual.
        num_vars (int): Número de variáveis.
        first_improvement (bool): Se True, retorna na primeira melhoria.
        seed (int, optional): Semente aleatória.
        sensitivity (np.ndarray, optional): P9 — scores de sensibilidade por variável.

    Returns:
        tuple: (melhor_solução, melhor_benefício)
    """

    rng = np.random.default_rng(seed)
    half = _get_half(num_vars)
    if sensitivity is not None and np.any(sensitivity > 0):
        # P9: ordenar por sensibilidade (maior primeiro), com ruído para exploração
        noise = rng.uniform(0, 0.1, size=num_vars) * np.max(sensitivity)
        priority = sensitivity + noise
        indices = np.argsort(-priority)
    else:
        indices = rng.permutation(num_vars)

    int_indices = indices[indices >= half]
    cont_indices = indices[indices < half]

    best_solution = solution.copy()
    best_benefit = current_benefit

    int_best_sol, int_best_ben = _search_integer_flip_module(
        solution,
        best_benefit,
        int_indices,
        cost_fn,
        lower_arr,
        upper_arr,
        first_improvement,
        deadline=deadline,
    )
    if int_best_ben < best_benefit:
        best_solution, best_benefit = int_best_sol, int_best_ben

    cont_best_sol, cont_best_ben = _search_continuous_flip_module(
        solution,
        best_benefit,
        cont_indices,
        cost_fn,
        rng,
        lower_arr,
        upper_arr,
        first_improvement,
        deadline=deadline,
    )
    if cont_best_ben < best_benefit:
        best_solution, best_benefit = cont_best_sol, cont_best_ben

    return best_solution, best_benefit


def _neighborhood_swap(
    cost_fn: Callable,
    solution: np.ndarray,
    current_benefit: float,
    num_vars: int,
    first_improvement: bool = True,
    max_attempts: int = 50,
    lower_arr: np.ndarray | None = None,
    upper_arr: np.ndarray | None = None,
    deadline: float = 0.0,
) -> tuple[np.ndarray, float]:
    """
    Vizinhança de bloco: perturba todas as variáveis de uma mesma usina/hora simultaneamente.

    Para o domínio hidrelétrico, modifica conjuntamente variáveis de vazão e geradores
    de um mesmo time-step para uma usina, capturando correlações que a busca 1-opt ignora.

    Args:
        cost_fn (Callable): Função de custo.
        solution (np.ndarray): Solução atual.
        current_benefit (float): Benefício atual.
        num_vars (int): Número de variáveis.
        first_improvement (bool): Se True, retorna na primeira melhoria.
        max_attempts (int): Número máximo de tentativas.
        lower_arr (np.ndarray | None): Limites inferiores.
        upper_arr (np.ndarray | None): Limites superiores.

    Returns:
        tuple: (melhor_solução, melhor_benefício)
    """
    best_solution = solution.copy()
    best_benefit = current_benefit
    rng = np.random.default_rng()
    half = _get_half(num_vars)
    if half <= 0 or half >= num_vars:
        # Neighborhood requires both continuous and integer halves.
        return best_solution, best_benefit
    # Calcula qtd_usinas e tempo a partir de num_vars: half = qtd_usinas * tempo
    # Assume estrutura: [vazao(usina0,t0), vazao(usina0,t1), ..., vazao(usina1,t0), ...
    #                     gens(usina0,t0), gens(usina0,t1), ...]
    # Tenta inferir usinas/tempo; fallback para pares aleatórios se impossível
    for _ in range(max_attempts):
        if _expired(deadline):
            break
        # Selecionar um bloco: uma variável contínua + sua correspondente inteira
        cont_idx = rng.integers(0, half)
        int_idx = cont_idx + half

        old_cont = solution[cont_idx]
        old_int = solution[int_idx]

        # Perturbar ambas simultaneamente
        if lower_arr is not None and upper_arr is not None:
            span_cont = upper_arr[cont_idx] - lower_arr[cont_idx]
            solution[cont_idx] = float(
                np.clip(
                    old_cont + rng.uniform(-0.08 * span_cont, 0.08 * span_cont),
                    lower_arr[cont_idx],
                    upper_arr[cont_idx],
                )
            )
            lo_int = int(np.ceil(lower_arr[int_idx]))
            hi_int = int(np.floor(upper_arr[int_idx]))
            new_int = int(np.rint(old_int)) + int(rng.integers(-1, 2))
            solution[int_idx] = float(np.clip(new_int, lo_int, hi_int))
        else:
            solution[cont_idx] = old_cont + rng.uniform(-0.1, 0.1)
            solution[int_idx] = float(int(np.rint(old_int)) + int(rng.integers(-1, 2)))

        cost = cost_fn(solution)
        if cost < best_benefit:
            best_benefit = cost
            best_solution = solution.copy()
            if first_improvement:
                return best_solution, best_benefit

        solution[cont_idx] = old_cont
        solution[int_idx] = old_int

    return best_solution, best_benefit


def _neighborhood_multiflip(
    cost_fn: Callable,
    solution: np.ndarray,
    current_benefit: float,
    num_vars: int,
    k: int = 3,
    max_attempts: int = 50,
    seed: int | None = None,
    lower_arr: np.ndarray | None = None,
    upper_arr: np.ndarray | None = None,
    deadline: float = 0.0,
) -> tuple[np.ndarray, float]:
    """
    Vizinhança k-opt: Inverte k bits simultaneamente na solução.

    Args:
        cost_fn (Callable): Função de custo.
        solution (np.ndarray): Solução atual.
        current_benefit (float): Benefício atual.
        num_vars (int): Número de variáveis.
        k (int): Número de bits a inverter.
        max_attempts (int): Número máximo de tentativas.
        seed (int, optional): Semente aleatória.

    Returns:
        tuple: (melhor_solução, melhor_benefício)
    """
    best_solution = solution.copy()
    best_benefit = current_benefit
    rng = np.random.default_rng(seed)

    for _ in range(max_attempts):
        if _expired(deadline):
            break
        indices = rng.choice(num_vars, size=min(k, num_vars), replace=False)
        old_vals = _modify_indices_for_multiflip(
            solution, indices, rng, lower_arr, upper_arr
        )
        cost = cost_fn(solution)
        if cost < best_benefit:
            best_benefit = cost
            best_solution = solution.copy()
        solution[indices] = old_vals

    return best_solution, best_benefit


def _neighborhood_cascade(
    cost_fn: Callable,
    solution: np.ndarray,
    current_benefit: float,
    num_vars: int,
    first_improvement: bool = True,
    max_attempts: int = 30,
    lower_arr: np.ndarray | None = None,
    upper_arr: np.ndarray | None = None,
    deadline: float = 0.0,
) -> tuple[np.ndarray, float]:
    """
    P13: Vizinhança por cascata — perturba todas as horas de uma mesma usina.

    Estrutura: half = n_usinas * n_horas.
    Contínuas [0..half): usina0_h0, usina0_h1, ..., usina1_h0, ...
    Inteiras [half..num_vars): mesma ordem.
    Perturba todas as variáveis de uma usina com amplitude correlacionada.
    """
    best_solution = solution.copy()
    best_benefit = current_benefit
    rng = np.random.default_rng()
    half = _get_half(num_vars)
    if half <= 0 or half >= num_vars:
        return best_solution, best_benefit
    # Inferir estrutura: 3 usinas, 24 horas (CERAN)
    n_usinas = 3
    n_horas = half // n_usinas
    if n_horas * n_usinas != half:
        return best_solution, best_benefit  # fallback se estrutura não bate

    for _ in range(max_attempts):
        if _expired(deadline):
            break
        usina = rng.integers(0, n_usinas)
        start = usina * n_horas
        end = start + n_horas

        old_cont = solution[start:end].copy()
        old_int = solution[half + start : half + end].copy()

        # Perturbação correlacionada: mesma direção para todas as horas
        if lower_arr is not None and upper_arr is not None:
            span = upper_arr[start:end] - lower_arr[start:end]
            base_delta = rng.uniform(-0.05, 0.05)
            noise = rng.uniform(-0.02, 0.02, size=n_horas)
            solution[start:end] = np.clip(
                old_cont + (base_delta + noise) * span,
                lower_arr[start:end],
                upper_arr[start:end],
            )
            # Inteiras: ajustar geradores de acordo com variação de vazão
            int_start = half + start
            for h in range(n_horas):
                lo = int(np.ceil(lower_arr[int_start + h]))
                hi = int(np.floor(upper_arr[int_start + h]))
                delta_int = 1 if base_delta > 0 else -1 if base_delta < 0 else 0
                new_val = int(np.rint(old_int[h])) + delta_int
                solution[int_start + h] = float(np.clip(new_val, lo, hi))

        cost = cost_fn(solution)
        if cost < best_benefit:
            best_benefit = cost
            best_solution = solution.copy()
            if first_improvement:
                return best_solution, best_benefit

        solution[start:end] = old_cont
        solution[half + start : half + end] = old_int

    return best_solution, best_benefit


def _neighborhood_temporal_block(
    cost_fn: Callable,
    solution: np.ndarray,
    current_benefit: float,
    num_vars: int,
    first_improvement: bool = True,
    max_attempts: int = 30,
    lower_arr: np.ndarray | None = None,
    upper_arr: np.ndarray | None = None,
    deadline: float = 0.0,
) -> tuple[np.ndarray, float]:
    """
    P13: Vizinhança temporal — perturba um bloco contíguo de horas para todas as usinas.

    Move 3-6 horas consecutivas em direção coordenada, capturando efeitos temporais
    como rampas de carga e restrições de volume inter-período.
    """
    best_solution = solution.copy()
    best_benefit = current_benefit
    rng = np.random.default_rng()
    half = _get_half(num_vars)
    if half <= 0 or half >= num_vars:
        return best_solution, best_benefit
    n_usinas = 3
    n_horas = half // n_usinas
    if n_horas * n_usinas != half:
        return best_solution, best_benefit

    for _ in range(max_attempts):
        if _expired(deadline):
            break
        block_size = rng.integers(3, min(7, n_horas + 1))
        t_start = rng.integers(0, n_horas - block_size + 1)
        t_end = t_start + block_size

        old_vals = solution.copy()

        # Aplicar perturbação ao bloco temporal para todas as usinas
        base_delta = rng.uniform(-0.04, 0.04)
        for u in range(n_usinas):
            offset = u * n_horas
            for t in range(t_start, t_end):
                ci = offset + t
                ii = half + ci
                if lower_arr is not None and upper_arr is not None:
                    span = upper_arr[ci] - lower_arr[ci]
                    solution[ci] = float(
                        np.clip(
                            old_vals[ci] + base_delta * span,
                            lower_arr[ci],
                            upper_arr[ci],
                        )
                    )
                    lo = int(np.ceil(lower_arr[ii]))
                    hi = int(np.floor(upper_arr[ii]))
                    d = 1 if base_delta > 0 else -1 if base_delta < 0 else 0
                    solution[ii] = float(
                        np.clip(
                            int(np.rint(old_vals[ii])) + d,
                            lo,
                            hi,
                        )
                    )

        cost = cost_fn(solution)
        if cost < best_benefit:
            best_benefit = cost
            best_solution = solution.copy()
            if first_improvement:
                return best_solution, best_benefit

        np.copyto(solution, old_vals)

    return best_solution, best_benefit


def _find_best_move(
    cost_fn,
    current,
    target,
    indices,
    source,
    best_benefit,
    diff_indices,
    deadline=0.0,
):
    best_move_idx = None
    best_move_benefit = best_benefit
    for count, idx in enumerate(indices):
        if count % 5 == 0 and _expired(deadline):
            break
        if current[idx] == target[idx]:
            continue
        current[idx] = target[idx]
        cost = cost_fn(current)
        if cost < best_move_benefit:
            best_move_benefit = cost
            best_move_idx = idx
        current[idx] = source[idx] if idx in diff_indices else current[idx]
    return best_move_idx, best_move_benefit


def _path_relinking_best(
    cost_fn: Callable,
    source: np.ndarray,
    target: np.ndarray,
    diff_indices: np.ndarray,
    deadline: float = 0.0,
) -> tuple[np.ndarray, float]:
    current = source.copy().astype(float)
    best_solution = current.copy()
    best_benefit = cost_fn(current)
    indices = diff_indices.copy()
    while len(indices) > 0:
        if _expired(deadline):
            break
        best_move_idx, best_move_benefit = _find_best_move(
            cost_fn,
            current,
            target,
            indices,
            source,
            best_benefit,
            diff_indices,
            deadline=deadline,
        )
        if best_move_idx is not None:
            current[best_move_idx] = target[best_move_idx]
            indices = indices[indices != best_move_idx]
            if best_move_benefit < best_benefit:
                best_benefit = best_move_benefit
                best_solution = current.copy()
        else:
            break
    return best_solution, best_benefit


def _path_relinking_forward(
    cost_fn: Callable,
    source: np.ndarray,
    target: np.ndarray,
    diff_indices: np.ndarray,
    deadline: float = 0.0,
) -> tuple[np.ndarray, float]:
    current = source.copy().astype(float)
    best_solution = current.copy()
    best_benefit = cost_fn(current)
    for count, idx in enumerate(diff_indices):
        if count % 5 == 0 and _expired(deadline):
            break
        current[idx] = target[idx]
        cost = cost_fn(current)
        if cost < best_benefit:
            best_benefit = cost
            best_solution = current.copy()
    return best_solution, best_benefit


def path_relinking(
    cost_fn: Callable,
    source: np.ndarray,
    target: np.ndarray,
    strategy: str = "best",
    seed: int | None = None,
    deadline: float = 0.0,
) -> tuple[np.ndarray, float]:
    """
    Executa Path Relinking entre duas soluções, modificando um bit por vez na direção da
    solução destino.

    Args:
        cost_fn (Callable): Função de custo.
        source (np.ndarray): Solução origem.
        target (np.ndarray): Solução destino.
        strategy (str): 'best' (melhor a cada passo) ou 'forward' (todos os passos).
        seed (int, optional): Semente aleatória.

    Returns:
        tuple: (melhor_solução_no_caminho, melhor_benefício)
    """
    source = np.array(source, dtype=float)
    target = np.array(target, dtype=float)
    diff_indices = np.where(np.abs(source - target) > 1e-9)[0]
    if len(diff_indices) == 0:
        return source.copy(), cost_fn(source)

    # Limitar a top-K variáveis mais diferentes para evitar O(n²) no path relinking
    max_pr_vars = 25
    if len(diff_indices) > max_pr_vars:
        diffs = np.abs(source[diff_indices] - target[diff_indices])
        top_k_local = np.argpartition(diffs, -max_pr_vars)[-max_pr_vars:]
        diff_indices = diff_indices[top_k_local]

    rng = np.random.default_rng(seed)
    rng.shuffle(diff_indices)
    if strategy == "best":
        return _path_relinking_best(
            cost_fn, source, target, diff_indices, deadline=deadline
        )
    else:
        return _path_relinking_forward(
            cost_fn, source, target, diff_indices, deadline=deadline
        )


def bidirectional_path_relinking(
    cost_fn: Callable,
    sol1: np.ndarray,
    sol2: np.ndarray,
    deadline: float = 0.0,
) -> tuple[np.ndarray, float]:
    """
    Executa Path Relinking bidirecional entre duas soluções, explorando ambos os caminhos.

    Args:
        cost_fn (Callable): Função de custo.
        sol1 (np.ndarray): Primeira solução.
        sol2 (np.ndarray): Segunda solução.

    Returns:
        tuple: Melhor solução encontrada e seu benefício.
    """
    best1, cost1 = path_relinking(
        cost_fn, sol1, sol2, strategy="forward", deadline=deadline
    )

    if _expired(deadline):
        return best1, cost1

    best2, cost2 = path_relinking(
        cost_fn, sol2, sol1, strategy="forward", deadline=deadline
    )

    if cost1 <= cost2:
        return best1, cost1
    else:
        return best2, cost2


def perturb_solution_numpy(
    solution: np.ndarray,
    num_vars: int,
    strength: int = 4,
    seed: int | None = None,
    lower_arr: np.ndarray | None = None,
    upper_arr: np.ndarray | None = None,
) -> np.ndarray:
    """
    Aplica perturbação à solução, invertendo 'strength' bits aleatórios.

    Args:
        solution (np.ndarray): Solução original.
        num_vars (int): Número de variáveis.
        strength (int): Intensidade da perturbação (quantos bits inverter).
        seed (int, optional): Semente aleatória.

    Returns:
        np.ndarray: Solução perturbada.
    """
    perturbed = solution.copy().astype(float)
    rng = np.random.default_rng(seed)
    # P15: perturbação mais agressiva — num_vars//5 variáveis para escapar ótimos locais
    n_perturb = min(max(strength, num_vars // 5), num_vars)
    indices = rng.choice(num_vars, size=n_perturb, replace=False)
    for idx in indices:
        _perturb_index(perturbed, idx, strength, rng, lower_arr, upper_arr)
    return perturbed


def get_current_alpha(iteration: int, config: GraspIlsVndConfig) -> float:
    """Calcula alpha adaptativo baseado no progresso das iterações."""
    if config.adaptive_alpha:
        progress = iteration / max(1, config.max_iterations - 1)
        current_alpha = (
            config.alpha_min + (config.alpha_max - config.alpha_min) * progress
        )
        current_alpha += float(np.random.default_rng().uniform(-0.02, 0.02))
        return float(np.clip(current_alpha, config.alpha_min, config.alpha_max))
    else:
        return config.alpha


def ils_search(
    solution: np.ndarray,
    current_cost: float,
    num_vars: int,
    cost_fn: Callable,
    config: GraspIlsVndConfig,
    lower_arr: np.ndarray | None = None,
    upper_arr: np.ndarray | None = None,
    cache: "EvaluationCache | None" = None,
    deadline: float = 0.0,
) -> tuple[np.ndarray, float]:
    """
    Executa Iterated Local Search (ILS) sobre a solução, aplicando perturbações e busca local.

    Args:
        solution (np.ndarray): Solução inicial.
        current_cost (float): Custo atual (minimização).
        num_vars (int): Número de variáveis.
        cost_fn (Callable): Função de custo.
        config (GraspIlsVndConfig): Configuração do algoritmo.
        lower_arr (np.ndarray, optional): Limites inferiores das variáveis.
        upper_arr (np.ndarray, optional): Limites superiores das variáveis.
        cache (EvaluationCache, optional): Cache de avaliações.

    Returns:
        tuple: (solução final, custo final)
    """
    best_solution = solution.copy()
    best_cost = current_cost
    for ils_iter in range(config.ils_iterations):
        if _expired(deadline):
            break
        # P12: amplitude progressiva suave (5%→12%) em vez de dobrar na metade
        progress = ils_iter / max(1, config.ils_iterations - 1)
        adaptive_strength = max(
            config.perturbation_strength,
            int(config.perturbation_strength * (1.0 + progress)),
        )
        perturbed = perturb_solution_numpy(
            solution, num_vars, strength=adaptive_strength,
            lower_arr=lower_arr, upper_arr=upper_arr,
        )
        perturbed = local_search_vnd(
            cost_fn,
            perturbed,
            num_vars,
            max_iter=config.vnd_iterations // 2,
            lower_arr=lower_arr,
            upper_arr=upper_arr,
            cache=cache,
            deadline=deadline,
        )
        perturbed_cost = cost_fn(perturbed)
        # P15: aceitar pior com probabilidade maior (max 25%) para escapar ótimos locais
        temperature = 1.0 - progress
        accept_worse = np.random.default_rng().random() < temperature * 0.25
        if perturbed_cost < current_cost or accept_worse:
            solution = perturbed
            current_cost = perturbed_cost
        # Sempre manter a melhor solução global
        if current_cost < best_cost:
            best_cost = current_cost
            best_solution = solution.copy()
    # Retornar a melhor encontrada, não a última aceita
    return best_solution, best_cost


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
    except RuntimeError as e:
        if verbose:
            print(f"iteration_callback error at iter {iter_idx}: {e}")


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
            return cached
        cost = cost_fn(sol)
        cache.put(sol, cost)
        return cost
    return cost_fn(sol)


def _print_iteration_status(
    verbose: bool,
    iter_idx: int,
    max_iterations: int,
    construction_cost: float,
    best_cost: float,
    original_best: float,
) -> None:
    """Imprime status da iteração."""
    if not verbose:
        return

    iter_str = f"Iter {iter_idx + 1:3d}/{max_iterations}"
    is_new_best = construction_cost < original_best

    if is_new_best:
        print(f"{iter_str}: Custo={construction_cost:>12.2f}")
    else:
        print(f"{iter_str}: Custo={best_cost:>12.2f}")


def _run_iteration_step(
    iter_idx: int,
    cost_fn: Callable,
    num_vars: int,
    lower_arr: np.ndarray,
    upper_arr: np.ndarray,
    initial_guess: np.ndarray | None,
    config: GraspIlsVndConfig,
    callbacks: tuple[
        Callable | None,
        ElitePool | None,
        "EvaluationCache | None",
        "ConvergenceMonitor | None",
    ],
    verbose: bool,
    state: tuple[float, np.ndarray, int],
    deadline: float = 0.0,
) -> tuple[float, np.ndarray, int]:
    """Executa uma iteração do GRASP (construção, busca local, path relinking).

    Retorna os valores atualizados (best_benefit, best_solution, stagnation).
    """
    iteration_callback, elite_pool, cache, conv_monitor = callbacks
    best_cost, best_solution, stagnation = state

    current_alpha = get_current_alpha(iter_idx, config)
    sol = construct_solution_numpy(
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
        verbose, iter_idx, config.max_iterations, construction_cost, best_cost, state[0]
    )

    # P15: escalonamento reativo à estagnação — restart parcial
    if stagnation > config.max_iterations // 4:
        if verbose:
            print(
                f"  🔄 Estagnação detectada ({stagnation} iter sem melhoria) — restart parcial"
            )
        # Reiniciar de solução aleatória para diversificação real
        rng = np.random.default_rng()
        initial_arr = lower_arr + (upper_arr - lower_arr) * rng.random(size=num_vars)
        # Arredondar variáveis inteiras (segunda metade)
        half = _get_half(num_vars)
        initial_arr[half:] = np.rint(initial_arr[half:])
        restart_cost = cost_fn(initial_arr)
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
    config: "GraspIlsVndConfig",
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
    config: "GraspIlsVndConfig",
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
        config (GraspIlsVndConfig): Configuração do algoritmo.
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
    config: "GraspIlsVndConfig",
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
    config: "GraspIlsVndConfig",
    verbose: bool,
) -> bool:
    """Verifica se deve fazer early stopping."""
    if conv_monitor is None:
        return False

    if conv_monitor.no_improve_count >= config.early_stop_threshold:
        if verbose:
            print(
                f"\n⏹️  EARLY STOP: {conv_monitor.no_improve_count} iterações sem melhoria"
            )
        return True
    return False


def _print_cache_stats(cache: "EvaluationCache | None", verbose: bool) -> None:
    """Imprime estatísticas do cache."""
    if verbose and cache is not None:
        stats = cache.stats()
        print(
            f"\n📊 Cache Stats: {stats['hits']} hits, {stats['misses']} misses, "
            f"taxa={stats['hit_rate']:.1f}%, tamanho={stats['size']}"
        )


def grasp_ils_vnd(
    cost_fn: Callable,
    num_vars: int,
    config: "GraspIlsVndConfig" = GraspIlsVndConfig(),
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
        config (GraspIlsVndConfig, optional): Configuração do algoritmo.

    Returns:
        tuple: (melhor solução como lista binária, benefício da solução)
    """
    if lower is None or upper is None:
        raise ValueError("lower and upper bounds must be provided")
    lower_arr = np.array(lower, dtype=float)
    upper_arr = np.array(upper, dtype=float)

    if np.any(upper_arr <= lower_arr):
        raise ValueError("each element of upper must be strictly greater than lower")

    # Configure the continuous/integer split for the duration of this run.
    _set_integer_split(config.integer_split)

    if initial_guess is not None:
        initial_arr = np.array(initial_guess, dtype=float)
    else:
        rng = np.random.default_rng()
        initial_arr = lower_arr + (upper_arr - lower_arr) * rng.random(size=num_vars)

    elite_pool, cache, conv_monitor = _initialize_optimization_components(
        config, lower_arr, upper_arr
    )

    # P14: warm start — avaliar initial_guess e inserir no elite pool
    best_solution = np.zeros(num_vars, dtype=float)
    best_cost = float("inf")
    stagnation = 0

    if initial_guess is not None and elite_pool is not None:
        init_cost = cost_fn(initial_arr)
        elite_pool.add(initial_arr.copy(), init_cost)
        if init_cost < best_cost:
            best_cost = init_cost
            best_solution = initial_arr.copy()
        if verbose:
            print(f"[P14 warm-start] initial_guess cost = {init_cost:.2f}")

    _start_time = _time_mod.perf_counter()
    _deadline = (_start_time + config.time_limit) if config.time_limit > 0 else 0.0

    for iteration in range(config.max_iterations):
        if _expired(_deadline):
            if verbose:
                print(
                    f"\n⏱️  TIME LIMIT: {config.time_limit:.0f}s "
                    f"atingido na iteração {iteration + 1}"
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
            deadline=_deadline,
        )

        if _check_early_stopping(conv_monitor, config, verbose):
            break

    _print_cache_stats(cache, verbose)

    return best_solution.tolist(), best_cost

