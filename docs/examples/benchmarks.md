# **Benchmarks — exemplos de uso**

Este documento mostra exemplos de uso das funções clássicas de otimização
disponibilizadas em `givp.benchmarks`. As funções são úteis para validar e
documentar comportamento do otimizador em problemas com ótimo conhecido.

Importação básica::

    from givp.benchmarks import (
        sphere, rosenbrock, rastrigin, ackley, griewank, schwefel,
        knapsack_dp, knapsack_penalty, qap_cost, g06
    )

Exemplo rápido — avaliar os valores ótimos conhecidos:

```python
import numpy as np

print('Sphere at zero:', sphere(np.zeros(10)))
print('Rosenbrock at ones:', rosenbrock(np.ones(5)))
print('Rastrigin at zero:', rastrigin(np.zeros(4)))
```

Knapsack DP (exemplo):

```python
values = [60, 100, 120]
weights = [10, 20, 30]
best_value, selection = knapsack_dp(values, weights, capacity=50)
print(best_value)  # => 220
print(selection)    # => array([0, 1, 1])
```

Veja também os testes em `tests/test_benchmarks.py` para exemplos adicionais
de uso e validação automatizada.
