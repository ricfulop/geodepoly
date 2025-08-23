from __future__ import annotations
from math import factorial, comb
from typing import Dict, Iterable, List


def hyper_catalan_coefficient(m_counts: Dict[int, int]) -> int:
    """
    Return the hyper-Catalan coefficient for a given multi-index m_counts.

    The generating function S[t2,t3,...] has coefficients
      coef(m) = ( (2 m2 + 3 m3 + 4 m4 + ...)!)
                 / ( (1 + m2 + 2 m3 + 3 m4 + ...)! * Π_k m_k! ).

    m_counts: mapping k -> m_k for k >= 2 with nonnegative integers m_k.
    Returns a Python int (arbitrary precision).
    """
    if not m_counts:
        # zero multi-index → numerator 0! / (1! * 1) = 1
        return 1
    # Validate keys and values
    for k, m in m_counts.items():
        if k < 2:
            raise ValueError("Indices must be >= 2 for t_k variables")
        if m < 0:
            raise ValueError("Counts m_k must be nonnegative")

    total_vertices = sum(k * m for k, m in m_counts.items())
    layered_faces = 1 + sum((k - 1) * m for k, m in m_counts.items())

    num = factorial(total_vertices)
    den = factorial(layered_faces)
    for m in m_counts.values():
        den *= factorial(m)
    return num // den


def _iterate_multi_indices(
    keys: List[int], max_weight: int
) -> Iterable[Dict[int, int]]:
    """
    Iterate all nonnegative integer assignments m_k over given keys
    such that sum_k k*m_k <= max_weight. Includes the zero assignment.
    """
    keys = sorted(keys)

    def backtrack(idx: int, remaining: int, current: List[int]):
        if idx == len(keys):
            yield {keys[i]: current[i] for i in range(len(keys)) if current[i] != 0}
            return
        k = keys[idx]
        max_m = remaining // k
        for m in range(max_m + 1):
            current.append(m)
            yield from backtrack(idx + 1, remaining - k * m, current)
            current.pop()

    yield from backtrack(0, max_weight, [])


def evaluate_hyper_catalan(t_values: Dict[int, complex], max_weight: int) -> complex:
    """
    Evaluate the truncated S[t2,t3,...] at given t_k values, summing all terms
    with weighted degree sum_k k*m_k <= max_weight.

    - t_values: mapping k (>=2) -> complex value for t_k. Missing k treated as 0.
    - max_weight: positive integer cutoff for weighted degree.
    Returns a complex number approximating alpha with alpha(0)=1.
    """
    if max_weight < 0:
        raise ValueError("max_weight must be >= 0")
    if not t_values:
        return 1.0 + 0j
    keys = [k for k, v in t_values.items() if k >= 2 and v != 0]
    if not keys:
        return 1.0 + 0j

    total = 0.0 + 0j
    for m_counts in _iterate_multi_indices(keys, max_weight):
        coef = hyper_catalan_coefficient(m_counts)
        term = 1.0 + 0j
        for k, m in m_counts.items():
            term *= t_values.get(k, 0) ** m
        total += coef * term
    return total


def catalan_number(n: int) -> int:
    """Return the nth Catalan number C_n."""
    if n < 0:
        raise ValueError("n must be >= 0")
    return comb(2 * n, n) // (n + 1)


def evaluate_quadratic_slice(t2: complex, max_weight: int) -> complex:
    """
    Evaluate S with only t2 nonzero up to weighted degree `max_weight`.
    This slice should match the classical Catalan generating function for
    the quadratic equation 1 - alpha + t2 * alpha^2 = 0, selecting the branch
    alpha(0)=1.
    """
    return evaluate_hyper_catalan({2: t2}, max_weight=max_weight)
