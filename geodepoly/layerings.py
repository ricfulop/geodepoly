from __future__ import annotations

from typing import Callable, Dict, List, Mapping, Tuple

from .hyper_catalan import hyper_catalan_coefficient


def _enumerate_layered(
    keys: List[int],
    max_level: int,
    weight_of: Callable[[int], int],
) -> List[Dict[int, int]]:
    """Enumerate all multi-indices m_counts over given keys such that
    sum_k weight_of(k) * m_k <= max_level.

    Returns a list of dictionaries {k: m_k} with only nonzero entries.
    """
    out: List[Dict[int, int]] = []

    def backtrack(idx: int, remaining: int, current: List[int]):
        if idx == len(keys):
            m_counts = {keys[i]: current[i] for i in range(len(keys)) if current[i] != 0}
            out.append(m_counts)
            return
        k = keys[idx]
        w = max(1, int(weight_of(k)))
        m_max = remaining // w
        for m in range(m_max + 1):
            current.append(m)
            backtrack(idx + 1, remaining - w * m, current)
            current.pop()

    backtrack(0, int(max_level), [])
    return out


def _layered_values(
    t: Mapping[int, complex],
    max_level: int,
    weight_of: Callable[[int], int],
) -> List[complex]:
    keys = sorted(k for k, v in t.items() if k >= 2 and v != 0)
    if not keys:
        # Only constant term 1 for all levels
        return [1.0 + 0.0j] * (int(max_level) + 1)

    values: List[complex] = []
    for L in range(int(max_level) + 1):
        total = 0.0 + 0.0j
        for m_counts in _enumerate_layered(keys, L, weight_of):
            coef = hyper_catalan_coefficient(m_counts)
            term = 1.0 + 0.0j
            for k, m in m_counts.items():
                term *= t.get(k, 0) ** m
            total += coef * term
        values.append(total)
    return values


def vertex_layering(t: Mapping[int, complex], Vmax: int) -> List[complex]:
    """Return partial sums by vertex layering: sum m_k <= L for L=0..Vmax."""
    return _layered_values(t, Vmax, weight_of=lambda k: 1)


def edge_layering(t: Mapping[int, complex], Emax: int) -> List[complex]:
    """Return partial sums by edge layering: sum (k-1) m_k <= L for L=0..Emax."""
    return _layered_values(t, Emax, weight_of=lambda k: k - 1)


def face_layering(t: Mapping[int, complex], Fmax: int) -> List[complex]:
    """Return partial sums by face layering: 1 + sum (k-1) m_k <= 1+L ⇒ sum(k-1)m_k<=L."""
    return _layered_values(t, Fmax, weight_of=lambda k: k - 1)


__all__ = ["vertex_layering", "edge_layering", "face_layering"]


