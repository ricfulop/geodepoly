from __future__ import annotations

from typing import Iterable, List, Sequence

from ...solver import solve_all


def charpoly_roots(coeffs: Sequence[complex], method: str = "hybrid") -> List[complex]:
    """Solve a characteristic polynomial's roots.

    Expects coefficients in ascending order a0 + a1 x + ... + an x^n.
    Simply forwards to the library solver with a sensible default method.
    """
    return solve_all(list(coeffs), method=method)


__all__ = ["charpoly_roots"]


