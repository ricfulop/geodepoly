from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

from ...solver import solve_all


def ray_intersect_quartic(coeffs: Sequence[complex]) -> List[float]:
    """Intersect a ray parameterized by t>=0 with a quartic scalar equation p(t)=0.

    `coeffs` are polynomial coefficients a0 + a1 t + ... + a4 t^4 (or higher),
    and the function returns the sorted list of nonnegative real roots.
    """
    roots = solve_all(list(coeffs), method="hybrid")
    real_nonneg: List[float] = []
    for z in roots:
        if abs(z.imag) < 1e-9 and z.real >= -1e-12:
            real_nonneg.append(float(max(0.0, z.real)))
    real_nonneg.sort()
    return real_nonneg


__all__ = ["ray_intersect_quartic"]


