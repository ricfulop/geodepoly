from __future__ import annotations

from typing import Iterable, List, Sequence

from ...solver import solve_all


def ar_roots(ar_coeffs: Sequence[complex], method: str = "hybrid") -> List[complex]:
    """Return poles of an AR process given AR coefficients.

    For AR(p): x_t + a1 x_{t-1} + ... + ap x_{t-p} = e_t, the Z-transform
    characteristic polynomial is 1 + a1 z^{-1} + ... + ap z^{-p} = 0.
    We convert to a polynomial in y = 1/z: 1 + a1 y + ... + ap y^p = 0 and
    solve for y, then return z = 1/y.
    """
    coeffs = [1.0 + 0.0j] + [complex(a) for a in ar_coeffs]
    roots_y = solve_all(coeffs, method=method)
    roots_z = []
    for y in roots_y:
        roots_z.append(1.0 / y if y != 0 else complex("inf"))
    return roots_z


__all__ = ["ar_roots"]


