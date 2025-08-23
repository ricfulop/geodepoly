from __future__ import annotations

from typing import List, Sequence

from .finishers import aberth_ehrlich, durand_kerner, halley_refine
from .series_solve import newton_refine


def newton(coeffs: Sequence[complex], x0: complex, steps: int = 50, tol: float = 1e-14) -> complex:
    return newton_refine([complex(a) for a in coeffs], complex(x0), steps=steps, tol=tol)


def aberth(coeffs: Sequence[complex], iters: int = 200, tol: float = 1e-14):
    c = [complex(a) for a in coeffs]
    roots = aberth_ehrlich(c, iters=iters, tol=tol, restarts=3) or []
    return [halley_refine(c, z, steps=4) for z in roots]


def dk(coeffs: Sequence[complex], iters: int = 600, tol: float = 1e-14):
    c = [complex(a) for a in coeffs]
    roots = durand_kerner(c, iters=iters, tol=tol, restarts=4) or []
    return [halley_refine(c, z, steps=4) for z in roots]


def companion_roots(coeffs: Sequence[complex]):
    import numpy as np

    a = [complex(x) for x in coeffs]
    if a[-1] == 0:
        raise ValueError("Leading coefficient is zero.")
    a = [x / a[-1] for x in a]
    n = len(a) - 1
    C = np.zeros((n, n), dtype=complex)
    C[1:, :-1] = np.eye(n - 1, dtype=complex)
    C[:, -1] = -np.array(a[:-1], dtype=complex)
    w = np.linalg.eigvals(C)
    return [complex(z) for z in w]


