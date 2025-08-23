from __future__ import annotations
from typing import List


def characteristic_coeffs_faddeev_lev_errier(A) -> List[complex]:
    """Return coefficients of the characteristic polynomial of A via Faddeev–LeVerrier.
    Returns [c0, c1, ..., c_n] low-to-high with c_n = 1 for monic p(λ)=det(λI - A).
    """
    import numpy as np
    A = np.array(A, dtype=complex)
    n = A.shape[0]
    I = np.eye(n, dtype=complex)
    # Faddeev–LeVerrier using B_0=0, c_0=1, and c_k = -(1/k) tr(A B_k)
    Bk = np.zeros_like(A)
    c = [0j]*(n+1)
    c[0] = 1.0 + 0j
    for k in range(1, n+1):
        Bk = A @ Bk + c[k-1] * I
        c[k] = - (np.trace(A @ Bk) / k)
    # Polynomial: det(λI - A) = λ^n + c1 λ^{n-1} + ... + cn
    coeffs_high_to_low = [1.0+0j] + [complex(c[k]) for k in range(1, n+1)]
    # Reverse to low->high
    coeffs = list(reversed(coeffs_high_to_low))
    # reverse to low->high
    return coeffs


def solve_eigs(A) -> List[complex]:
    """Compute eigenvalues by forming the characteristic polynomial and calling geodepoly solver."""
    from .solver import solve_all
    coeffs = characteristic_coeffs_faddeev_lev_errier(A)
    return solve_all(coeffs, method="hybrid", resum="auto")


