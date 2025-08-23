from __future__ import annotations
from typing import List, Dict, Optional

from .util import shift_expand
from .resummation import (
    eval_series_plain,
    eval_series_pade,
    eval_series_borel,
    eval_series_borel_pade,
    eval_series_auto,
)


def series_seed_step(coeffs, center, max_order=24, resum: Optional[str] = None):
    """
    One analytic step around `center`:
      q(y) = p(center + y) = a0 + a1 y + a2 y^2 + ...
      Let F(y) = y + sum_{k>=2} beta_k y^k, beta_k = a_k/a1 (if a1!=0).
      Invert F by Lagrange to get inverse coefficients g_m and set y ≈ sum g_m t^m, t = -a0/a1.
    Returns (y_est, a0, a1, ok_flag).  `resum` selects resummation for the inverse series evaluation.
    """
    q = shift_expand(coeffs, center)
    a0 = complex(q[0])
    a1 = complex(q[1]) if len(q) >= 2 else 0j
    if abs(a1) == 0:
        return 0j, a0, a1, False

    # Build beta dict beta[k] = a_k / a1
    beta = {}
    for k in range(2, min(len(q), max_order + 2)):
        beta[k] = q[k] / a1

    # Compute inverse-series coefficients g_m via Lagrange:
    g = inverseseries_g_coeffs(beta, max_order=max_order)

    # Evaluate y(t) with optional resummation
    t = -a0 / a1
    if resum is None:
        y = eval_series_plain(g, t)
    elif resum == "pade":
        y = eval_series_pade(g, t)
    elif resum == "borel":
        y = eval_series_borel(g, t)
    elif resum in ("borel-pade", "bp"):
        y = eval_series_borel_pade(g, t)
    elif resum == "auto":
        y = eval_series_auto(g, t)
    else:
        y = eval_series_plain(g, t)
    # early-bail: if |t| large, don't trust the step
    if abs(t) > 0.85:
        return 0j, a0, a1, False
    return y, a0, a1, True


def inverseseries_g_coeffs(beta: Dict[int, complex], max_order: int) -> List[complex]:
    """
    Compute coefficients g_m (m=1..max_order) of the compositional inverse of
    F(y) = y + sum_{k>=2} beta[k] y^k, using Lagrange inversion:
      g_m = (1/m) * [y^{m-1}] (1 / F'(y))^m,
    where F'(y) = 1 + sum_{m>=1} a_m y^m, a_m = (m+1)*beta[m+1].
    """
    # Build U(y) = sum_{m>=1} a_m y^m up to degree max_order-1
    a = [0j] * (max_order)  # a[0] unused
    for m in range(1, max_order):
        b = beta.get(m + 1, 0)
        a[m] = (m + 1) * b

    # Convolution helpers
    def series_mul(x, y, deg):
        out = [0j] * (deg + 1)
        lx, ly = len(x), len(y)
        for i in range(min(lx - 1, deg) + 1):
            xi = x[i] if i < lx else 0
            if xi == 0:
                continue
            maxj = min(deg - i, ly - 1)
            for j in range(maxj + 1):
                yj = y[j] if j < ly else 0
                if yj == 0:
                    continue
                out[i + j] += xi * yj
        return out

    deg = max_order - 1
    U = [0j] * (deg + 1)
    for m in range(1, deg + 1):
        U[m] = a[m] if m < len(a) else 0

    # Precompute U^j
    U_pows = []
    one = [0j] * (deg + 1)
    one[0] = 1
    U_pows.append(one)
    if deg >= 1:
        U_pows.append(U)
    for j in range(2, max_order):
        U_pows.append(series_mul(U_pows[-1], U, deg))

    from math import comb

    g = [0j] * max_order
    for m in range(1, max_order + 1):
        target_deg = m - 1
        coeff = 0j
        for j in range(0, m):
            C_mj = ((-1) ** j) * comb(m + j - 1, j)  # (-1)^j * (m+j-1 choose j)
            coeff += C_mj * (
                U_pows[j][target_deg] if target_deg < len(U_pows[j]) else 0
            )
        g[m - 1] = coeff / m
    return g
