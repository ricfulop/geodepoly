from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

from .formal import FormalSeries
from .series_core import series_seed_step
from .util import poly_eval
from .series_solve import inverseseries_g_coeffs
from .hyper_catalan import hyper_catalan_coefficient
from fractions import Fraction as _Fr
import numpy as _np  # type: ignore


def series_root(c: Sequence[complex], order: int, variant: str = "raw") -> FormalSeries:
    """Return the truncated formal series y(t) for a nearby root up to `order`.

    Implements the Lagrange inversion coefficients g_m of the inverse of
    F(y) = y + sum_{k>=2} beta_k y^k with beta_k = a_k/a_1 for the recentered
    polynomial q(y) = a0 + a1 y + a2 y^2 + ... around 0. The formal series is
    y(t) = sum_{m>=1} g_m t^m with t := -a0/a1, truncated to degree `order`.
    """
    _ = variant  # reserved for future variants (raw/factorized)
    q = [complex(a) for a in c]
    if len(q) < 2 or q[1] == 0:
        # Degenerate linearization; return zero series
        return FormalSeries({}, var_names=("t",))
    a1 = complex(q[1])
    beta = {k: (q[k] / a1) for k in range(2, min(len(q), order + 2))}
    g = inverseseries_g_coeffs(beta, max_order=max(1, order))
    coeffs = {}
    for m in range(1, min(order, len(g)) + 1):
        if g[m - 1] != 0:
            coeffs[(m,)] = complex(g[m - 1])
    return FormalSeries(coeffs, var_names=("t",))


def series_bootstrap(
    coeffs: Sequence[complex],
    x0: complex,
    series_order: int,
    rounds: int,
    damping: float = 1.0,
) -> complex:
    """Shift–solve–add bootstrapping using the series inversion step.

    Uses `series_seed_step` to compute an analytic step y around the current center x,
    then updates x <- x + damping * y for a few rounds.
    """
    c = [complex(a) for a in coeffs]
    x = complex(x0)
    for _ in range(max(1, int(rounds))):
        y, a0, a1, ok = series_seed_step(
            c, x, max_order=int(series_order), resum="auto"
        )
        if (not ok) or a1 == 0:
            break
        step = damping * y
        x_candidate = x + step
        # Optional safeguard: accept if residual improves, otherwise keep original
        try:
            if abs(poly_eval(c, x_candidate)) <= abs(poly_eval(c, x)):
                x = x_candidate
            else:
                # small backoff
                x = x + 0.5 * step
        except Exception:
            x = x_candidate
    return x


def geode_factorize(order: int, tmax: int | None = None):
    """Construct S, S1, G with S − 1 = S1 · G up to total degree `order`.

    - S is built combinatorially from Hyper-Catalan coefficients:
      coef(m) = (2 m2 + 3 m3 + ...)!
                / ( (1 + m2 + 2 m3 + ...)! * Π m_k! ).
    - S1 = sum_i t_i (linear part).
    - G is solved degree-by-degree from (S−1) = S1 * G as a linear system.

    Variables are named t2, t3, ..., up to `tmax` (default: t2..t5).
    """
    num_vars = max(1, (tmax or 5) - 1)  # t2..t{tmax}
    var_names = tuple(f"t{k}" for k in range(2, 2 + num_vars))

    # Enumerate monomials of given total degree in num_vars variables
    def monomials_of_degree(deg: int) -> List[Tuple[int, ...]]:
        out: List[Tuple[int, ...]] = []
        def backtrack(i: int, remaining: int, cur: List[int]):
            if i == num_vars - 1:
                out.append(tuple(cur + [remaining]))
                return
            for e in range(remaining + 1):
                backtrack(i + 1, remaining - e, cur + [e])
        backtrack(0, deg, [])
        return out

    # Build S from Hyper-Catalan coefficients, truncated by total degree
    S_coeffs: Dict[Tuple[int, ...], complex] = {(): 1.0 + 0.0j}
    for deg in range(1, max(1, order) + 1):
        for m in monomials_of_degree(deg):
            # Map variable index to m_k (k starts at 2)
            m_counts = {2 + i: m[i] for i in range(num_vars) if m[i]}
            coef = hyper_catalan_coefficient(m_counts)
            if coef != 0:
                S_coeffs[m] = S_coeffs.get(m, 0.0) + complex(coef)
    S = FormalSeries(S_coeffs, var_names=var_names).truncate_total_degree(order)

    # S1 = sum t_i
    S1 = FormalSeries({}, var_names=var_names)
    unit_monos = [tuple(1 if j == i else 0 for j in range(num_vars)) for i in range(num_vars)]
    for mono in unit_monos:
        S1 = S1 + FormalSeries({mono: 1.0 + 0.0j}, var_names=var_names)
    S1 = S1.truncate_total_degree(order)

    # Target T = (S - 1) truncated
    T = (S + FormalSeries({(): -1.0 + 0.0j}, var_names=var_names)).truncate_total_degree(order)

    # Solve for G degree-by-degree using linear systems A x = b where
    # for deg d >= 1: b indexes monomials of degree d in T, x indexes monomials of degree d-1 in G,
    # and A[m,u] = 1 if there exists i with u + e_i = m, else 0 (since S1 has unit coefficients).
    G_coeffs: Dict[Tuple[int, ...], complex] = {}
    # constant term from degree-1 identity: T[e_i] = G[()] for all i
    deg1_monos = monomials_of_degree(1)
    if deg1_monos:
        g0 = _np.mean([T.coeff(m) for m in deg1_monos])
        if abs(g0) == 0:
            g0 = 0.0
        G_coeffs[()] = complex(g0)

    def _solve_normal_eq_fraction(A_num, b_num):
        """Solve (A^T A) x = A^T b over exact rationals using Gauss-Jordan."""
        m = len(A_num)
        n = len(A_num[0]) if m else 0
        # Build ATA and ATb as Fractions
        ATA = [[_Fr(0, 1) for _ in range(n)] for _ in range(n)]
        ATb = [_Fr(0, 1) for _ in range(n)]
        for i in range(n):
            for j in range(n):
                s = _Fr(0, 1)
                for r in range(m):
                    s += _Fr(A_num[r][i]) * _Fr(A_num[r][j])
                ATA[i][j] = s
            t = _Fr(0, 1)
            for r in range(m):
                t += _Fr(A_num[r][i]) * _Fr(b_num[r])
            ATb[i] = t
        # Gauss-Jordan elimination on [ATA | ATb]
        # Convert to augmented matrix
        aug = [row[:] + [ATb[i]] for i, row in enumerate(ATA)]
        # Forward elimination
        for col in range(n):
            # find pivot
            piv = col
            for r in range(col, n):
                if aug[r][col] != 0:
                    piv = r
                    break
            if aug[piv][col] == 0:
                continue
            if piv != col:
                aug[col], aug[piv] = aug[piv], aug[col]
            # normalize pivot row
            pivval = aug[col][col]
            for j in range(col, n + 1):
                aug[col][j] /= pivval
            # eliminate others
            for r in range(n):
                if r == col:
                    continue
                factor = aug[r][col]
                if factor == 0:
                    continue
                for j in range(col, n + 1):
                    aug[r][j] -= factor * aug[col][j]
        x = [aug[i][n] for i in range(n)]
        return x

    for d in range(2, order + 1):
        rows = monomials_of_degree(d)
        cols = monomials_of_degree(d - 1)
        if not rows or not cols:
            continue
        # Build integer A and b for exact rational solve
        A_mat: List[List[int]] = [[0 for _ in range(len(cols))] for _ in range(len(rows))]
        b_vec: List[int] = [0 for _ in range(len(rows))]
        # Fill b from T (assumed integer coefficients for S from hyper_catalan)
        for i, m in enumerate(rows):
            val = T.coeff(m)
            b_vec[i] = int(round(val.real))
        # Build A mapping
        e = unit_monos
        col_index = {u: j for j, u in enumerate(cols)}
        for i, m in enumerate(rows):
            for ei in e:
                u = tuple(m[k] - ei[k] for k in range(num_vars))
                if min(u) < 0:
                    continue
                j = col_index.get(u)
                if j is not None:
                    A_mat[i][j] += 1
        # Solve normal equations exactly
        x_frac = _solve_normal_eq_fraction(A_mat, b_vec)
        for j, u in enumerate(cols):
            val = complex(float(x_frac[j]))
            if abs(val) != 0:
                G_coeffs[u] = val

    G = FormalSeries(G_coeffs, var_names=var_names).truncate_total_degree(order)
    return S, S1, G
