from __future__ import annotations

from typing import Mapping, Sequence

from .formal import FormalSeries
from .series_core import series_seed_step
from .util import poly_eval


def series_root(c: Sequence[complex], order: int, variant: str = "raw") -> FormalSeries:
    """Stub for the soft polynomial formula returning a formal series root.

    This is a scaffold: it returns the linear term y ≈ -a0/a1 as a one-variable series
    `y(t) = t` where t stands for -a0/a1. Future work: implement Theorems 4/7.
    """
    _ = (c, order, variant)
    # series y = t (placeholder); variable name t
    return FormalSeries({(1,): 1.0 + 0.0j}, var_names=("t",))


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
        y, a0, a1, ok = series_seed_step(c, x, max_order=int(series_order), resum="auto")
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
    """Construct S, S1, G with S − 1 = S1 · G, truncated to total degree `order`.

    Variables are named t2, t3, ..., up to `tmax` (default: 4 variables).
    This is a constructive identity to enable early tests; later replace with
    the actual Geode construction (paper mapping).
    """
    num_vars = max(1, (tmax or 5) - 1)  # t2..t{tmax} → count
    var_names = tuple(f"t{k}" for k in range(2, 2 + num_vars))

    # Build S1 = sum t_i
    S1 = FormalSeries({}, var_names=var_names)
    for i in range(num_vars):
        mono = tuple(1 if j == i else 0 for j in range(num_vars))
        S1 = S1 + FormalSeries({mono: 1.0 + 0.0j}, var_names=var_names)
    S1 = S1.truncate_total_degree(order)

    # Build a small nontrivial G with linear and quadratic terms
    G = FormalSeries({(): 1.0 + 0.0j}, var_names=var_names)
    # Linear terms with decreasing weights
    for i in range(num_vars):
        mono = tuple(1 if j == i else 0 for j in range(num_vars))
        coef = 1.0 / (i + 2)
        G = G + FormalSeries({mono: coef}, var_names=var_names)
    # Quadratic cross terms
    for i in range(num_vars):
        for j in range(i, num_vars):
            mono = tuple((1 if k == i else 0) + (1 if k == j else 0) for k in range(num_vars))
            coef = 0.1 / (1 + abs(i - j))
            G = G + FormalSeries({mono: coef}, var_names=var_names)
    G = G.truncate_total_degree(order)

    # Define S so that S − 1 = S1 * G (truncated)
    SG = (S1 * G).truncate_total_degree(order)
    S = FormalSeries({(): 1.0 + 0.0j}, var_names=var_names) + SG
    return S.truncate_total_degree(order), S1, G


