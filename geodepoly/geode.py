from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, MutableMapping, Sequence, List

from .hyper_catalan import evaluate_hyper_catalan
from .series import geode_factorize, series_bootstrap
from .series_solve import inverseseries_g_coeffs, series_one_root


@dataclass
class SeriesOptions:
    Fmax: int = 6
    use_geode: bool = False
    bootstrap: bool = False
    bootstrap_passes: int = 1
    t_guard: float = 0.6


def map_t_from_poly(coeffs: Sequence[complex]) -> Dict[int, complex]:
    """
    Map polynomial coefficients a0 + a1 x + a2 x^2 + ... to Theorem 4 variables t_k.

    t_k = (a0^(k-1) * a_k) / (a1^k), for k >= 2. Requires a1 != 0.
    """
    if len(coeffs) < 2:
        raise ValueError("At least two coefficients required (a0, a1, ...)")
    a0 = complex(coeffs[0])
    a1 = complex(coeffs[1])
    if a1 == 0:
        raise ValueError("a1 cannot be zero for t_k mapping")
    out: Dict[int, complex] = {}
    for k, ak in enumerate(coeffs[2:], start=2):
        if ak != 0:
            out[k] = (a0 ** (k - 1) * complex(ak)) / (a1 ** k)
    return out


def _eval_formal_series(series, t_values: Mapping[int, complex]) -> complex:
    """Evaluate a multivariate FormalSeries at numeric t_k values.

    - series: geodepoly.formal.FormalSeries
    - t_values: mapping k>=2 -> complex
    """
    total = 0.0 + 0.0j
    # series.support() returns tuples of exponents in order of variables.
    # geode_factorize uses var_names = ("t2","t3",...)
    # so exponent index i corresponds to k = 2 + i
    for mono in series.support():
        coeff = series.coeff(mono)
        if coeff == 0:
            continue
        term = coeff
        for i, e in enumerate(mono):
            if e:
                k = 2 + i
                term *= t_values.get(k, 0) ** e
        total += term
    return total


def S_eval(t: Mapping[int, complex], Fmax: int, use_geode: bool = False) -> complex:
    """Evaluate the Hyper-Catalan generating function S at t_k values.

    If use_geode is True, evaluate via factorization S - 1 = S1 * G built up to
    total degree Fmax, then return 1 + S1(t) * G(t). Otherwise, directly sum
    the hyper-catalan series up to weighted degree Fmax.
    """
    # Guard near convergence boundary
    if any(abs(v) > 1e6 for v in t.values()):
        raise ValueError("t_k values appear divergent; check scaling")
    if not use_geode:
        return evaluate_hyper_catalan(dict(t), max_weight=int(Fmax))
    # Factorized path
    tmax = max(t.keys(), default=5)
    S, S1, G = geode_factorize(order=int(max(1, Fmax)), tmax=int(max(2, tmax)))
    s1_val = _eval_formal_series(S1, t)
    g_val = _eval_formal_series(G, t)
    return 1.0 + 0.0j + s1_val * g_val


def eval_S_via_geode(t: Mapping[int, complex], Fmax: int) -> complex:
    return S_eval(t, Fmax=Fmax, use_geode=True)


def Q_cubic(t2: complex, t3: complex) -> complex:
    """One-line cubic approximant Q(t2, t3) (Theorem 10 shape).

    Polynomial in t2, t3 up to modest degree capturing the Bi–Tri slice.
    Coefficients follow the reference spec.
    """
    return (
        1
        + (t2 + t3)
        + (2 * t2**2 + 5 * t2 * t3 + 3 * t3**2)
        + (5 * t3**2 + 21 * t2**2 * t3 + 28 * t2 * t3**2 + 12 * t3**3)
    )


def solve_series(coeffs: Sequence[complex], opts: SeriesOptions) -> complex:
    """Compute a single root using series-based bootstrapping.

    - Builds an initial center x0 = 0 and takes 1 analytic step via series.
    - If opts.bootstrap: runs `bootstrap_passes` Horner-shift rounds using
      existing `series_bootstrap` utility. Series order is opts.Fmax.
    Returns the candidate x.
    """
    # Basic single-seed bootstrap around zero. Existing solver uses richer heuristics;
    # this API keeps the surface area small and deterministic.
    rounds = max(1, int(opts.bootstrap_passes if opts.bootstrap else 1))
    # Prefer robust series-only seed finder
    try:
        x = series_one_root(list(coeffs), center=None, max_order=int(opts.Fmax), boots=rounds, tol=1e-12, refine=False)
    except Exception:
        x = series_bootstrap(coeffs, x0=0.0 + 0.0j, series_order=int(opts.Fmax), rounds=rounds)
    return complex(x)


__all__ = [
    "SeriesOptions",
    "map_t_from_poly",
    "S_eval",
    "eval_S_via_geode",
    "Q_cubic",
    "solve_series",
]


def series_reversion_coeffs(a: Dict[int, complex], order: int) -> List[complex]:
    """Return coefficients of the inverse y(t) of F(y) = y + sum_{m>=1} a[m] y^{m+1}.

    This matches Lagrange inversion used elsewhere: given beta_k = a_{k-1}, we compute
    g_m via inverseseries_g_coeffs and return [g1, ..., g_order].
    """
    beta = {k: a.get(k - 1, 0.0 + 0.0j) for k in range(2, order + 2)}
    g = inverseseries_g_coeffs(beta, max_order=max(1, int(order)))
    return [complex(x) for x in g]


def bring_radical_series(t: complex, d: int = 5, terms: int = 20) -> complex:
    """Truncated Bring/Eisenstein-style series for the root of y - t - y^d = 0.

    Solves F(y) = y + (-1)*y^d with driving t = +t. We compute the inverse of
    F(y) = y + sum_{k>=2} beta_k y^k where only beta_d = -1 is nonzero, then
    evaluate y(t) ≈ sum_{m=1..terms} g_m t^m.
    """
    if d < 3:
        raise ValueError("d must be >= 3 for Bring-style radical")
    beta = {k: 0.0 + 0.0j for k in range(2, terms + 2)}
    beta[d] = -1.0 + 0.0j
    g = inverseseries_g_coeffs(beta, max_order=max(1, int(terms)))
    # Horner evaluate sum g_m t^m
    y = 0.0 + 0.0j
    for m in range(terms, 0, -1):
        y = y * t + g[m - 1]
    y = y * t
    return complex(y)



