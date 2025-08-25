from __future__ import annotations

from geodepoly.geode import Q_cubic, SeriesOptions, solve_series


def test_q_cubic_values_are_polynomial():
    # Basic sanity: small t2,t3 produce value near 1
    v = Q_cubic(0.01, -0.005)
    assert abs(v - 1.0) < 0.1


def test_series_bootstrap_smoke():
    # Simple cubic with known root at x=1
    coeffs = [-6, 11, -6, 1]
    x = solve_series(coeffs, SeriesOptions(Fmax=16, bootstrap=True, bootstrap_passes=4))
    # residual should be reasonably small
    # Evaluate polynomial
    p = 0j
    for a in reversed(coeffs):
        p = p * x + a
    assert abs(p) < 5e-2


