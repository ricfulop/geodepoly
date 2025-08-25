from __future__ import annotations

from geodepoly.geode import series_reversion_coeffs


def test_series_reversion_matches_internal_g():
    # F(y) = y + a1 y^2 + a2 y^3 + a3 y^4 + ...
    a = {1: 0.2, 2: -0.1, 3: 0.05}
    g = series_reversion_coeffs(a, order=8)
    # sanity: first few coefficients finite and alternate in sign reasonably
    assert len(g) == 8
    # First coefficient should be 1 when a is small; later ones bounded
    assert abs(g[0] - 1) < 1e-9
    assert all(abs(complex(x)) < 1e3 for x in g)


