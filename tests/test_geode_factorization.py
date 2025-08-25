from __future__ import annotations

from geodepoly.series import geode_factorize
from geodepoly.formal import FormalSeries


def test_geode_identity_degree_4():
    # Build S,S1,G up to degree 4 and check (S-1) and S1*G coefficients match
    S, S1, G = geode_factorize(order=4, tmax=5)
    SG = (S1 * G).truncate_total_degree(4)
    Sm1 = (S + FormalSeries({(): -1.0 + 0.0j}, var_names=S.vars)).truncate_total_degree(4)
    # Compare sparse supports
    supp = set(tuple(m) for m in Sm1.support()) | set(tuple(m) for m in SG.support())
    for m in supp:
        assert abs(Sm1.coeff(m) - SG.coeff(m)) < 1e-12


