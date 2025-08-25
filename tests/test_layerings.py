from __future__ import annotations

from geodepoly.layerings import vertex_layering, edge_layering, face_layering
from math import comb


def test_layerings_smoke():
    vals = {2: 0.1, 3: 0.02}
    SV = vertex_layering(vals, Vmax=4)
    SE = edge_layering(vals, Emax=4)
    SF = face_layering(vals, Fmax=4)
    assert len(SV) == 5 and len(SE) == 5 and len(SF) == 5
    # Each series starts at 1.0 at level 0
    assert abs(SV[0] - 1.0) < 1e-12
    assert abs(SE[0] - 1.0) < 1e-12
    assert abs(SF[0] - 1.0) < 1e-12
    # monotone increasing in level for positive t-values
    assert all(SV[i].real <= SV[i + 1].real + 1e-12 for i in range(4))
    assert all(SE[i].real <= SE[i + 1].real + 1e-12 for i in range(4))
    assert all(SF[i].real <= SF[i + 1].real + 1e-12 for i in range(4))


def test_layerings_oeis_checks_small():
    # On t2-only slice, coefficients along vertex layering correspond to Catalans
    # S(t2) = sum_m C_m t2^m, so partial sums yield sum_{m<=L} C_m t2^m.
    vals = {2: 0.01}
    SV = vertex_layering(vals, Vmax=6)
    # Compare coefficients by finite differences at t2->0
    # Approximate C_m via successive differences divided by t2^m
    t2 = vals[2]
    # Recover first few coefficients by undetermined coefficients using small t2
    # SV[L] ~ 1 + sum_{m=1..L} C_m t2^m
    C_est = []
    prev = 1.0
    for L in range(1, 6):
        delta = (SV[L] - prev).real
        C_est.append(delta / (t2 ** L))
        prev = SV[L]
    # Catalan numbers for n=1..5
    catalans = [1, 2, 5, 14, 42]
    for ce, c in zip(C_est[:5], catalans):
        assert abs(ce - c) < 1e-1


