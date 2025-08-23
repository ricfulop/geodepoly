import numpy as np
from geodepoly.solver import solve_all


def test_conjugate_symmetry_real_coeffs():
    # Real coefficients => roots come in conjugate pairs
    coeffs = [1.0, -3.0, 3.0, -1.0]  # (x-1)^3
    roots = solve_all(coeffs, method="hybrid", resum="auto")
    conj_counts = sum(1 for z in roots if any(abs(z.conjugate()-w)<1e-10 for w in roots))
    assert conj_counts == len(roots)

