import numpy as np
from geodepoly.solver import solve_all


def test_conjugate_symmetry_real_coeffs():
    # Real coefficients => single non-real roots must appear in conjugate pairs
    coeffs = [1.0, 0.0, 0.0, -1.0]  # x^3 - 1: roots 1, e^{±2πi/3}
    roots = solve_all(coeffs, method="hybrid", resum="auto")
    # Count non-real roots and check conjugate pairing
    nonreal = [z for z in roots if abs(z.imag) > 1e-8]
    assert len(nonreal) % 2 == 0
    # For each nonreal root, its conjugate should be present
    for z in nonreal:
        assert any(abs(z.conjugate() - w) < 1e-6 for w in roots)

