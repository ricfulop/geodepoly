import numpy as np

from geodepoly.numeric import newton, dk, companion_roots


def test_newton_simple():
    # x^2 - 2 => root near sqrt(2)
    coeffs = [-2.0, 0.0, 1.0]
    x = newton(coeffs, x0=1.0, steps=20)
    assert abs(x - np.sqrt(2)) < 1e-8


def test_dk_matches_numpy_for_quadratic():
    coeffs = [-1.0, 0.0, 1.0]  # x^2 - 1 => roots +-1
    roots = sorted(dk(coeffs, iters=200), key=lambda z: (round(z.real, 6), round(z.imag, 6)))
    assert abs(roots[0] + 1) < 1e-6 and abs(roots[1] - 1) < 1e-6


def test_companion_roots():
    coeffs = [-6.0, 11.0, -6.0, 1.0]
    roots = sorted(companion_roots(coeffs), key=lambda z: (round(z.real, 6), round(z.imag, 6)))
    assert [round(r.real, 6) for r in roots] == [1.0, 2.0, 3.0]


