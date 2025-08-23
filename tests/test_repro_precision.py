import random, math, cmath
from geodepoly.solver import solve_all
from geodepoly.util import poly_eval


def test_reproducibility_fixed_seed():
    random.seed(123)
    coeffs = [1, -3, 3, -1]  # (x-1)^3
    r1 = solve_all(coeffs, method="hybrid", resum="auto", tol=1e-12)
    random.seed(123)
    r2 = solve_all(coeffs, method="hybrid", resum="auto", tol=1e-12)
    # order-insensitive compare
    r1s = sorted(r1, key=lambda z: (round(z.real, 12), round(z.imag, 12)))
    r2s = sorted(r2, key=lambda z: (round(z.real, 12), round(z.imag, 12)))
    assert all(abs(a-b) < 1e-12 for a,b in zip(r1s, r2s))


def test_precision_control_tol():
    coeffs = [1, 0, -7, 6]
    r_loose = solve_all(coeffs, method="hybrid", resum="auto", tol=1e-8)
    r_tight = solve_all(coeffs, method="hybrid", resum="auto", tol=1e-14)
    res_loose = max(abs(poly_eval(coeffs, z)) for z in r_loose)
    res_tight = max(abs(poly_eval(coeffs, z)) for z in r_tight)
    assert res_tight <= res_loose + 1e-12

