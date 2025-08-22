import cmath
from geodepoly.finishers import aberth_ehrlich
from geodepoly.util import poly_eval


def coeffs_from_roots(roots):
    # Build monic coefficients high->low, then reverse to low->high
    c = [1.0]
    for r in roots:
        c = [0.0] + c
        for k in range(len(c)-1):
            c[k] -= c[k+1]*r
    return [complex(x) for x in c[::-1]]


def test_double_root_and_simple_root():
    roots_true = [1.0, 1.0, 2.0]
    coeffs = coeffs_from_roots(roots_true)
    roots = aberth_ehrlich(coeffs, iters=200, tol=1e-12, restarts=3)
    assert max(abs(poly_eval(coeffs, z)) for z in roots) < 1e-6


def test_clustered_pair():
    roots_true = [1.0+1e-6, 1.0-1e-6, -1.5]
    coeffs = coeffs_from_roots(roots_true)
    roots = aberth_ehrlich(coeffs, iters=300, tol=1e-12, restarts=3)
    assert max(abs(poly_eval(coeffs, z)) for z in roots) < 1e-6


