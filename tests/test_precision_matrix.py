import numpy as np
from geodepoly.solver import solve_all
from geodepoly.util import poly_eval


def test_precision_matrix():
    coeffs = [1, 0, -7, 6]
    for tol in [1e-8, 1e-10, 1e-12, 1e-14]:
        roots = solve_all(coeffs, method="hybrid", resum="auto", tol=tol)
        res = max(abs(poly_eval(coeffs, z)) for z in roots)
        # residual should roughly track tol scale (allow slack factor)
        assert res < 100*tol

